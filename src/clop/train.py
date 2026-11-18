"""Training loop: AdamW with a one-cycle schedule, optional AMP and torch.compile, early stopping on validation
zero-shot macro F1, and self-describing checkpoints."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch

from .data import CaptionTable, SequenceDataset, batch_captions, loader
from .losses import clip_loss, multi_positive_loss, sigmoid_loss
from .models import CLOP

logger = logging.getLogger(__name__)
FORMAT = "clop-2"


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 0.01
    warmup: float = 0.1
    loss: str = "multi_positive"
    crop_lengths: list[int] = field(default_factory=lambda: [100, 500])  # log-uniform range; [] for no crops
    random_rc: bool = False
    patience: int = 5
    grad_clip: float = 1.0
    amp: bool = False
    compile: bool = False
    workers: int = 0
    seed: int = 0
    eval_batch_size: int = 512

    @classmethod
    def from_dict(cls, data: dict) -> TrainConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_loss(model: CLOP, batch: dict, captions: CaptionTable, loss: str, device: torch.device) -> torch.Tensor:
    codes, lengths = batch["codes"].to(device), batch["lengths"].to(device)
    if loss == "clip":  # CLOP v1: one caption row per sequence, diagonal positives only
        tokens, token_lengths = captions.tokens[batch["owner"]], captions.lengths[batch["owner"]]
        dna, text = model(codes, lengths, tokens[:, : int(token_lengths.max())].to(device), token_lengths.to(device))
        return clip_loss(dna, text, model.scale())
    tokens, token_lengths, owner = batch_captions(batch, captions)
    dna, text = model(codes, lengths, tokens.to(device), token_lengths.to(device))
    owner = owner.to(device)
    if loss == "sigmoid":
        return sigmoid_loss(dna, text, owner, model.scale(), model.logit_bias)
    return multi_positive_loss(dna, text, owner, model.scale())


class Trainer:
    def __init__(
        self, model: CLOP, captions: CaptionTable, config: TrainConfig, device: torch.device | str = "cpu", validate=None
    ):
        """``validate(model) -> dict`` returns validation metrics including ``score`` (higher is better)."""
        self.model, self.captions, self.cfg = model.to(device), captions, config
        self.device = torch.device(device)
        self.validate = validate
        self.history: list[dict] = []
        decay = [p for n, p in model.named_parameters() if p.ndim > 1]
        rest = [p for n, p in model.named_parameters() if p.ndim <= 1]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": config.weight_decay}, {"params": rest, "weight_decay": 0.0}], lr=config.lr
        )
        self.forward_model = torch.compile(model) if config.compile and hasattr(torch, "compile") else model
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp and self.device.type == "cuda")

    def fit(
        self, train: SequenceDataset, out_dir: str | Path | None = None, extra: dict | None = None, resume: bool = True
    ) -> list[dict]:
        """Train for ``cfg.epochs`` epochs. With ``out_dir`` and ``resume``, an interrupted run continues from
        ``out_dir/state.pt`` (model, optimiser, schedule, RNG and early-stopping state after the last full epoch)."""
        cfg = self.cfg
        seed_everything(cfg.seed)
        batches = loader(train, cfg.batch_size, shuffle=True, workers=cfg.workers, seed=cfg.seed)
        steps = max(len(batches) * cfg.epochs, 8)
        warmup = min(max(cfg.warmup, 2.0 / steps), 0.5)  # OneCycle needs at least one step in each phase
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=cfg.lr, total_steps=steps, pct_start=warmup, anneal_strategy="cos"
        )
        best, best_state, waited, first = -np.inf, None, 0, 1
        state_path = Path(out_dir) / "state.pt" if out_dir is not None else None
        if resume and state_path is not None and state_path.exists():
            saved = torch.load(state_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            schedule.load_state_dict(saved["schedule"])
            torch.set_rng_state(saved["rng"])
            self.history, best, best_state, waited = saved["history"], saved["best"], saved["best_state"], saved["waited"]
            first = len(self.history) + 1
            logger.info("resuming at epoch %d", first)
        stopped = waited >= cfg.patience
        for epoch in range(first, cfg.epochs + 1):
            if stopped:
                break
            self.model.train()
            batches.generator.manual_seed(cfg.seed * 1000 + epoch)  # per-epoch order, identical after a resume
            started, total, n = time.time(), 0.0, 0
            for batch in batches:
                with torch.autocast(self.device.type, enabled=self.scaler.is_enabled()):
                    loss = compute_loss(self.forward_model, batch, self.captions, cfg.loss, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                schedule.step()
                total, n = total + loss.item() * len(batch["owner"]), n + len(batch["owner"])
            row = {
                "epoch": epoch,
                "train_loss": total / max(n, 1),
                "seconds": round(time.time() - started, 1),
                "temperature": 1.0 / self.model.scale().item(),
                "lr": schedule.get_last_lr()[0],
            }
            if self.validate is not None:
                self.model.eval()
                row.update(self.validate(self.model))
            self.history.append(row)
            logger.info("epoch %d %s", epoch, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}))
            score = row.get("score", -row["train_loss"])
            if score > best:
                best, waited = score, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                waited += 1
                stopped = waited >= cfg.patience
                if stopped:
                    logger.info("early stopping at epoch %d (best score %.4f)", epoch, best)
            if state_path is not None:
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "schedule": schedule.state_dict(),
                        "rng": torch.get_rng_state(),
                        "history": self.history,
                        "best": best,
                        "best_state": best_state,
                        "waited": waited,
                    },
                    state_path,
                )
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if out_dir is not None:
            save_checkpoint(Path(out_dir) / "model.pt", self.model, self.captions, cfg, self.history, extra)
            if state_path.exists():
                state_path.unlink()
        return self.history


def save_checkpoint(
    path: str | Path, model: CLOP, captions: CaptionTable, config: TrainConfig, history: list[dict], extra: dict | None = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "model_config": model.cfg.to_dict(),
            "state_dict": model.state_dict(),
            "captions": captions.captions,
            "train_config": asdict(config),
            "history": history,
            **(extra or {}),
        },
        path,
    )
