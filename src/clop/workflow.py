"""End-to-end runs shared by the CLI and the benchmark scripts: split, caption, train, evaluate, save."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import CaptionTable, SequenceDataset, assign_splits, cap_per_cell, drop_rare_cells, split_records
from .evaluate import evaluate, hierarchical_metrics, length_metrics, strip_predictions
from .io import Record, summarise
from .models import CLOP, ModelConfig
from .pipeline import Pipeline
from .text import TextTokenizer, caption, record_caption
from .train import TrainConfig, Trainer

logger = logging.getLogger(__name__)


def label_space(records: Sequence[Record]) -> dict[str, str]:
    """Training cells and their prompt captions, sorted by species then biotype."""
    cells = sorted({(r.species, r.biotype, r.meta.get("lineage", "")) for r in records})
    return {f"{s}|{b}": caption(s, b, lineage=lineage) for s, b, lineage in cells}


def prepare_records(
    records: list[Record],
    split: str = "auto",
    fractions=(0.15, 0.15),
    cap: int = 0,
    min_train: int = 10,
    seed: int = 0,
    holdout: dict | None = None,
) -> dict[str, list[Record]]:
    records = assign_splits(records, split, fractions, holdout, seed)
    records = drop_rare_cells(records, min_train)
    parts = split_records(records)
    parts["train"] = cap_per_cell(parts["train"], cap, seed)
    return parts


def train_model(
    parts: dict[str, list[Record]],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    out_dir: str | Path,
    device: str = "cpu",
    use_description: bool = False,
    validation_sample: int = 3000,
    evaluate_test: bool = True,
    knn_k: int = 15,
) -> dict:
    """Train on ``parts['train']``, early-stop on validation zero-shot cell macro F1, evaluate on test.
    Writes ``model.pt``, ``history.json``, ``metrics.json`` and ``embeddings.npz`` under ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(train_cfg.seed)
    caption_fn = lambda r: record_caption(r, use_description)
    labels = label_space(parts["train"])
    tokenizer = TextTokenizer.build([caption_fn(r) for r in parts["train"]], extra=labels.values())
    captions = CaptionTable([caption_fn(r) for r in parts["train"]], tokenizer)
    model_cfg.text_vocab_size = len(tokenizer)
    if not model_cfg.window:  # longer inputs are tiled at inference into windows of the longest training input
        longest = max(len(r.sequence) for r in parts["train"])
        model_cfg.window = min(longest, max(train_cfg.crop_lengths or [longest]), model_cfg.max_length)
    model = CLOP(model_cfg)
    train_set = SequenceDataset(
        parts["train"], captions, caption_fn, model_cfg.max_length, train_cfg.crop_lengths, train_cfg.random_rc
    )
    rng = np.random.default_rng(train_cfg.seed)
    validation = [r for r in parts["validation"] if r.cell in labels]
    if len(validation) > validation_sample:
        validation = [validation[i] for i in sorted(rng.choice(len(validation), validation_sample, replace=False))]

    def validate(m: CLOP) -> dict:
        pipe = Pipeline(m, tokenizer, labels, device)
        if not validation:
            return {}
        metrics = hierarchical_metrics(
            [r.cell for r in validation], pipe.scores(pipe.embed([r.sequence for r in validation])), pipe.label_names
        )
        return {
            "val_species_f1": metrics["species_macro_f1"],
            "val_biotype_f1": metrics["biotype_macro_f1"],
            "val_cell_f1": metrics["cell_macro_f1"],
            "score": metrics["cell_macro_f1"],
        }

    extra = {
        "text_vocab": tokenizer.to_dict(),
        "labels": labels,
        "use_description": use_description,
        "data": {s: summarise(v) for s, v in parts.items()},
        "parameters": model.n_parameters(),
    }
    trainer = Trainer(model, captions, train_cfg, device, validate)
    history = trainer.fit(train_set, out, extra)
    (out / "history.json").write_text(json.dumps(history, indent=1))
    pipe = Pipeline(trainer.model, tokenizer, labels, device)
    summary = {
        "model_config": model_cfg.to_dict(),
        "train_config": asdict(train_cfg),
        "parameters": model.n_parameters(),
        "epochs_run": len(history),
        "best_epoch": int(np.argmax([h.get("score", -h["train_loss"]) for h in history])) + 1,
        "data": extra["data"],
    }
    if evaluate_test:
        reference = parts["train"]
        ref_emb = pipe.embed([r.sequence for r in reference])
        for split in ("validation", "test"):
            records = [r for r in parts[split] if r.cell in labels]
            if not records:
                continue
            result, emb = evaluate(pipe, records, reference, ref_emb, knn_k, ci=split == "test")
            summary[split] = strip_predictions(result)
            if split == "test":
                np.savez_compressed(
                    out / "embeddings.npz",
                    embeddings=emb.astype(np.float16),
                    cells=np.array([r.cell for r in records]),
                    ids=np.array([r.id for r in records]),
                    prototypes=pipe.prototypes,
                    labels=np.array(pipe.label_names),
                    predictions=np.array(result["ensemble"]["predictions"]["cell"]),
                )
                summary["length"] = length_metrics(pipe, records, seed=train_cfg.seed)
    (out / "metrics.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary
