"""Inference API: load a checkpoint, embed sequences and texts, classify by text prompts or neighbours, search."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .models import CLOP, ModelConfig
from .text import TextTokenizer, caption
from .tokenizers import encode, pad_codes


def tile_starts(length: int, window: int) -> np.ndarray:
    """Starts of windows of ``window`` bases covering ``length`` bases with half-window overlap, the last one flush with
    the end (a single window when the sequence fits)."""
    if length <= window:
        return np.zeros(1, dtype=np.int64)
    stride = max(window // 2, 1)
    return np.minimum(np.arange(-(-(length - window) // stride) + 1) * stride, length - window)


def _batches(n: int, size: int):
    for lo in range(0, n, size):
        yield slice(lo, min(lo + size, n))


class Pipeline:
    """A trained CLOP model with its tokenizer and label space.

    ``labels`` maps a label (``species|biotype``) to the caption that serves as its zero-shot prompt."""

    def __init__(
        self,
        model: CLOP,
        tokenizer: TextTokenizer,
        labels: dict[str, str] | None = None,
        device: str | torch.device = "cpu",
        meta: dict | None = None,
    ):
        self.model, self.tokenizer = model.to(device).eval(), tokenizer
        self.device = torch.device(device)
        self.labels = dict(labels or {})
        self.meta = meta or {}
        self._prototypes = None

    # ------------------------------------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> Pipeline:
        state = torch.load(path, map_location=device, weights_only=False)
        if state.get("format") != "clop-2":
            raise ValueError(
                f"{path} is not a CLOP 2 checkpoint (format={state.get('format')!r}); "
                "retrain or use `clop convert` for v1 checkpoints"
            )
        model = CLOP(ModelConfig.from_dict(state["model_config"]))
        model.load_state_dict(state["state_dict"])
        return cls(
            model,
            TextTokenizer(state["text_vocab"]),
            state.get("labels", {}),
            device,
            {k: v for k, v in state.items() if k not in ("state_dict",)},
        )

    # ------------------------------------------------------------------------------------------- embedding
    @torch.no_grad()
    def embed(self, sequences: Sequence[str], batch_size: int = 512, pooled: bool = False) -> np.ndarray:
        """L2-normalised DNA embeddings (n, d). A sequence longer than the model's window is embedded as the normalised
        mean of overlapping windows of the length the model was trained on, so long inputs stay in distribution and
        never change the embedding just by being long. Windows are sorted by length so padding stays small.
        ``pooled`` returns the DNA encoder's pooled features before its projection head instead (tile means)."""
        window = self.model.cfg.window
        tiles, groups = [], np.zeros(len(sequences), dtype=np.int64)
        for i, sequence in enumerate(sequences):
            codes = encode(sequence if window else sequence[: self.model.cfg.max_length])
            starts, size = (tile_starts(len(codes), window), window) if window else ((0,), len(codes))
            tiles.extend(codes[s : s + size] for s in starts)
            groups[i] = len(starts)
        encoder = self.model.dna.pooled if pooled else self.model.encode_dna
        order = np.argsort([len(c) for c in tiles], kind="stable")
        emb = None
        for part in _batches(len(tiles), batch_size):
            idx = order[part]
            batch, lengths = pad_codes([tiles[i] for i in idx])
            out = (
                encoder(torch.from_numpy(batch).to(self.device), torch.from_numpy(lengths).to(self.device)).float().cpu().numpy()
            )
            if emb is None:
                emb = np.empty((len(tiles), out.shape[1]), dtype=np.float32)
            emb[idx] = out
        if emb is None:
            return np.zeros((0, self.model.cfg.embed_dim), dtype=np.float32)
        if len(tiles) == len(sequences):
            return emb
        out = np.add.reduceat(emb, np.concatenate([[0], np.cumsum(groups)[:-1]]), axis=0)
        if pooled:
            return out / groups[:, None]
        return out / np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-12)

    @torch.no_grad()
    def embed_texts(self, texts: Sequence[str], batch_size: int = 1024) -> np.ndarray:
        out = []
        for part in _batches(len(texts), batch_size):
            tokens, lengths = self.tokenizer.batch(list(texts[part]))
            out.append(
                self.model.encode_text(torch.from_numpy(tokens).to(self.device), torch.from_numpy(lengths).to(self.device))
                .float()
                .cpu()
                .numpy()
            )
        return np.concatenate(out) if out else np.zeros((0, self.model.cfg.embed_dim), dtype=np.float32)

    # ------------------------------------------------------------------------------------------- zero-shot
    def set_labels(self, labels: dict[str, str]) -> None:
        self.labels, self._prototypes = dict(labels), None

    def labels_from_pairs(self, pairs: Sequence[tuple[str, str]]) -> dict[str, str]:
        return {f"{s}|{b}": caption(s, b) for s, b in pairs}

    def pool_labels(self, part: str = "biotype", extra: Sequence[str] = ()) -> None:
        """Collapse the label space to one prototype per biotype (or species): the renormalised mean of the prompts
        that share it. Species-agnostic biotype prompts suit genomes absent from training. ``extra`` biotypes absent
        from the label space get a prompt made of their biotype words alone."""
        index = 1 if part == "biotype" else 0
        names, prototypes = self.label_names, self.prototypes
        groups = sorted({n.split("|")[index] for n in names})
        pooled = [prototypes[[i for i, n in enumerate(names) if n.split("|")[index] == g]].mean(0) for g in groups]
        missing = [b for b in extra if b not in groups]
        if missing:
            pooled.extend(self.embed_texts([caption("", b) for b in missing]))
            groups.extend(missing)
        matrix = np.asarray(pooled, dtype=np.float32)
        self.labels = {g: g for g in groups}
        self._prototypes = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    @property
    def prototypes(self) -> np.ndarray:
        if self._prototypes is None:
            self._prototypes = self.embed_texts(list(self.labels.values()))
        return self._prototypes

    @property
    def label_names(self) -> list[str]:
        return list(self.labels)

    def scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Softmax over labels of the temperature-scaled cosine to each prompt (n, n_labels)."""
        logits = self.model.scale().item() * embeddings @ self.prototypes.T
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(1, keepdims=True)

    def classify(self, sequences: Sequence[str], top: int = 3) -> list[dict]:
        """Zero-shot calls: top labels, plus species and biotype marginals."""
        p = self.scores(self.embed(sequences))
        names = self.label_names
        species = [n.split("|")[0] for n in names]
        biotypes = [n.split("|")[-1] for n in names]
        out = []
        for row in p:
            order = np.argsort(-row)[:top]
            out.append(
                {
                    "top": [{"label": names[j], "score": float(row[j])} for j in order],
                    "species": _marginal(row, species, top),
                    "biotype": _marginal(row, biotypes, top),
                }
            )
        return out

    def search(self, query: str, embeddings: np.ndarray, top: int = 10) -> list[tuple[int, float]]:
        """Indices of the ``top`` sequences most similar to a free-text query."""
        sims = embeddings @ self.embed_texts([query])[0]
        order = np.argsort(-sims)[:top]
        return [(int(i), float(sims[i])) for i in order]


def _marginal(row: np.ndarray, groups: list[str], top: int) -> list[dict]:
    names, inverse = np.unique(groups, return_inverse=True)
    totals = np.bincount(inverse, weights=row, minlength=len(names))
    order = np.argsort(-totals)[:top]
    return [{"label": str(names[j]), "score": float(totals[j])} for j in order]
