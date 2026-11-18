"""Sequence-statistics baselines on the same windows: canonical k-mer profiles with logistic regression or
cosine k-NN, and the majority class. Any learned model should be compared with these."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .tokenizers import encode, kmer_ids, pad_codes


def canonical_index(k: int) -> np.ndarray:
    """Map each of the 4^k k-mers to the index of its canonical (lexicographically smaller strand) class."""
    ids = np.arange(4**k)
    digits = (ids[:, None] >> (2 * np.arange(k - 1, -1, -1))) & 3
    rc = ((3 - digits[:, ::-1]) << (2 * np.arange(k - 1, -1, -1))).sum(1)
    canonical = np.minimum(ids, rc)
    _, index = np.unique(canonical, return_inverse=True)
    return index


def kmer_profiles(sequences: Sequence[str], k: int = 4, batch: int = 2048) -> np.ndarray:
    """Canonical k-mer frequency profiles (n, n_canonical), counting both strands, k-mers with N ignored."""
    index = torch.from_numpy(canonical_index(k))
    width = int(index.max()) + 1
    out = np.zeros((len(sequences), width), dtype=np.float32)
    for lo in range(0, len(sequences), batch):
        codes, lengths = pad_codes([encode(s) for s in sequences[lo : lo + batch]])
        ids, _ = kmer_ids(torch.from_numpy(codes), torch.from_numpy(lengths), k)
        valid = ids >= 2
        cls = torch.where(valid, index[(ids - 2).clamp(min=0)], torch.zeros_like(ids))
        counts = torch.zeros(len(ids), width).scatter_add_(1, cls, valid.float())
        out[lo : lo + batch] = (counts / counts.sum(1, keepdim=True).clamp(min=1)).numpy()
    return out


class KmerLogistic:
    def __init__(self, k: int = 4, C: float = 1.0, max_iter: int = 2000):
        self.k = k
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=C, max_iter=max_iter)

    def fit(self, sequences: Sequence[str], labels: Sequence[str]) -> KmerLogistic:
        self.model.fit(self.scaler.fit_transform(kmer_profiles(sequences, self.k)), labels)
        return self

    def predict_proba(self, sequences: Sequence[str]) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(kmer_profiles(sequences, self.k)))

    @property
    def classes_(self) -> np.ndarray:
        return self.model.classes_


def knn_predict(
    reference: np.ndarray, labels: Sequence[str], query: np.ndarray, k: int = 15, batch: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Similarity-weighted vote among the ``k`` most cosine-similar reference rows.
    Returns (class names, (n_query, n_classes) vote shares)."""
    names, y = np.unique(np.asarray(labels), return_inverse=True)
    ref = torch.from_numpy(_normalise(reference))
    votes = np.zeros((len(query), len(names)), dtype=np.float32)
    yt = torch.from_numpy(y)
    for lo in range(0, len(query), batch):
        q = torch.from_numpy(_normalise(query[lo : lo + batch]))
        sims, idx = (q @ ref.T).topk(min(k, len(ref)), dim=1)
        w = sims.clamp(min=0) + 1e-6
        v = torch.zeros(len(q), len(names)).scatter_add_(1, yt[idx], w)
        votes[lo : lo + batch] = (v / v.sum(1, keepdim=True)).numpy()
    return names, votes


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
