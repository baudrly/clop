"""DNA encoding. Sequences become uint8 codes (A0 C1 G2 T3, anything else 4 = N, which also pads); k-mer token ids
are computed from codes on the fly, so every encoder shares one data path and no vocabulary has to be built."""

from __future__ import annotations

import numpy as np
import torch

N_CODE = 4
_LOOKUP = np.full(256, N_CODE, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    _LOOKUP[ord(_b)] = _LOOKUP[ord(_b.lower())] = _i
_LOOKUP[ord("U")] = _LOOKUP[ord("u")] = 3
LETTERS = np.frombuffer(b"ACGTN", dtype=np.uint8)


def encode(sequence: str) -> np.ndarray:
    """
    >>> encode("ACGTNu").tolist()
    [0, 1, 2, 3, 4, 3]
    """
    return _LOOKUP[np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)]


def decode(codes: np.ndarray) -> str:
    return LETTERS[np.minimum(codes, N_CODE)].tobytes().decode()


def pad_codes(codes: list[np.ndarray], length: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Stack variable-length code arrays into an (n, L) matrix padded with N, and their lengths."""
    lengths = np.fromiter((len(c) for c in codes), dtype=np.int64, count=len(codes))
    width = int(length or lengths.max(initial=1))
    out = np.full((len(codes), width), N_CODE, dtype=np.uint8)
    for i, c in enumerate(codes):
        out[i, : min(len(c), width)] = c[:width]
    return out, np.minimum(lengths, width)


def reverse_complement_codes(codes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse complement of padded code batches, keeping padding at the end of each row."""
    length = codes.shape[1]
    positions = torch.arange(length, device=codes.device)
    source = (lengths[:, None] - 1 - positions[None, :]).clamp(min=0)
    flipped = torch.gather(codes, 1, source)
    complemented = torch.where(flipped < N_CODE, 3 - flipped, flipped)
    return torch.where(positions[None, :] < lengths[:, None], complemented, torch.full_like(codes, N_CODE))


def kmer_ids(codes: torch.Tensor, lengths: torch.Tensor, k: int, stride: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Overlapping k-mer token ids (0 = pad, 1 = k-mer containing N, 2 + base-4 index otherwise) and token counts.

    >>> ids, n = kmer_ids(torch.tensor([[0, 1, 2, 3, 4]]), torch.tensor([5]), 2)
    >>> ids.tolist(), n.tolist()
    ([[3, 8, 13, 1]], [4])
    """
    if codes.shape[1] < k:
        codes = torch.nn.functional.pad(codes, (0, k - codes.shape[1]), value=N_CODE)
    windows = codes.long().unfold(1, k, stride)  # (B, T, k)
    weights = 4 ** torch.arange(k - 1, -1, -1, device=codes.device)
    has_n = (windows >= N_CODE).any(-1)
    ids = torch.where(has_n, torch.ones_like(windows[..., 0]), (windows.clamp(max=3) * weights).sum(-1) + 2)
    counts = ((lengths - k) // stride + 1).clamp(min=0)
    valid = torch.arange(ids.shape[1], device=codes.device)[None, :] < counts[:, None]
    return torch.where(valid, ids, torch.zeros_like(ids)), counts


def kmer_vocab_size(k: int) -> int:
    return 4**k + 2
