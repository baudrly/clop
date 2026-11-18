"""Datasets, batching, leakage-aware splits and class balancing."""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .io import Record, reverse_complement
from .text import TextTokenizer, record_caption
from .tokenizers import encode, pad_codes

logger = logging.getLogger(__name__)
SPLITS = ("train", "validation", "test")


class CaptionTable:
    """The distinct captions of a dataset, tokenised once."""

    def __init__(self, captions: Sequence[str], tokenizer: TextTokenizer, max_length: int = 64):
        self.captions = list(dict.fromkeys(captions))
        self.index = {c: i for i, c in enumerate(self.captions)}
        tokens, lengths = tokenizer.batch(self.captions, max_length)
        self.tokens, self.lengths = torch.from_numpy(tokens), torch.from_numpy(lengths)

    def __len__(self) -> int:
        return len(self.captions)


class SequenceDataset(Dataset):
    """Records as codes plus caption indices. In training mode each item is a random crop and, when ``random_rc``, a
    random strand. Crop lengths: one value is a fixed length; two or more span a log-uniform range from their minimum to
    their maximum, drawn independently of the label so that no class owns a length."""

    def __init__(
        self,
        records: Sequence[Record],
        captions: CaptionTable | None = None,
        caption_fn: Callable[[Record], str] = record_caption,
        max_length: int = 4096,
        crop_lengths: Sequence[int] = (),
        random_rc: bool = False,
    ):
        self.records = list(records)
        self.codes = [encode(r.sequence[:max_length]) for r in self.records]
        self.captions = captions
        self.owner = (
            np.array([captions.index.get(caption_fn(r), -1) for r in self.records], dtype=np.int64)
            if captions is not None
            else np.full(len(self.records), -1, dtype=np.int64)
        )
        self.crop_log = (float(np.log(min(crop_lengths))), float(np.log(max(crop_lengths)))) if crop_lengths else None
        self.random_rc = random_rc

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        codes = self.codes[i]
        if self.crop_log:  # torch RNG: seeded per DataLoader worker, unlike a shared numpy generator
            lo, hi = self.crop_log
            length = min(int(round(np.exp(lo + (hi - lo) * float(torch.rand(()))))), len(codes))
            start = int(torch.randint(len(codes) - length + 1, ()))
            codes = codes[start : start + length]
        if self.random_rc and bool(torch.rand(()) < 0.5):
            codes = np.where(codes < 4, 3 - codes, codes)[::-1].copy()
        return codes, int(self.owner[i])


def collate(batch: list[tuple[np.ndarray, int]]) -> dict[str, torch.Tensor]:
    codes, lengths = pad_codes([b[0] for b in batch])
    owner = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return {"codes": torch.from_numpy(codes), "lengths": torch.from_numpy(lengths), "owner": owner}


def loader(dataset: SequenceDataset, batch_size: int, shuffle: bool, workers: int = 0, seed: int = 0) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=workers,
        generator=generator,
        drop_last=shuffle and len(dataset) > batch_size,
    )


def batch_captions(batch: dict[str, torch.Tensor], captions: CaptionTable) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distinct captions of a batch: tokens, lengths, and each sequence's index among them."""
    unique, owner = torch.unique(batch["owner"], return_inverse=True)
    tokens, lengths = captions.tokens[unique], captions.lengths[unique]
    return tokens[:, : int(lengths.max())], lengths, owner


# ----------------------------------------------------------------------------------------------- splits
def _hash_unit(text: str, seed: int) -> float:
    return int(hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()[:12], 16) / 16**12


def chromosome_split(records: Sequence[Record], fractions: tuple[float, float] = (0.15, 0.15), seed: int = 0) -> dict[str, str]:
    """Assign whole chromosomes of each species to validation and test so that each takes roughly its share of the
    species' records. Species with a single contig fall back to contiguous 1/10-of-contig blocks, so neighbouring
    windows never straddle splits. Returns a map from ``species|chrom`` (or ``species|chrom|block``) to split."""
    by_species: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        by_species[r.species][_unit(r)] += 1
    assignment = {}
    for species, units in by_species.items():
        names = sorted(units, key=lambda u: _hash_unit(f"{species}|{u}", seed))
        total = sum(units.values())
        targets = {"test": fractions[1] * total, "validation": fractions[0] * total}
        filled = {"test": 0, "validation": 0}
        for name in names:
            open_ = [s for s in ("test", "validation") if filled[s] < targets[s] and filled[s] + units[name] <= targets[s] * 1.6]
            split = open_[0] if open_ and len(names) > 2 else "train"
            if split != "train":
                filled[split] += units[name]
            assignment[f"{species}|{name}"] = split
    return assignment


def _unit(record: Record) -> str:
    if record.chrom and record.start >= 0 and record.meta.get("_single_contig"):
        return f"{record.chrom}|{record.start // max(int(record.meta['_block']), 1)}"
    return record.chrom or record.id


def assign_splits(
    records: list[Record],
    strategy: str = "auto",
    fractions: tuple[float, float] = (0.15, 0.15),
    holdout: dict[str, Sequence[str]] | None = None,
    seed: int = 0,
) -> list[Record]:
    """Set ``record.split`` in place and return the records.

    ``header``: keep ``split=`` from the input. ``chromosome``: whole chromosomes per species (or ``holdout``, a map
    from split to chromosome names). ``random``: hash of the sequence, so exact duplicates (either strand) always land
    together. ``auto``: header if every record has one, else chromosome when coordinates exist, else random.
    """
    if strategy == "auto":
        if all(r.split in SPLITS for r in records):
            strategy = "header"
        elif holdout or all(r.chrom for r in records):
            strategy = "chromosome"
        else:
            strategy = "random"
    if strategy == "header":
        missing = sum(r.split not in SPLITS for r in records)
        if missing:
            raise ValueError(f"{missing} records lack a valid split= field")
    elif strategy == "chromosome":
        if holdout:
            lookup = {c: s for s, chroms in holdout.items() for c in chroms}
            for r in records:
                r.split = lookup.get(r.chrom, lookup.get(f"{r.species}|{r.chrom}", "train"))
        else:
            contigs = defaultdict(set)
            spans = defaultdict(int)
            for r in records:
                contigs[r.species].add(r.chrom)
                spans[r.species] = max(spans[r.species], r.end)
            for r in records:
                if len(contigs[r.species]) < 3 and r.start >= 0:
                    r.meta["_single_contig"], r.meta["_block"] = "1", str(max(spans[r.species] // 10, 1))
            assignment = chromosome_split(records, fractions, seed)
            for r in records:
                r.split = assignment[f"{r.species}|{_unit(r)}"]
                r.meta.pop("_single_contig", None)
                r.meta.pop("_block", None)
    elif strategy == "random":
        for r in records:
            key = min(r.sequence, reverse_complement(r.sequence))
            u = _hash_unit(key, seed)
            r.split = "test" if u < fractions[1] else "validation" if u < fractions[0] + fractions[1] else "train"
    else:
        raise ValueError(f"Unknown split strategy {strategy!r}")
    counts = {s: sum(r.split == s for r in records) for s in SPLITS}
    logger.info("Split (%s): %s", strategy, counts)
    return records


def cap_per_cell(records: Sequence[Record], cap: int, seed: int = 0) -> list[Record]:
    """At most ``cap`` records per species x biotype cell (deterministic by hash), to stop large classes dominating."""
    if cap <= 0:
        return list(records)
    by_cell = defaultdict(list)
    for r in records:
        by_cell[r.cell].append(r)
    out = []
    for members in by_cell.values():
        members.sort(key=lambda r: _hash_unit(r.id + r.sequence[:32], seed))
        out.extend(members[:cap])
    return out


def drop_rare_cells(records: Sequence[Record], minimum: int) -> list[Record]:
    """Remove species x biotype cells with fewer than ``minimum`` training records."""
    counts = defaultdict(int)
    for r in records:
        counts[r.cell] += r.split == "train"
    return [r for r in records if counts[r.cell] >= minimum]


def split_records(records: Sequence[Record]) -> dict[str, list[Record]]:
    return {s: [r for r in records if r.split == s] for s in SPLITS}
