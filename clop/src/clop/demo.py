"""A synthetic genome atlas that trains in seconds: species differ in GC content and CpG depletion, biotypes carry
planted signals (a TATA box for promoters, a B box for tRNAs, codon periodicity for coding sequence)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .io import Record, write_fasta

logger = logging.getLogger(__name__)
SPECIES = {"Synthetica alba": (0.35, 1.0), "Synthetica rubra": (0.50, 0.25), "Fictivus viridis": (0.62, 0.8)}
CODONS = np.array([list(c) for c in ("ATG", "GCT", "GAA", "AAA", "CTG", "GGC", "TTC", "CAG")])


def background(rng: np.random.Generator, n: int, gc: float, cpg: float) -> np.ndarray:
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    s = rng.choice(np.array(list("ACGT")), n, p=p)
    cg = np.flatnonzero((s[:-1] == "C") & (s[1:] == "G"))
    drop = cg[rng.random(len(cg)) > cpg]
    s[drop + 1] = "A"  # deplete CpG like a methylated genome
    return s


def plant(s: np.ndarray, motif: str, rng: np.random.Generator, copies: int = 2) -> None:
    for _ in range(copies):
        at = int(rng.integers(0, len(s) - len(motif)))
        s[at : at + len(motif)] = list(motif)


def synthetic_records(n_per_cell: int = 150, length: int = 300, seed: int = 0) -> list[Record]:
    rng = np.random.default_rng(seed)
    records = []
    for species, (gc, cpg) in SPECIES.items():
        for biotype in ("CDS", "promoter", "tRNA", "intergenic"):
            for i in range(n_per_cell):
                s = background(rng, length, gc, cpg)
                if biotype == "CDS":
                    s = CODONS[rng.integers(0, len(CODONS), length // 3)].ravel()
                    s = np.concatenate([s, background(rng, length - len(s), gc, cpg)])
                elif biotype == "promoter":
                    plant(s, "TATAAA", rng)
                elif biotype == "tRNA":
                    plant(s, "GGTTCGAATCC", rng)
                chrom = f"chr{i % 7 + 1}"
                records.append(
                    Record(
                        id=f"{species.split()[1]}_{biotype}_{i}",
                        sequence="".join(s),
                        species=species,
                        biotype=biotype,
                        chrom=chrom,
                        start=i * length,
                        end=(i + 1) * length,
                    )
                )
    return records


def run_demo(output: str | Path, device: str = "cpu") -> dict:
    from .models import ModelConfig
    from .report import write_report
    from .train import TrainConfig
    from .workflow import prepare_records, train_model

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    records = synthetic_records()
    write_fasta(records, out / "synthetic.fa")
    parts = prepare_records(records, "chromosome", (0.15, 0.15))
    summary = train_model(
        parts, ModelConfig(width=48, depth=2, embed_dim=32), TrainConfig(epochs=6, batch_size=64, lr=3e-3), out, device
    )
    write_report(out)
    m = summary["test"]["zero_shot"]
    logger.info(
        "demo: zero-shot species F1 %.2f, biotype F1 %.2f, cell F1 %.2f -> %s",
        m["species_macro_f1"],
        m["biotype_macro_f1"],
        m["cell_macro_f1"],
        out / "report.html",
    )
    return summary
