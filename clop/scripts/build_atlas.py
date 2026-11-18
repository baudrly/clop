#!/usr/bin/env python3
"""Convert the six-genome window atlas (one 500 bp window per annotated element, whole chromosomes held out) into a
CLOP FASTA with key=value headers.

The atlas table is produced by the genome-atlas pipeline (``manifest.tsv``: id, species, biotype, chrom, center,
strand, start, end, split, sequence). Usage:  python scripts/build_atlas.py manifest.tsv data/atlas6.fa.gz
"""

import sys

import pandas as pd

from clop.io import Record, write_fasta

SPECIES = {
    "ecoli": "Escherichia coli",
    "yeast": "Saccharomyces cerevisiae",
    "arabidopsis": "Arabidopsis thaliana",
    "worm": "Caenorhabditis elegans",
    "fly": "Drosophila melanogaster",
    "human": "Homo sapiens",
}


def main(manifest: str, output: str) -> None:
    table = pd.read_csv(
        manifest,
        sep="\t",
        keep_default_na=False,
        usecols=["id", "species", "biotype", "chrom", "strand", "start", "end", "split", "sequence"],
    )
    records = (
        Record(
            id=row.id,
            sequence=row.sequence,
            species=SPECIES[row.species],
            biotype=row.biotype,
            chrom=row.chrom,
            start=int(row.start),
            end=int(row.end),
            strand=row.strand,
            split=row.split,
        )
        for row in table.itertuples(index=False)
    )
    print(f"wrote {write_fasta(records, output)} records to {output}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
