"""Annotation text: building captions from records, prompt templates, and the word-level text tokenizer.

Captions combine the species (expanded to a short lineage when known, so the text side can compose what it has
seen: a new species x biotype pair shares words with both), the biotype (expanded to synonyms), and the free-form
description.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np

PAD, UNK = "<pad>", "<unk>"
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Common species -> lineage words. Users can extend this table or pass ``lineage=...`` in FASTA headers.
LINEAGE = {
    "homo sapiens": "human primate mammal vertebrate animal eukaryote",
    "mus musculus": "mouse rodent mammal vertebrate animal eukaryote",
    "rattus norvegicus": "rat rodent mammal vertebrate animal eukaryote",
    "macaca mulatta": "macaque primate mammal vertebrate animal eukaryote",
    "danio rerio": "zebrafish fish vertebrate animal eukaryote",
    "gallus gallus": "chicken bird vertebrate animal eukaryote",
    "drosophila melanogaster": "fruit fly insect arthropod animal eukaryote",
    "caenorhabditis elegans": "nematode worm animal eukaryote",
    "saccharomyces cerevisiae": "budding yeast fungus eukaryote",
    "schizosaccharomyces pombe": "fission yeast fungus eukaryote",
    "arabidopsis thaliana": "thale cress flowering plant eudicot eukaryote",
    "oryza sativa": "rice grass flowering plant monocot eukaryote",
    "zea mays": "maize grass flowering plant monocot eukaryote",
    "actinidia chinensis": "kiwifruit flowering plant eudicot eukaryote",
    "plasmodium falciparum": "malaria parasite apicomplexan protist eukaryote",
    "escherichia coli": "proteobacterium bacterium prokaryote",
    "bacillus subtilis": "firmicute bacterium prokaryote",
    "bacteroides fragilis": "bacteroidetes bacterium prokaryote",
    "mycobacterium tuberculosis": "actinobacterium bacterium prokaryote",
    "methanocaldococcus jannaschii": "archaeon prokaryote",
}

BIOTYPE_WORDS = {
    "cds": "coding sequence cds protein coding exon",
    "protein_coding": "protein coding gene mrna",
    "mrna": "messenger rna mrna protein coding transcript",
    "five_prime_utr": "five prime untranslated region 5' utr",
    "three_prime_utr": "three prime untranslated region 3' utr",
    "utr": "untranslated region utr",
    "exon": "exon",
    "intron": "intron spliced",
    "intergenic": "intergenic spacer non genic",
    "promoter": "promoter transcription start upstream regulatory",
    "enhancer": "enhancer distal regulatory element",
    "open_chromatin": "open chromatin accessible regulatory region",
    "lncrna": "long non coding rna lncrna",
    "pseudogene": "pseudogene",
    "processed_pseudogene": "processed pseudogene",
    "trna": "transfer rna trna",
    "rrna": "ribosomal rna rrna",
    "snrna": "small nuclear rna snrna spliceosomal",
    "snorna": "small nucleolar rna snorna",
    "mirna": "microrna mirna precursor",
    "misc_rna": "miscellaneous rna",
    "ncrna": "non coding rna ncrna",
    "transposable_element": "transposable element transposon repeat",
    "repeat": "repeat element",
    "replication_origin": "replication origin ars",
    "centromere": "centromere",
    "telomere": "telomere repeat",
}


def species_words(species: str, lineage: str = "") -> str:
    key = species.strip().lower().replace("_", " ")
    return " ".join(w for w in (key, lineage or LINEAGE.get(key, "")) if w)


def biotype_words(biotype: str) -> str:
    key = biotype.strip().lower()
    return BIOTYPE_WORDS.get(key, key.replace("_", " "))


def caption(species: str, biotype: str, description: str = "", lineage: str = "", use_description: bool = True) -> str:
    """Annotation text for one sequence.

    >>> caption("Homo sapiens", "snRNA")
    'homo sapiens human primate mammal vertebrate animal eukaryote small nuclear rna snrna spliceosomal'
    """
    parts = [species_words(species, lineage), biotype_words(biotype)]
    if use_description and description:
        parts.append(description.lower())
    return " ".join(p for p in parts if p)


def record_caption(record, use_description: bool = False) -> str:
    return caption(record.species, record.biotype, record.description, record.meta.get("lineage", ""), use_description)


class TextTokenizer:
    """Lower-cased word tokenizer with a frozen vocabulary (index 0 = padding, 1 = unknown)."""

    def __init__(self, vocab: dict[str, int] | None = None):
        self.vocab = {PAD: 0, UNK: 1}
        for word in sorted(vocab or {}, key=lambda w: (vocab or {})[w]):
            self.vocab.setdefault(word, len(self.vocab))

    @staticmethod
    def words(text: str) -> list[str]:
        """
        >>> TextTokenizer.words("Small nuclear RNA (snRNA), U1")
        ['small', 'nuclear', 'rna', 'snrna', 'u1']
        """
        return _WORD.findall(text.lower())

    @classmethod
    def build(cls, texts: Iterable[str], min_count: int = 1, extra: Iterable[str] = ()) -> TextTokenizer:
        counts = Counter(w for t in texts for w in cls.words(t))
        for t in extra:
            counts.update(cls.words(t))
        words = sorted((w for w, c in counts.items() if c >= min_count), key=lambda w: (-counts[w], w))
        return cls({w: i + 2 for i, w in enumerate(words)})

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, max_length: int = 64) -> list[int]:
        ids = [self.vocab.get(w, 1) for w in self.words(text)][:max_length]
        return ids or [1]

    def batch(self, texts: list[str], max_length: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Padded (n, L) int64 token matrix and lengths."""
        encoded = [self.encode(t, max_length) for t in texts]
        lengths = np.fromiter((len(e) for e in encoded), dtype=np.int64, count=len(encoded))
        tokens = np.zeros((len(encoded), int(lengths.max(initial=1))), dtype=np.int64)
        for i, e in enumerate(encoded):
            tokens[i, : len(e)] = e
        return tokens, lengths

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.vocab))

    @classmethod
    def load(cls, path: str | Path) -> TextTokenizer:
        return cls(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, int]:
        return dict(self.vocab)


DEFAULT_TEMPLATES = ("{species} {biotype}",)
