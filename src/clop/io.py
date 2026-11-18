"""Reading sequences and annotations: FASTA, BED, GFF3/GTF and GenBank, plus fixed-length windowing.

Every reader returns :class:`Record` objects. A record carries the sequence and the metadata CLOP needs to build
annotation text (species, biotype, description) and to make leakage-free splits (chromosome, coordinates).
"""

from __future__ import annotations

import gzip
import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

UNKNOWN = "unknown"
EXCLUDED_BIOTYPES = frozenset({"", UNKNOWN, "unknown_biotype", "n/a", "na", "random_dna", "biological_region"})
_COMPLEMENT = str.maketrans("ACGTRYKMBDHVNacgtrykmbdhvn", "TGCAYRMKVHDBNtgcayrmkvhdbn")
_ATTRIBUTE_GFF = re.compile(r"\s*([^=;]+)=([^;]*)")
_ATTRIBUTE_GTF = re.compile(r'\s*(\S+)\s+"([^"]*)"')
_BIOTYPE_KEYS = ("biotype", "gene_biotype", "transcript_biotype", "gene_type", "transcript_type")
_SPLIT_NAMES = {"train", "validation", "test"}


@dataclass
class Record:
    """One sequence with its annotation."""

    id: str
    sequence: str
    species: str = UNKNOWN
    biotype: str = UNKNOWN
    description: str = ""
    chrom: str = ""
    start: int = -1
    end: int = -1
    strand: str = "+"
    split: str = ""
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def cell(self) -> str:
        """The species x biotype label used for stratification and evaluation."""
        return f"{self.species}|{self.biotype}"

    def header(self) -> str:
        """A FASTA header that :func:`parse_header` reads back losslessly."""
        fields = {
            "species": self.species,
            "biotype": self.biotype,
            "description": self.description,
            "chrom": self.chrom,
            "start": self.start if self.start >= 0 else "",
            "end": self.end if self.end >= 0 else "",
            "strand": self.strand if self.chrom else "",
            "split": self.split,
            **self.meta,
        }
        return "|".join([self.id] + [f"{k}={_escape(v)}" for k, v in fields.items() if v != ""])


def _escape(value) -> str:
    return str(value).replace("|", "/").replace("\n", " ")


def reverse_complement(sequence: str) -> str:
    """Reverse complement, IUPAC-aware and case-preserving.

    >>> reverse_complement("ATGCN")
    'NGCAT'
    """
    return sequence.translate(_COMPLEMENT)[::-1]


def open_text(path: str | Path):
    path = str(path)
    return gzip.open(path, "rt", errors="replace") if path.endswith((".gz", ".bgz")) else open(path, errors="replace")


def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Stream (header without '>', uppercase sequence) pairs."""
    header, chunks = None, []
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line[0] == ">":
                if header is not None:
                    yield header, "".join(chunks).upper()
                header, chunks = line[1:].strip(), []
            elif header is not None:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks).upper()


def parse_header(header: str) -> dict[str, str]:
    """Read metadata from a FASTA header.

    Three conventions are understood, in this order:

    * key=value fields separated by ``|``: ``>id|species=Homo sapiens|biotype=lncRNA|description=...``
    * chromosample positional fields: ``>species|biotype|description``
    * anything else: the first word is the id and the rest is a free-form description.

    >>> parse_header("s1|species=H|biotype=G")["species"]
    'H'
    >>> parse_header("Homo sapiens|snRNA|U1 spliceosomal RNA")["biotype"]
    'snRNA'
    >>> parse_header("s2 some description")["description"]
    'some description'
    """
    header = header.lstrip(">").strip()
    parts = [p.strip() for p in header.split("|")]
    keyed = [p for p in parts if "=" in p]
    if keyed:
        meta = {"id": parts[0] if "=" not in parts[0] else ""}
        for part in keyed:
            key, _, value = part.partition("=")
            meta[key.strip().lower()] = value.strip()
        meta["id"] = meta["id"] or meta.get("name", "") or meta.get("species", "seq")
        return meta
    if len(parts) >= 2:
        return {
            "id": f"{parts[0]}|{parts[1]}",
            "species": parts[0] or UNKNOWN,
            "biotype": parts[1] or UNKNOWN,
            "description": "|".join(parts[2:]),
        }
    word, _, rest = header.partition(" ")
    return {"id": word, "description": rest.strip()}


_RECORD_FIELDS = {"species", "biotype", "description", "chrom", "start", "end", "strand", "split", "id"}


def record_from_header(header: str, sequence: str, index: int = 0) -> Record:
    meta = parse_header(header)
    extra = {k: v for k, v in meta.items() if k not in _RECORD_FIELDS}
    return Record(
        id=meta.get("id") or f"seq{index}",
        sequence=sequence,
        species=meta.get("species", UNKNOWN) or UNKNOWN,
        biotype=meta.get("biotype", UNKNOWN) or UNKNOWN,
        description=meta.get("description", ""),
        chrom=meta.get("chrom", ""),
        start=int(meta.get("start", -1) or -1),
        end=int(meta.get("end", -1) or -1),
        strand=meta.get("strand", "+") or "+",
        split=meta.get("split", "").lower(),
        meta=extra,
    )


def parse_fasta(path: str | Path, exclude_biotypes: Iterable[str] = EXCLUDED_BIOTYPES, keep_unknown: bool = True) -> list[Record]:
    """Parse an annotated FASTA file into records.

    Records whose biotype is in ``exclude_biotypes`` are dropped, except unannotated records when ``keep_unknown``
    (useful for embedding or classifying new sequences).
    """
    excluded = {b.lower() for b in exclude_biotypes}
    records = []
    for i, (header, sequence) in enumerate(read_fasta(path)):
        record = record_from_header(header, sequence, i)
        if record.biotype.lower() in excluded and not (keep_unknown and record.biotype == UNKNOWN):
            continue
        records.append(record)
    logger.info("Parsed %d sequences from %s", len(records), path)
    return records


def write_fasta(records: Iterable[Record], path: str | Path, width: int = 0) -> int:
    """Write records with key=value headers; returns the number written."""
    n = 0
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as handle:
        for record in records:
            sequence = (
                record.sequence
                if not width
                else "\n".join(record.sequence[i : i + width] for i in range(0, len(record.sequence), width))
            )
            handle.write(f">{record.header()}\n{sequence}\n")
            n += 1
    return n


# ----------------------------------------------------------------------------------------------- reference genomes
class Reference:
    """Chromosome sequences of a genome, with per-contig header metadata and chr-prefix aliases."""

    def __init__(self, sequences: dict[str, str], meta: dict[str, dict[str, str]] | None = None, species: str = ""):
        self.sequences, self.meta, self.species = sequences, meta or {}, species
        self._alias = {}
        for name in sequences:
            self._alias[name] = name
            self._alias.setdefault(name[3:] if name.startswith("chr") else f"chr{name}", name)

    @classmethod
    def from_fasta(cls, path: str | Path, species: str = "") -> Reference:
        sequences, meta = {}, {}
        for header, sequence in read_fasta(path):
            fields = parse_header(header)
            name = header.split("|")[0].split()[0]
            sequences[name], meta[name] = sequence, fields
        return cls(sequences, meta, species)

    def resolve(self, chrom: str) -> str | None:
        return self._alias.get(chrom)

    def species_of(self, chrom: str) -> str:
        name = self.resolve(chrom)
        return self.species or (self.meta.get(name, {}).get("species", UNKNOWN) if name else UNKNOWN)

    def fetch(self, chrom: str, start: int, end: int, strand: str = "+") -> str:
        """0-based half-open slice, reverse-complemented on the minus strand; out-of-range parts are padded with N."""
        sequence = self.sequences[self.resolve(chrom)]
        left, right = max(start, 0), min(end, len(sequence))
        piece = "N" * (left - start) + sequence[left:right] + "N" * (end - right)
        return reverse_complement(piece) if strand == "-" else piece

    def length(self, chrom: str) -> int:
        return len(self.sequences[self.resolve(chrom)])


def parse_attributes(text: str) -> dict[str, str]:
    """GFF3 (key=value;) or GTF (key "value";) attributes.

    >>> parse_attributes('ID=g1;Name=ABC;biotype=lncRNA')["biotype"]
    'lncRNA'
    >>> parse_attributes('gene_id "g1"; gene_biotype "snRNA";')["gene_biotype"]
    'snRNA'
    """
    text = text.strip()
    if not text or text == ".":
        return {}
    pattern = _ATTRIBUTE_GTF if '"' in text and "=" not in text.split(";")[0] else _ATTRIBUTE_GFF
    return {k.strip(): _unquote(v.strip()) for k, v in pattern.findall(text)}


def _unquote(value: str) -> str:
    from urllib.parse import unquote

    return unquote(value)


def _biotype(attributes: dict[str, str], default: str) -> str:
    return next((attributes[k] for k in _BIOTYPE_KEYS if attributes.get(k)), default)


def parse_bed(path: str | Path, reference: Reference | str | Path, species: str = "") -> list[Record]:
    """BED3+ intervals extracted from a reference. Column 4 is the name; an optional attribute column (any column
    containing ``key=value``) supplies biotype and description; otherwise the biotype is the reference contig's."""
    reference = reference if isinstance(reference, Reference) else Reference.from_fasta(reference, species)
    records, missing = [], 0
    with open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            cols = line.rstrip("\n").split("\t")
            chrom, start, end = cols[0], int(cols[1]), int(cols[2])
            if reference.resolve(chrom) is None:
                missing += 1
                continue
            name = cols[3] if len(cols) > 3 and cols[3] not in ("", ".") else f"{chrom}:{start}-{end}"
            strand = cols[5] if len(cols) > 5 and cols[5] in "+-" else "+"
            attributes = next((parse_attributes(c) for c in cols[6:] if "=" in c or '"' in c), {})
            contig = reference.meta.get(reference.resolve(chrom), {})
            records.append(
                Record(
                    id=name,
                    sequence=reference.fetch(chrom, start, end, strand),
                    species=species or reference.species_of(chrom),
                    biotype=_biotype(attributes, contig.get("biotype", UNKNOWN)),
                    description=attributes.get("description", attributes.get("Note", "")),
                    chrom=reference.resolve(chrom),
                    start=start,
                    end=end,
                    strand=strand,
                    meta={k: v for k, v in attributes.items() if k not in _BIOTYPE_KEYS and k != "description"},
                )
            )
    if missing:
        logger.warning("%s: %d BED lines on contigs absent from the reference were skipped", path, missing)
    return records


def iter_gff(path: str | Path) -> Iterator[tuple[str, str, int, int, str, dict[str, str]]]:
    """(chrom, feature type, 0-based start, end, strand, attributes) for every GFF3/GTF feature line."""
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("##FASTA"):
                return
            if not line.strip() or line[0] == "#":
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            yield cols[0], cols[2], int(cols[3]) - 1, int(cols[4]), cols[6] if cols[6] in "+-" else "+", parse_attributes(cols[8])


def parse_gff3(
    path: str | Path, reference: Reference | str | Path, feature_types: Iterable[str] | None = None, species: str = ""
) -> list[Record]:
    """Features of the requested types (all when ``None``) extracted from a reference, strand-aware.
    The biotype is the ``biotype``/``gene_biotype``/... attribute, falling back to the feature type."""
    reference = reference if isinstance(reference, Reference) else Reference.from_fasta(reference, species)
    wanted = set(feature_types) if feature_types else None
    records, missing = [], 0
    for chrom, kind, start, end, strand, attributes in iter_gff(path):
        if wanted is not None and kind not in wanted:
            continue
        if reference.resolve(chrom) is None:
            missing += 1
            continue
        identifier = (
            attributes.get("ID") or attributes.get("gene_id") or attributes.get("transcript_id") or f"{chrom}:{start}-{end}"
        )
        records.append(
            Record(
                id=identifier,
                sequence=reference.fetch(chrom, start, end, strand),
                species=species or reference.species_of(chrom),
                biotype=_biotype(attributes, kind),
                description=attributes.get("description", attributes.get("Note", "")),
                chrom=reference.resolve(chrom),
                start=start,
                end=end,
                strand=strand,
                meta={"feature_type": kind, **({"name": attributes["Name"]} if "Name" in attributes else {})},
            )
        )
    if missing:
        logger.warning("%s: %d features on contigs absent from the reference were skipped", path, missing)
    return records


# ----------------------------------------------------------------------------------------------- GenBank
_LOCATION_SPAN = re.compile(r"(\d+)")


def _genbank_location(text: str) -> tuple[int, int, str, list[tuple[int, int]]]:
    """Span (0-based start, end), strand and parts of a GenBank location such as complement(join(1..10,20..>30))."""
    strand = "-" if text.startswith("complement(") else "+"
    parts = []
    for piece in re.findall(r"<?(\d+)(?:\.\.>?(\d+))?", text):
        a = int(piece[0])
        b = int(piece[1]) if piece[1] else a
        parts.append((min(a, b) - 1, max(a, b)))
    return min(p[0] for p in parts), max(p[1] for p in parts), strand, sorted(parts)


def read_genbank(path: str | Path) -> Iterator[tuple[dict[str, str], str, list[dict]]]:
    """Stream (header fields, sequence, features) per GenBank record without Biopython.

    Features are dicts with ``type``, ``start``, ``end``, ``strand``, ``parts`` and qualifiers."""
    fields, features, sequence, section, current = {}, [], [], None, None
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("//"):
                yield fields, "".join(sequence).upper(), features
                fields, features, sequence, section, current = {}, [], [], None, None
                continue
            if line.startswith("LOCUS"):
                fields["locus"] = line.split()[1]
            elif line.startswith("DEFINITION"):
                fields["definition"] = line[12:].strip()
            elif line.startswith("ACCESSION"):
                fields["accession"] = line[12:].split()[0] if line[12:].split() else ""
            elif line.startswith("  ORGANISM"):
                fields["organism"] = line[12:].strip()
            elif line.startswith("FEATURES"):
                section = "features"
            elif line.startswith("ORIGIN"):
                section = "origin"
            elif section == "features" and len(line) > 21 and line[5] != " ":
                kind, location = line[5:21].strip(), line[21:].strip()
                current = {"type": kind, "location": location}
                features.append(current)
            elif section == "features" and current is not None and line.startswith(" " * 21):
                text = line[21:].rstrip("\n")
                if text.startswith("/"):
                    key, _, value = text[1:].partition("=")
                    current[key] = value.strip('"')
                    current["_last"] = key
                elif "_last" in current:
                    key = current["_last"]
                    current[key] = (current[key] + " " + text.strip().strip('"')).strip()
                else:
                    current["location"] += text.strip()
            elif section == "origin":
                sequence.append("".join(ch for ch in line if ch.isalpha()))
    for feature in features:
        feature.pop("_last", None)


def parse_genbank(
    path: str | Path, feature_types: Iterable[str] | None = ("CDS", "tRNA", "rRNA", "ncRNA"), species: str = ""
) -> tuple[list[Record], Reference]:
    """Features of GenBank records as records (span of joined locations), plus the records as a reference."""
    wanted = set(feature_types) if feature_types else None
    records, sequences, meta = [], {}, {}
    for fields, sequence, features in read_genbank(path):
        name = fields.get("accession") or fields.get("locus", f"record{len(sequences)}")
        organism = species or fields.get("organism", UNKNOWN)
        sequences[name], meta[name] = sequence, {"species": organism}
        reference = Reference({name: sequence})
        for feature in features:
            if wanted is not None and feature["type"] not in wanted or feature.get("pseudo") is not None:
                continue
            start, end, strand, _ = _genbank_location(feature["location"])
            label = feature.get("locus_tag") or feature.get("gene") or f"{name}:{start}-{end}"
            records.append(
                Record(
                    id=label,
                    sequence=reference.fetch(name, start, end, strand),
                    species=organism,
                    biotype=feature.get("ncRNA_class", feature["type"]),
                    description=feature.get("product", ""),
                    chrom=name,
                    start=start,
                    end=end,
                    strand=strand,
                    meta={"feature_type": feature["type"]},
                )
            )
    return records, Reference(sequences, meta, species)


# ----------------------------------------------------------------------------------------------- windows
def centred_window(record: Record, reference: Reference, size: int, anchor: str = "center") -> Record:
    """A fixed-size window around a feature: on its midpoint (``center``) or on its 5' end (``start``, e.g. a TSS),
    in the feature's orientation. Fixed windows stop sequence length from carrying the label."""
    if anchor == "start":
        point = record.start if record.strand == "+" else record.end - 1
    else:
        point = (record.start + record.end) // 2
    start = point - size // 2
    return replace(
        record,
        sequence=reference.fetch(record.chrom, start, start + size, record.strand),
        start=start,
        end=start + size,
        meta={**record.meta, "anchor": anchor, "feature_start": str(record.start), "feature_end": str(record.end)},
    )


def promoter_windows(records: Iterable[Record], reference: Reference, size: int, upstream: int = 0) -> list[Record]:
    """Windows centred ``upstream`` bp 5' of each feature start (transcription or translation start), labelled
    ``promoter``."""
    out = []
    for record in records:
        shift = -upstream if record.strand == "+" else upstream
        anchor = (record.start if record.strand == "+" else record.end - 1) + shift
        start = anchor - size // 2
        out.append(
            replace(
                record,
                id=f"{record.id}:promoter",
                biotype="promoter",
                start=start,
                end=start + size,
                sequence=reference.fetch(record.chrom, start, start + size, record.strand),
                meta={**record.meta, "anchor": "start"},
            )
        )
    return out


def intergenic_windows(
    features: Iterable[Record], reference: Reference, size: int, n: int, margin: int = 1000, species: str = "", seed: int = 0
) -> list[Record]:
    """Up to ``n`` random windows at least ``margin`` bp away from every feature, labelled ``intergenic``."""
    rng = np.random.default_rng(seed)
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for f in features:
        by_chrom.setdefault(f.chrom, []).append((f.start - margin, f.end + margin))
    free = []  # (chrom, start, end) of allowed window starts
    for chrom in reference.sequences:
        blocked = sorted(by_chrom.get(chrom, []))
        cursor = 0
        for a, b in blocked + [(reference.length(chrom), reference.length(chrom))]:
            if a - size > cursor:
                free.append((chrom, cursor, a - size))
            cursor = max(cursor, b)
    if not free:
        return []
    widths = np.array([e - s for _, s, e in free], dtype=float)
    picks = rng.choice(len(free), size=n, p=widths / widths.sum())
    out, seen = [], set()
    for k in picks:
        chrom, s, e = free[k]
        start = int(rng.integers(s, e))
        if (chrom, start // size) in seen:
            continue
        seen.add((chrom, start // size))
        sequence = reference.fetch(chrom, start, start + size)
        if sequence.count("N") > size // 10:
            continue
        out.append(
            Record(
                id=f"{chrom}:{start}:intergenic",
                sequence=sequence,
                species=species or reference.species_of(chrom),
                biotype="intergenic",
                chrom=chrom,
                start=start,
                end=start + size,
            )
        )
    return out


def deduplicate(records: list[Record], core: int = 0) -> list[Record]:
    """Drop exact duplicates (of the whole sequence, or of its central ``core`` bp on either strand)."""
    seen, out = set(), []
    for record in records:
        s = record.sequence
        if core:
            mid = len(s) // 2
            s = s[max(mid - core // 2, 0) : mid + core - core // 2]
        key = min(s, reverse_complement(s))
        if key not in seen:
            seen.add(key)
            out.append(record)
    return out


def load_records(
    path: str | Path,
    input_type: str = "auto",
    reference: str | Path | None = None,
    feature_types: Iterable[str] | None = None,
    species: str = "",
) -> list[Record]:
    """Dispatch on file type (``fasta``, ``bed``, ``gff3``/``gtf``, ``genbank`` or ``auto`` from the extension)."""
    kind = input_type
    if kind == "auto":
        name = str(path).lower().removesuffix(".gz").removesuffix(".bgz")
        kind = next(
            (
                t
                for t, ext in (
                    ("bed", (".bed",)),
                    ("gff3", (".gff", ".gff3", ".gtf")),
                    ("genbank", (".gb", ".gbk", ".gbff", ".genbank")),
                )
                if name.endswith(ext)
            ),
            "fasta",
        )
    if kind == "fasta":
        records = parse_fasta(path)
        if species:
            for r in records:
                r.species = species if r.species == UNKNOWN else r.species
        return records
    if kind == "genbank":
        return parse_genbank(path, feature_types, species)[0]
    if reference is None:
        raise ValueError(f"--reference is required for {kind} input")
    if kind == "bed":
        return parse_bed(path, reference, species)
    if kind in ("gff3", "gff", "gtf"):
        return parse_gff3(path, reference, feature_types, species)
    raise ValueError(f"Unknown input type {input_type!r}")


def summarise(records: list[Record]) -> dict:
    lengths = np.fromiter((len(r.sequence) for r in records), dtype=np.int64, count=len(records))
    species, biotypes = {}, {}
    for r in records:
        species[r.species] = species.get(r.species, 0) + 1
        biotypes[r.biotype] = biotypes.get(r.biotype, 0) + 1
    return {
        "n": len(records),
        "species": dict(sorted(species.items(), key=lambda x: -x[1])),
        "biotypes": dict(sorted(biotypes.items(), key=lambda x: -x[1])),
        "length": {
            "min": int(lengths.min(initial=0)),
            "median": float(np.median(lengths)) if len(lengths) else 0.0,
            "max": int(lengths.max(initial=0)),
        },
    }
