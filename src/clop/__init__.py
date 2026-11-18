"""CLOP: Contrastive Language-Omics Pre-training.

Joint embeddings of DNA sequences and their annotation text, for zero-shot classification by text prompts,
cross-modal search and embedding-based exploration of genomes.
"""

__version__ = "2.1.0"

from .io import Record, load_records, parse_bed, parse_fasta, parse_genbank, parse_gff3, reverse_complement  # noqa: E402
from .models import CLOP, ModelConfig  # noqa: E402
from .pipeline import Pipeline  # noqa: E402

__all__ = [
    "CLOP",
    "ModelConfig",
    "Pipeline",
    "Record",
    "load_records",
    "parse_bed",
    "parse_fasta",
    "parse_genbank",
    "parse_gff3",
    "reverse_complement",
    "__version__",
]
