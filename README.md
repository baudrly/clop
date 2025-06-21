# CLOP: Contrastive Learning for Omics Pre-training

CLOP is a PyTorch implementation of a CLIP-like model for learning joint
embeddings of DNA sequences and their textual annotations. The model can be
trained on FASTA, BED, or GFF3 files to learn meaningful representations that
enable cross-modal retrieval and classification. It focuses on species and
biotype as primary annotations and optional futher descriptions.

A live demo (embedding visualization, classification) is available
[here](https://baudrly.github.io/clop).

## Features

- **Dual-Encoder Architecture**: Separate encoders for DNA sequences and text
  annotations
- **Multiple Input Formats**:
  - FASTA (with metadata in headers)
  - BED (with reference FASTA)
  - GFF3 (with reference FASTA)
- **Tokenization Options**:
  - K-mer tokenization for DNA
  - Character-level tokenization for DNA
  - Customizable text tokenization
- **Training Features**:
  - Contrastive loss with learnable temperature
  - Mixed-precision training (AMP)
  - Learning rate scheduling
  - Early stopping
- **Evaluation Metrics**:
  - Retrieval metrics (MRR, Recall@K)
  - k-NN classification
  - Clustering metrics (Silhouette, Davies-Bouldin)
- **Export Options**:
  - ONNX format
  - TensorFlow.js (via ONNX)
  - Embedding exports (CSV/Parquet)
- **Comprehensive Reporting**:
  - Markdown and PDF reports
  - Embedding visualizations (PCA, t-SNE, UMAP)
  - Training curves

## Installation

> [!TIP]
> If `just` is available on your system (it is included in the nix flake, to
> install it manually see
> [here](https://github.com/casey/just?tab=readme-ov-file#installation)), you
> can run commands through just recipes. Run `just` to see what's available.

### Complete installation (recommended)

A nix flake is provided to install everything for the project, including CUDA,
python and uv.

> [!NOTE]
> You will need the nix package manager on your system.
> ([installation](https://determinate.systems/posts/determinate-nix-installer/))

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/clop.git
   cd clop
   ```
2. Enter the development environmnent using nix:

   ```bash
   nix develop --no-pure-eval --accept-flake-config \
            "./tools/nix#default" --command $0
   ```

> [!TIP]
> if `just` just is installed, you can do `just dev` instead. If you have direnv
> on your system, run `direnv allow` once, then you will enter the environment
> whenever you `cd` into the directory.

### Python-only installation

The project can be installed with a python dependency manager like uv, but you
will need to setup CUDA yourself.

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/clop.git
   cd clop
   ```

2. Install dependencies defined in pyproject using your favourite python package
   manager. The following extra dependency groups are available:

- onnx: enables ONNX support
- webapp: includes tensorflowjs

  ```bash
  # with uv
  uv sync --all-extras
  ```

## Basic usage

```bash
# Direct FASTA file input

python clop.py \
  --input_file your_data.fasta \
  --input_type fasta \
  --output_dir output \
  --dna_tokenizer_type kmer \
  --kmer_k 6 \
  --epochs 20 \
  --batch_size 32

# BED/GFF files with a reference FASTA file

python clop.py \
  --input_file annotations.gff3 \
  --input_type gff3 \
  --reference_fasta reference_genome.fa \
  --output_dir gff3_output \
  --gff_feature_types gene,mRNA,ncRNA_gene

# Resuming from checkpoint

python clop.py \
  --resume_from_checkpoint output/best_model_checkpoint.pth \
  --output_dir continued_training

# Running tests

python clop.py --test_suite

# Running on a dummy example (for a quick demo)

python clop.py --run_dummy_example
```

## Input file formats

### FASTA

It's recommended that fasta files follow this convention
(Species/Biotype/Description):

```
>sequence_id|species=Human|biotype=protein_coding|description=Example gene
ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
```

The script is designed to be compatible with
[chromosample](https://github.com/baudrly/chromosample).

### BED Format

Standard BED format (6+ columns) with reference FASTA.

### GFF3 Format

Standard GFF3 format with reference FASTA. Use `--gff_feature_types` to specify
which feature types to process.

## Output structure

The script creates the following directory structure:

```
output_dir/
├── reports/               # Generated plots and reports
├── best_model_checkpoint.pth  # Best model weights
├── checkpoint_epoch_*.pth    # Periodic checkpoints
├── dna_tokenizer_vocab.json  # DNA tokenizer vocabulary
├── text_tokenizer_vocab.json # Text tokenizer vocabulary
├── final_embeddings.csv      # Exported embeddings (if enabled)
├── dna_encoder.onnx          # ONNX export (if enabled)
├── text_encoder.onnx         # ONNX export (if enabled)
└── tfjs_*/                  # TensorFlow.js exports (if enabled)
```

## Advanced options

### Model Architecture

```
--embedding_dim: Dimension of embeddings (default: 128)

--hidden_dim: LSTM hidden dimension (default: 256)

--num_layers: Number of LSTM layers (default: 1)

--dropout: Dropout rate (default: 0.1)
```

### Training Parameters

```
--learning_rate: Initial learning rate (default: 1e-3)

--weight_decay: L2 penalty (default: 0.01)

--lr_scheduler_patience: Epochs before reducing LR (default: 3)

--early_stopping_patience: Epochs before early stop (default: 5)

--use_amp: Enable mixed-precision training

--use_torch_compile: Use torch.compile() optimization (PyTorch 2.0+)
```

### Export Options

```
--export_onnx: Export encoders to ONNX format

--export_tfjs: Export ONNX encoders to TensorFlow.js

--export_embeddings_format: Export format for embeddings (csv, parquet, both, none)
```
