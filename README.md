# CLOP: Contrastive Language–Omics Pre-training

CLOP learns one embedding space for DNA sequences and the text that annotates them, the way CLIP does for images
and captions. A trained model names the species and biotype of a sequence from **text prompts** (zero-shot), finds
sequences that match a **free-text description**, and gives every sequence a vector you can cluster, search or plot.

**[Live reader](https://baudrly.github.io/clop)**: paste DNA, get calls, see it land in a 3D map of six genomes.
Everything runs in the browser, and the exported reader opens straight from disk.

## The web reader

`clop export -m model.pt -o site --web --atlas windows.fa` writes a static site. Double-click `site/index.html`, put the
folder on any web server, or send `site/clop-reader.html`, the same reader as a single file.

* **Read DNA**: paste or drop one sequence or a multi-record FASTA. You get species, biotype and species x biotype calls,
  the nearest annotated windows, and a white star in the 3D map. Long sequences get a scan: biotype probabilities in
  sliding 300 bp windows, with the called biotype and species as strips and, when the header carries
  `segments=CDS:0-500,...`, the annotation above them. Robustness checks compare the reverse complement (identical by
  construction), a composition-preserving shuffle, and centred crops from 60 bp up. Several records give a table of
  calls (checked against `species=`/`biotype=` header fields) that downloads as TSV.
* **Examples**: up to three annotated windows per species x biotype cell, a gene chimera, a six-record batch, and
  probes (reverse complement, shuffle, random sequence at 30% or 50% GC, telomeric repeat, CpG-rich sequence).
* **Search**: free text or composed prompts light up the closest windows, with the similarity histogram. Words the
  model never saw are listed.
* **Prompts**: write your own classifier, one prompt per line, and colour the whole map by it or call the current
  sequence. Preset sets cover kingdoms, gene structure, RNA genes, regulation and repeats.
* **Model**: size, window, pooling, and the similarity matrix of the label prompts.
* **Map**: colour by species, biotype, call correctness, confidence or your prompts. Each group's prompt is labelled
  where it points, and clicking a legend entry isolates that group.

## Install

```bash
pip install "clop[all] @ git+https://github.com/baudrly/clop"     # or, from a clone:  pip install -e ".[all]"
```

Python ≥ 3.10 and PyTorch ≥ 2.1. Extras: `umap` (UMAP layouts; t-SNE otherwise), `onnx` (export), `parquet`.
A Nix/uv development shell is still provided (`just dev`), and a container image is built on every release
(`docker run ghcr.io/baudrly/clop train -h`).

## Quick start

```bash
clop demo -o clop_demo        # trains on a synthetic genome atlas in ~30 s and opens a report
```

### Train on your own annotations

```bash
# 1. Fixed 500 bp windows around annotated features, plus promoters and intergenic background, split by chromosome
clop prepare -i annotation.gff3.gz --reference genome.fa.gz --species "Mus musculus" \
    --feature-types CDS,five_prime_UTR,three_prime_UTR,lncRNA,snRNA,mRNA --window-types CDS,five_prime_UTR,three_prime_UTR,lncRNA,snRNA \
    --promoters mRNA --intergenic 2000 --cap 2000 -o mouse.fa

# 2. Train (several species: pass several prepared FASTA files)
clop train -i mouse.fa human.fa yeast.fa -o run --epochs 20          # crops of 100-500 bp by default

# 3. Use it
clop classify -m run/model.pt -i new_sequences.fa -o calls.tsv
clop classify -m run/model.pt -i reads.fa --compose "Danio rerio,Mus musculus" "CDS,lncRNA,intergenic"   # new prompts
clop classify -m run/model.pt -i unknown_genome.fa --pooled biotype      # species-agnostic prompts for new genomes
clop search   -m run/model.pt -i library.fa -q "small nuclear rna spliceosomal" --top 20
clop embed    -m run/model.pt -i library.fa -o embeddings.parquet
clop export   -m run/model.pt -o site --web --atlas mouse.fa      # the web reader: open site/index.html
clop export   -m run/model.pt -o onnx --onnx
```

`clop train` writes `model.pt` (self-contained: weights, vocabulary, label prompts, configuration),
`metrics.json`, `history.json`, `embeddings.npz` and `report.html` (interactive 3D atlas, metrics, figures).

### Python

```python
from clop import Pipeline

pipe = Pipeline.load("run/model.pt")
calls = pipe.classify(["ATGGCC..."], top=3)            # zero-shot: top labels, species and biotype marginals
pipe.set_labels(pipe.labels_from_pairs([("Danio rerio", "lncRNA"), ("Danio rerio", "CDS")]))
emb = pipe.embed(sequences)                             # (n, d), L2-normalised
hits = pipe.search("budding yeast transfer rna", emb, top=10)
```

## Input formats

**FASTA** headers are read in three conventions:

```text
>id|species=Homo sapiens|biotype=lncRNA|description=...|chrom=chr2|split=test     key=value (any extra keys are kept)
>Homo sapiens|snRNA|U1 spliceosomal RNA                                         chromosample positional
>id free-text description                                                       plain
```

`chrom` enables chromosome splits, `split=train|validation|test` fixes the split, and `lineage=...` adds taxonomy
words to the caption (a built-in table covers common model organisms). **BED**, **GFF3** and **GTF** need
`--reference genome.fa`; the biotype comes from `biotype`/`gene_biotype`/`transcript_type`-style attributes, or the
feature type. **GenBank** files carry their own sequence. Compatible with
[chromosample](https://github.com/baudrly/chromosample).

## How it works

* **Captions.** Each sequence is described by its species (expanded to a short lineage, e.g. *homo sapiens human
  primate mammal…*) and its biotype (expanded to synonyms, e.g. *small nuclear rna snrna spliceosomal*), optionally
  followed by its description. Shared words let the text side compose prompts for species × biotype pairs it never saw.
* **Encoders.** The default DNA encoder is a dilated residual CNN on one-hot DNA whose pooled features are summed over
  both strands, so a sequence and its reverse complement get exactly the same embedding. Padding is masked, so a
  sequence's embedding never depends on the batch it is in.
* **Loss.** Many sequences share one caption. CLOP contrasts each sequence with the *distinct* captions of its batch
  (multi-positive InfoNCE). The CLIP loss gives the same model (identical captions give identical logits, so it only adds
  a caption-frequency prior), but per-row retrieval metrics become meaningless: v1's “very low” retrieval scores were
  this artefact. Retrieval is scored against labels.
* **Length invariance.** Classes differ in length (tRNAs are short, introns long), so a model that can see length can
  name the class without reading the sequence. CLOP closes this in the data and in the model. `clop prepare` cuts fixed
  windows. Training crops are drawn log-uniformly between 100 and 500 bp, independently of the label. The CNN pools
  the mean and mean square of its features over *interior* positions only (whose receptive field lies inside the
  sequence): both are densities whose expectation does not change with length, unlike a max (which grows with length)
  or edge positions (whose share is ~1/length). Inputs longer than the training window are embedded as the mean of
  half-overlapping windows, by `Pipeline.embed` and by the browser reader alike.
* **Leakage control.** Splits keep whole chromosomes (or contiguous blocks of single-contig genomes) apart, and random
  splits keep a sequence and its reverse complement together.
* **Evaluation.** Species, biotype *given the true species*, and the species × biotype cell are scored separately
  (macro F1 with bootstrap intervals), by text prompts (zero-shot), by nearest training neighbours, and by both;
  k-mer logistic regression on the same windows is the baseline to beat.

## Results

Six genomes (*E. coli* to human), 16 sequence classes, 500 bp windows, whole chromosomes held out
(44,258 training / 14,187 test windows). Test macro F1, default model (251k parameters, 68 min on two CPU cores):

| | species | biotype within species | species × biotype |
|---|---|---|---|
| CLOP 2, prompts only | 0.884 | 0.540 | 0.392 |
| CLOP 2, prompts + k-NN | **0.887** | **0.549** | **0.400** (0.392–0.407) |
| 4-mer logistic regression | 0.840 | 0.484 | 0.322 (0.313–0.331) |
| 6-mer logistic regression | 0.819 | 0.440 | 0.302 |

**Length.** Trained on whole kiwifruit CDS and introns, where length alone gives macro F1 0.73, and tested on their
central 100 bp, a CNN with mean+max pooling shifts 17% of its calls towards one class (total-variation distance
between called and true class mix, three seeds); with CLOP's interior moment pooling the shift is 4%, like 4-mer
profiles (7%), which cannot see length. On composition-preserving shuffles, a linear probe reads log length off
mean+max pooled features with R² 0.94 and off moment-pooled features with R² 0.00.

Ablations at 300 windows per cell: moment pooling beats mean+max pooling (cell F1 0.339 vs 0.308); summing both
strands lifts cell F1 from 0.316 to 0.339; the CLIP loss and the multi-positive loss give the same model (0.342 vs
0.339); the v1 BiLSTM reaches 0.303 with 4× the parameters. On three genomes never seen in training, species-agnostic
prompts (`clop classify --pooled biotype`) name biotypes better than 4-mers in *B. fragilis* (65.2% vs 57.0%) and the
*A. thaliana* chloroplast (41.9% vs 34.8%).