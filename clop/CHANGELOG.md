# Changelog

## 2.1.0 (2026-11)

Length invariance: CLOP no longer learns sequence length, even from data where length names the class.

### Changed
- CNN pooling (`--pooling moments`, the default): the mean and mean square of each channel over *interior* positions,
  those whose receptive field lies inside the sequence. The previous mean+max pooling sensed length twice: a max grows
  with the number of positions, and positions near the ends (which see padding) weigh ~1/length in a mean. On
  composition-preserving shuffles, a linear probe reads log length off mean+max pooled features with R² 0.94 and off
  moment-pooled features with R² ≈ 0 (100–500 bp), and moment pooling is also more accurate (cell F1 0.339 vs 0.308
  at 300 windows per cell). `--pooling meanmax` keeps the old behaviour.
- Training crops (`--crop`, default `100 500`): two or more values now span a log-uniform range, drawn independently of
  the label. Crops are on by default.
- Inputs longer than the training window are embedded as the normalised mean of half-overlapping windows
  (`Pipeline.embed`, the browser reader), instead of being truncated at `--max-length`. The window is stored in the
  checkpoint (`window`; `--window` to override).
- `clop train` reports length dependence in `metrics.json` (`length`): zero-shot metrics on 50-500 bp crops, the drift
  of calls from the true class mix, and linear length probes of the embedding and of the pooled features.
- The web reader opens from disk: model and atlas ship as script files (`clop-model.js`, `clop-atlas.js`) instead
  of JSON fetched over HTTP, which browsers block on `file://`. `clop-reader.html` is the whole reader in one file.
- The web reader was rebuilt. It reads multi-record FASTA (a table of calls, TSV download) and scans long sequences
  (biotype probabilities along the sequence, with an annotation strip when the header gives segments). It checks the
  reverse complement, a shuffle and centred crops; builds classifiers from your own prompts and colours the map by them;
  has a similarity histogram for text search, prompt labels in the 3D map, and confidence colouring. It ships 132
  annotated examples and eight probes, and reads up to 50 kb.

### Results (scripts/benchmark.py)
- Trained on whole kiwifruit CDS and introns (length alone: macro F1 0.73) and tested on their central 100 bp, calls of
  mean+max models drift 0.17 (total variation) from the true class mix; moment-pooled models 0.04, like 4-mer profiles
  (0.07).
- Main model on the six-genome atlas: test macro F1 species 0.887, biotype within species 0.549, cell 0.400
  (2.0.0: 0.882, 0.523, 0.374).

## 2.0.0 (2026-11)

A rewrite of the 2025 proof of concept as a package. The command line changed (`clop <command>`); v1 flags are still
accepted and mapped onto `clop train` with a deprecation warning.

### Fixed
- FASTA headers in the documented `>id|species=...|biotype=...` form were split positionally, so the *id* became the
  species and `species=...` the biotype. All three conventions (key=value, chromosample positional, plain) now parse.
- BED: attribute columns (`biotype=...`) were ignored and minus-strand features were not reverse-complemented.
  GFF3: species and description handling; URL-encoded attributes are decoded; GTF attributes are read.
- The unit tests did not run: `unittest.makeSuite` is gone in Python 3.13, one test referenced an undefined
  `SequenceEncoder`, and three tests asserted wrong sequences for 1-based GFF and BED slices.
- `--run_dummy_example` crashed while writing the PDF report (`Helvetica.ttf` not found). Reports are now
  self-contained HTML; the PDF dependency (`fpdf2`) is gone.
- ONNX export only worked through an LSTM-specific wrapper, and exported encoders were never checked against PyTorch;
  the TensorFlow.js path depended on the unmaintained `onnx-tf`.
- The README pointed at `python clop.py`, which no longer existed after packaging; the package required Python < 3.12;
  the container shipped PyTorch 2.0.1 without the package's own entry point.


### Changed (methodology)
- Captions are shared by many sequences (98.9% of rows in a 256-row batch of the benchmark atlas share their caption),
  which made v1's per-row retrieval scores (MRR, Recall@K) meaningless. Retrieval is now scored against labels. The default
  loss is multi-positive InfoNCE over distinct captions; it trains the same model as the CLIP loss (which differs only by a
  caption-frequency prior), still available with `--loss clip`, as is `--loss sigmoid`.
- Whole features let sequence length predict the biotype (macro F1 0.66–0.93 from length alone on three genomes).
  `clop prepare` cuts fixed windows; splits keep whole chromosomes apart; random splits keep reverse complements together.
- The BiLSTM no longer packs sequences: packed batches with varying per-step sizes made CPU training ~30x slower.
  Forward and backward directions run on length-reversed inputs, exactly as before but at fixed shapes.
- Default DNA encoder: strand-invariant dilated CNN. The k-mer BiLSTM (v1), a k-mer Transformer and a chaos-game (FCGR)
  CNN are available with `--dna-encoder`.

### Added
- Inputs: GTF and GenBank (no Biopython); windowing, promoter and intergenic sampling, deduplication, caps (`clop prepare`).
- Commands: `prepare`, `train`, `evaluate`, `embed`, `classify` (with composed prompts), `search`, `export`, `report`, `demo`.
- Evaluation: species, biotype-within-species and cell macro F1 with bootstrap intervals, by prompts, k-NN and both;
  text→DNA mAP and precision@k; DNA→text MRR; silhouette, k-means ARI, modality gap; k-mer baselines.
- Interactive HTML reports with a WebGL 3D embedding atlas; publication figures.
- `clop export --web`: a static site that runs the model in plain JavaScript (matches PyTorch to ~1e-7).
- Resumable training, early stopping on validation zero-shot F1, AMP, `torch.compile`, per-epoch deterministic shuffling.
- Benchmark, figure and number scripts for the paper; CI for tests, the container and the web reader.
