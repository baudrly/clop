"""Command-line interface: ``clop <command>``. Run ``clop <command> -h`` for options.

Commands: prepare, train, evaluate, embed, classify, search, export, report, demo.
The CLOP v1 single-command flags (``--input_file ...``) are still accepted and mapped onto ``train``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

logger = logging.getLogger("clop")


def _threads(n: int) -> None:
    import torch

    if n > 0:
        torch.set_num_threads(n)


def _device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _input_args(p: argparse.ArgumentParser, required: bool = True) -> None:
    p.add_argument("--input", "-i", nargs="+", required=required, help="FASTA, BED, GFF3/GTF or GenBank files")
    p.add_argument("--input-type", default="auto", choices=["auto", "fasta", "bed", "gff3", "gtf", "genbank"])
    p.add_argument("--reference", help="reference FASTA for BED/GFF3/GTF input")
    p.add_argument("--feature-types", help="comma-separated GFF/GenBank feature types to keep")
    p.add_argument("--species", default="", help="species name when the input does not carry one")


def _read(args) -> list:
    from .io import load_records

    types = args.feature_types.split(",") if getattr(args, "feature_types", None) else None
    records = []
    for path in args.input:
        records.extend(load_records(path, args.input_type, args.reference, types, args.species))
    if not records:
        sys.exit("no records read from the input")
    return records


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="auto", help="cpu, cuda, mps or auto")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0: library default)")
    p.add_argument("--verbose", "-v", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    from .losses import LOSSES

    parser = argparse.ArgumentParser(prog="clop", description="Contrastive Language-Omics Pre-training")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("prepare", help="build a length-controlled, deduplicated, split-annotated FASTA from annotations")
    _input_args(p)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--window", type=int, default=500, help="window size in bp (0: keep whole features)")
    p.add_argument("--anchor", default="center", choices=["center", "start"])
    p.add_argument("--window-types", default="", help="feature types to window (default: all read features)")
    p.add_argument("--promoters", default="", help="feature types whose 5' ends define promoter windows")
    p.add_argument("--promoter-offset", type=int, default=0, help="bp upstream of the feature start")
    p.add_argument("--intergenic", type=int, default=0, help="number of intergenic windows to sample")
    p.add_argument("--intergenic-margin", type=int, default=1000, help="minimum distance of intergenic windows from features")
    p.add_argument("--split", default="auto", choices=["auto", "header", "chromosome", "random", "none"])
    p.add_argument("--fractions", type=float, nargs=2, default=(0.15, 0.15), metavar=("VALIDATION", "TEST"))
    p.add_argument("--cap", type=int, default=0, help="maximum records per species x biotype (0: no cap)")
    p.add_argument("--rename", default="", help="biotype renames, e.g. 'gene_CDS=CDS,mRNA=transcript'")
    p.add_argument("--seed", type=int, default=0)
    _common(p)

    p = sub.add_parser("train", help="train a model; writes model.pt, metrics.json, report.html")
    _input_args(p)
    p.add_argument("--output", "-o", required=True, help="run directory")
    g = p.add_argument_group("model")
    g.add_argument("--dna-encoder", default="cnn", choices=["cnn", "transformer", "lstm", "fcgr"])
    g.add_argument("--text-encoder", default="bow", choices=["bow", "transformer"])
    g.add_argument("--embed-dim", type=int, default=128)
    g.add_argument("--width", type=int, default=96)
    g.add_argument("--depth", type=int, default=3)
    g.add_argument("--kernel", type=int, default=15)
    g.add_argument("--k", type=int, default=6, help="k-mer size (transformer, lstm, fcgr)")
    g.add_argument("--stride", type=int, default=1, help="k-mer stride (transformer, lstm)")
    g.add_argument("--layers", type=int, default=2)
    g.add_argument("--heads", type=int, default=4)
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--no-rc", action="store_true", help="do not make the DNA encoder strand-invariant")
    g.add_argument(
        "--pooling", choices=("moments", "meanmax"), default="moments", help="cnn pooling (meanmax is length-sensitive)"
    )
    g.add_argument("--window", type=int, default=0, help="tile longer inputs into windows of this size (0: longest crop)")
    g.add_argument("--max-length", type=int, default=4096, help="longest sequence read, in bp")
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--lr", type=float, default=2e-3)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--loss", default="multi_positive", choices=LOSSES)
    g.add_argument(
        "--crop",
        type=int,
        nargs="*",
        default=[100, 500],
        help="training crop lengths: one value is fixed, two or more a log-uniform range (no value: no crops)",
    )
    g.add_argument("--random-rc", action="store_true", help="random strand flips (for encoders built with --no-rc)")
    g.add_argument("--patience", type=int, default=5)
    g.add_argument("--amp", action="store_true")
    g.add_argument("--compile", action="store_true")
    g.add_argument("--workers", type=int, default=0)
    g.add_argument("--seed", type=int, default=0)
    g = p.add_argument_group("data")
    g.add_argument("--split", default="auto", choices=["auto", "header", "chromosome", "random"])
    g.add_argument("--fractions", type=float, nargs=2, default=(0.15, 0.15), metavar=("VALIDATION", "TEST"))
    g.add_argument("--cap", type=int, default=0, help="maximum training records per species x biotype")
    g.add_argument("--min-train", type=int, default=10, help="drop cells with fewer training records")
    g.add_argument("--use-description", action="store_true", help="append descriptions to captions")
    g.add_argument("--no-report", action="store_true")
    _common(p)

    p = sub.add_parser("evaluate", help="metrics of a trained model on annotated sequences")
    p.add_argument("--model", "-m", required=True)
    _input_args(p)
    p.add_argument("--reference-set", nargs="*", help="annotated sequences for k-NN (e.g. the training FASTA)")
    p.add_argument(
        "--labels",
        choices=["model", "input"],
        default="model",
        help="prompt set: the model's training cells, or the cells present in the input (zero-shot transfer)",
    )
    p.add_argument("--output", "-o", help="metrics JSON (default: stdout)")
    _common(p)

    p = sub.add_parser("embed", help="DNA embeddings to .npz, .csv or .parquet")
    p.add_argument("--model", "-m", required=True)
    _input_args(p)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--batch-size", type=int, default=512)
    _common(p)

    p = sub.add_parser("classify", help="zero-shot species and biotype calls (TSV)")
    p.add_argument("--model", "-m", required=True)
    _input_args(p)
    p.add_argument("--prompts", help="TSV of label<TAB>prompt text replacing the model's label space")
    p.add_argument(
        "--compose",
        nargs=2,
        metavar=("SPECIES", "BIOTYPES"),
        help="compose prompts from comma-separated species and biotypes, e.g. 'Mus musculus' 'CDS,lncRNA'",
    )
    p.add_argument(
        "--pooled",
        choices=["biotype", "species"],
        help="one species-agnostic prompt per biotype (or per species); suits genomes absent from training",
    )
    p.add_argument("--extra-biotypes", default="", help="with --pooled biotype: comma-separated biotypes to add as prompts")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--output", "-o", help="TSV (default: stdout)")
    _common(p)

    p = sub.add_parser("search", help="rank sequences by similarity to a free-text query")
    p.add_argument("--model", "-m", required=True)
    _input_args(p)
    p.add_argument("--query", "-q", required=True, nargs="+")
    p.add_argument("--top", type=int, default=10)
    _common(p)

    p = sub.add_parser("export", help="ONNX encoders, web bundle for the in-browser reader")
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--output", "-o", required=True, help="output directory")
    p.add_argument("--onnx", action="store_true")
    p.add_argument("--web", action="store_true", help="write the static web reader (opens from disk too; cnn + bow models)")
    p.add_argument("--atlas", nargs="*", help="annotated sequences to embed into the web atlas")
    p.add_argument("--atlas-size", type=int, default=6000)
    _common(p)

    p = sub.add_parser("report", help="(re)build report.html and figures for a run directory")
    p.add_argument("run", help="run directory written by `clop train`")
    _common(p)

    p = sub.add_parser("demo", help="train a tiny model on synthetic sequences in a few seconds")
    p.add_argument("--output", "-o", default="clop_demo")
    _common(p)
    return parser


# ----------------------------------------------------------------------------------------------- commands
def cmd_prepare(args) -> None:
    from dataclasses import replace

    from .data import assign_splits, cap_per_cell
    from .io import (
        Reference,
        centred_window,
        deduplicate,
        intergenic_windows,
        parse_genbank,
        promoter_windows,
        summarise,
        write_fasta,
    )

    records = _read(args)
    renames = dict(pair.split("=", 1) for pair in args.rename.split(",") if "=" in pair)
    if renames:
        records = [replace(r, biotype=renames.get(r.biotype, r.biotype)) for r in records]
    need_reference = args.window or args.promoters or args.intergenic
    reference = None
    if need_reference:
        if args.reference:
            reference = Reference.from_fasta(args.reference, args.species)
        elif all(str(p).lower().removesuffix(".gz").endswith((".gb", ".gbk", ".gbff")) for p in args.input):
            sequences, meta = {}, {}
            for path in args.input:
                ref = parse_genbank(path, (), args.species)[1]
                sequences.update(ref.sequences)
                meta.update(ref.meta)
            reference = Reference(sequences, meta, args.species)
        else:
            sys.exit("windows, promoters and intergenic sampling need --reference (or GenBank input)")
    keep = set(args.window_types.split(",")) if args.window_types else None
    out = [r for r in records if keep is None or r.meta.get("feature_type", r.biotype) in keep]
    if args.window:
        out = [centred_window(r, reference, args.window, args.anchor) for r in out if r.chrom]
    if args.promoters:
        kinds = set(args.promoters.split(","))
        sources = [r for r in records if r.meta.get("feature_type", r.biotype) in kinds]
        out += promoter_windows(sources, reference, args.window or 500, args.promoter_offset)
    if args.intergenic:
        out += intergenic_windows(
            records, reference, args.window or 500, args.intergenic, args.intergenic_margin, args.species, args.seed
        )
    out = deduplicate(out, core=min(args.window, 100) if args.window else 0)
    out = [r for r in out if r.sequence.count("N") <= len(r.sequence) // 10]
    if args.split != "none":
        out = assign_splits(out, args.split, tuple(args.fractions), seed=args.seed)
    if args.cap:
        out = cap_per_cell(out, args.cap, args.seed)
    n = write_fasta(out, args.output)
    logger.info("wrote %d records to %s: %s", n, args.output, json.dumps(summarise(out)["biotypes"]))


def cmd_train(args) -> None:
    from .models import ModelConfig
    from .train import TrainConfig
    from .workflow import prepare_records, train_model

    records = _read(args)
    parts = prepare_records(records, args.split, tuple(args.fractions), args.cap, args.min_train, args.seed)
    model_cfg = ModelConfig(
        dna_encoder=args.dna_encoder,
        text_encoder=args.text_encoder,
        embed_dim=args.embed_dim,
        width=args.width,
        depth=args.depth,
        kernel=args.kernel,
        k=args.k,
        stride=args.stride,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        rc=not args.no_rc,
        pooling=args.pooling,
        max_length=args.max_length,
        window=args.window,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss=args.loss,
        crop_lengths=args.crop,
        random_rc=args.random_rc,
        patience=args.patience,
        amp=args.amp,
        compile=args.compile,
        workers=args.workers,
        seed=args.seed,
    )
    summary = train_model(parts, model_cfg, train_cfg, args.output, _device(args.device), args.use_description)
    test = summary.get("test", {})
    for arm in ("zero_shot", "knn", "ensemble"):
        if arm in test:
            m = test[arm]
            logger.info(
                "test %-9s species F1 %.3f | biotype F1 %.3f | cell F1 %.3f",
                arm,
                m["species_macro_f1"],
                m["biotype_macro_f1"],
                m["cell_macro_f1"],
            )
    if not args.no_report:
        from .report import write_report

        write_report(args.output)


def cmd_evaluate(args) -> None:
    from .evaluate import evaluate, strip_predictions
    from .io import load_records
    from .pipeline import Pipeline
    from .workflow import label_space

    pipe = Pipeline.load(args.model, _device(args.device))
    records = _read(args)
    if args.labels == "input":
        pipe.set_labels(label_space(records))
    reference = None
    if args.reference_set:
        reference = [r for p in args.reference_set for r in load_records(p)]
    result, _ = evaluate(pipe, [r for r in records if r.cell in pipe.labels], reference, ci=True)
    text = json.dumps(strip_predictions(result), indent=1, default=float)
    Path(args.output).write_text(text) if args.output else print(text)


def cmd_embed(args) -> None:
    import numpy as np

    from .pipeline import Pipeline

    pipe = Pipeline.load(args.model, _device(args.device))
    records = _read(args)
    emb = pipe.embed([r.sequence for r in records], args.batch_size)
    out = Path(args.output)
    if out.suffix == ".npz":
        np.savez_compressed(
            out,
            embeddings=emb,
            ids=np.array([r.id for r in records]),
            species=np.array([r.species for r in records]),
            biotypes=np.array([r.biotype for r in records]),
        )
    else:
        import pandas as pd

        frame = pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])
        frame.insert(0, "biotype", [r.biotype for r in records])
        frame.insert(0, "species", [r.species for r in records])
        frame.insert(0, "id", [r.id for r in records])
        frame.to_parquet(out, index=False) if out.suffix == ".parquet" else frame.to_csv(out, index=False)
    logger.info("embedded %d sequences -> %s", len(records), out)


def cmd_classify(args) -> None:
    from .pipeline import Pipeline

    pipe = Pipeline.load(args.model, _device(args.device))
    if args.prompts:
        rows = [line.rstrip("\n").split("\t", 1) for line in Path(args.prompts).read_text().splitlines() if line.strip()]
        pipe.set_labels({label: text for label, text in rows})
    elif args.compose:
        species, biotypes = (x.split(",") for x in args.compose)
        pipe.set_labels(pipe.labels_from_pairs([(s.strip(), b.strip()) for s in species for b in biotypes]))
    if args.pooled:
        pipe.pool_labels(args.pooled, [b for b in args.extra_biotypes.split(",") if b])
    records = _read(args)
    calls = pipe.classify([r.sequence for r in records], args.top)
    lines = ["id\ttop_label\tscore\tspecies\tspecies_score\tbiotype\tbiotype_score\talternatives"]
    for r, c in zip(records, calls):
        alternatives = ";".join(f"{x['label']}:{x['score']:.3f}" for x in c["top"][1:])
        lines.append(
            f"{r.id}\t{c['top'][0]['label']}\t{c['top'][0]['score']:.4f}\t{c['species'][0]['label']}\t"
            f"{c['species'][0]['score']:.4f}\t{c['biotype'][0]['label']}\t{c['biotype'][0]['score']:.4f}\t{alternatives}"
        )
    text = "\n".join(lines) + "\n"
    Path(args.output).write_text(text) if args.output else sys.stdout.write(text)


def cmd_search(args) -> None:
    from .pipeline import Pipeline

    pipe = Pipeline.load(args.model, _device(args.device))
    records = _read(args)
    emb = pipe.embed([r.sequence for r in records])
    for rank, (i, score) in enumerate(pipe.search(" ".join(args.query), emb, args.top), 1):
        r = records[i]
        print(f"{rank}\t{score:.4f}\t{r.id}\t{r.species}\t{r.biotype}")


def cmd_export(args) -> None:
    from .export import export_onnx, export_web
    from .pipeline import Pipeline

    pipe = Pipeline.load(args.model, "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.onnx:
        export_onnx(pipe, out)
    if args.web:
        atlas = None
        if args.atlas:
            from .io import load_records

            atlas = [r for p in args.atlas for r in load_records(p)]
        export_web(pipe, out, atlas, args.atlas_size)
    if not (args.onnx or args.web):
        sys.exit("nothing to export: pass --onnx and/or --web")


def cmd_report(args) -> None:
    from .report import write_report

    print(write_report(args.run))


def cmd_demo(args) -> None:
    from .demo import run_demo

    run_demo(args.output, _device(args.device))


COMMANDS = {
    "prepare": cmd_prepare,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "embed": cmd_embed,
    "classify": cmd_classify,
    "search": cmd_search,
    "export": cmd_export,
    "report": cmd_report,
    "demo": cmd_demo,
}

# CLOP v1 flags -> v2 train flags
_LEGACY = {
    "--input_file": "--input",
    "--input_type": "--input-type",
    "--reference_fasta": "--reference",
    "--output_dir": "--output",
    "--gff_feature_types": "--feature-types",
    "--batch_size": "--batch-size",
    "--learning_rate": "--lr",
    "--weight_decay": "--weight-decay",
    "--embedding_dim": "--embed-dim",
    "--hidden_dim": "--width",
    "--num_layers": "--layers",
    "--early_stopping_patience": "--patience",
    "--use_amp": "--amp",
    "--use_torch_compile": "--compile",
    "--encoder_type": "--dna-encoder",
    "--kmer_k": "--k",
    "--epochs": "--epochs",
    "--dropout": "--dropout",
}
_LEGACY_IGNORED = {
    "--dna_tokenizer_type",
    "--lr_scheduler_patience",
    "--export_embeddings_format",
    "--export_onnx",
    "--export_tfjs",
    "--resume_from_checkpoint",
}


def translate_legacy(argv: list[str]) -> list[str]:
    """Map a CLOP v1 command line onto ``clop train`` (with a deprecation warning)."""
    if "--test_suite" in argv:
        return ["test"]
    if "--run_dummy_example" in argv:
        out = argv[argv.index("--output_dir") + 1] if "--output_dir" in argv else "clop_demo"
        return ["demo", "--output", out]
    new, skip = ["train"], False
    for i, token in enumerate(argv):
        if skip:
            skip = False
            continue
        flag, _, inline = token.partition("=")
        if flag in _LEGACY_IGNORED:
            skip = not inline and i + 1 < len(argv) and not argv[i + 1].startswith("--")
            continue
        new.append(_LEGACY.get(flag, flag) + (f"={inline}" if inline else ""))
    if "--dna-encoder" not in new:
        new += ["--dna-encoder", "lstm"]
    warnings.warn("CLOP v1 flags are deprecated; this run is `clop " + " ".join(new) + "`", DeprecationWarning, stacklevel=2)
    return new


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("--") and argv[0] not in ("--version", "--help", "-h"):
        argv = translate_legacy(argv)
        if argv == ["test"]:
            import pytest

            sys.exit(pytest.main(["-q", str(Path(__file__).resolve().parents[2] / "tests")]))
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from . import __version__

        print(__version__)
        return
    if not args.command:
        parser.print_help()
        return
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("fontTools", "matplotlib", "numba", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _threads(args.threads)
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
