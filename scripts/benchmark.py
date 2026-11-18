#!/usr/bin/env python3
"""The CLOP v2 benchmark: encoder and loss ablations, k-mer baselines, the main model, out-of-atlas transfer and the
length-leakage and length-shortcut checks. Every step is resumable: finished arms (metrics.json present) are skipped.

    python scripts/benchmark.py --atlas data/atlas6.fa.gz --transfer data/transfer --out results [--only main ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from clop.baselines import KmerLogistic, kmer_profiles, knn_predict
from clop.evaluate import align_scores, hierarchical_metrics, label_bias, macro_f1, retrieval_metrics, strip_predictions
from clop.io import load_records, parse_fasta
from clop.models import ModelConfig
from clop.pipeline import Pipeline
from clop.text import caption
from clop.train import TrainConfig
from clop.workflow import label_space, prepare_records, train_model

logger = logging.getLogger("benchmark")
CROPS = [100, 500]  # log-uniform crop range
ABLATION_CAP, MAIN_CAP = 300, 1200

# name -> (model config, train config, training cap)
ARMS = {
    "cnn": (dict(dna_encoder="cnn", width=96, depth=3), dict(loss="multi_positive", crop_lengths=CROPS), ABLATION_CAP),
    "cnn-clip-loss": (dict(dna_encoder="cnn", width=96, depth=3), dict(loss="clip", crop_lengths=CROPS), ABLATION_CAP),
    "cnn-sigmoid-loss": (dict(dna_encoder="cnn", width=96, depth=3), dict(loss="sigmoid", crop_lengths=CROPS), ABLATION_CAP),
    "cnn-no-rc": (
        dict(dna_encoder="cnn", width=96, depth=3, rc=False),
        dict(loss="multi_positive", crop_lengths=CROPS, random_rc=True),
        ABLATION_CAP,
    ),
    "cnn-no-crop": (dict(dna_encoder="cnn", width=96, depth=3), dict(loss="multi_positive", crop_lengths=[]), ABLATION_CAP),
    "cnn-meanmax": (
        dict(dna_encoder="cnn", width=96, depth=3, pooling="meanmax"),
        dict(loss="multi_positive", crop_lengths=[200, 500]),
        ABLATION_CAP,
    ),
    "fcgr": (dict(dna_encoder="fcgr", width=96, k=6), dict(loss="multi_positive", crop_lengths=CROPS), ABLATION_CAP),
    "transformer": (
        dict(dna_encoder="transformer", width=96, k=6, stride=3, layers=2, heads=4),
        dict(loss="multi_positive", crop_lengths=CROPS, epochs=8),
        ABLATION_CAP,
    ),
    "lstm": (
        dict(dna_encoder="lstm", width=128, k=6, layers=1, rc=False),
        dict(loss="multi_positive", crop_lengths=CROPS),
        ABLATION_CAP,
    ),
    "lstm-clip-loss (CLOP v1)": (
        dict(dna_encoder="lstm", width=128, k=6, layers=1, rc=False),
        dict(loss="clip", crop_lengths=CROPS),
        ABLATION_CAP,
    ),
    "main": (
        dict(dna_encoder="cnn", width=96, depth=3),
        dict(loss="multi_positive", crop_lengths=CROPS, epochs=16, patience=5),
        MAIN_CAP,
    ),
}


def slug(name: str) -> str:
    return name.split(" ")[0]


def run_arm(name: str, records, out: Path, device: str) -> dict:
    target = out / "runs" / slug(name)
    if (target / "metrics.json").exists():
        return json.loads((target / "metrics.json").read_text())
    model_kw, train_kw, cap = ARMS[name]
    parts = prepare_records(list(records), "header", cap=cap)
    train_cfg = TrainConfig(**{"epochs": 10, "batch_size": 256, "lr": 2e-3, "patience": 3, **train_kw})
    started = time.time()
    summary = train_model(parts, ModelConfig(**model_kw), train_cfg, target, device)
    summary["wall_seconds"] = round(time.time() - started)
    (target / "metrics.json").write_text(json.dumps(summary, indent=1, default=float))
    logger.info("%s done in %.0f min", name, summary["wall_seconds"] / 60)
    return summary


def baselines(records, out: Path) -> dict:
    target = out / "baselines.json"
    if target.exists():
        return json.loads(target.read_text())
    results = {}
    for cap, tag in ((ABLATION_CAP, "ablation"), (MAIN_CAP, "main")):
        parts = prepare_records(list(records), "header", cap=cap)
        train, test = parts["train"], parts["test"]
        labels = label_space(train)
        names = list(labels)
        test = [r for r in test if r.cell in labels]
        truth = [r.cell for r in test]
        y_train = [r.cell for r in train]
        counts = Counter(y_train)
        majority = np.zeros((len(test), len(names)))
        majority[:, names.index(counts.most_common(1)[0][0])] = 1
        results[f"majority/{tag}"] = strip_predictions(hierarchical_metrics(truth, majority, names))
        for k in (4, 6):
            model = KmerLogistic(k=k, C=0.5).fit([r.sequence for r in train], y_train)
            scores = align_scores(model.classes_, model.predict_proba([r.sequence for r in test]), names)
            results[f"{k}-mer logistic/{tag}"] = strip_predictions(hierarchical_metrics(truth, scores, names, ci=tag == "main"))
            logger.info(
                "baseline %d-mer logistic (%s): cell F1 %.3f", k, tag, results[f"{k}-mer logistic/{tag}"]["cell_macro_f1"]
            )
        ref, query = kmer_profiles([r.sequence for r in train], 4), kmer_profiles([r.sequence for r in test], 4)
        classes, votes = knn_predict(ref - ref.mean(0), y_train, query - ref.mean(0), 15)
        results[f"4-mer kNN/{tag}"] = strip_predictions(hierarchical_metrics(truth, align_scores(classes, votes, names), names))
    target.write_text(json.dumps(results, indent=1, default=float))
    return results


def transfer(out: Path, transfer_dir: Path, device: str) -> dict:
    """The main model on genomes it never saw.

    (i) Open set: the nearest of the 44 training cells (species and biotype calls by the training prompts).
    (ii) Closed set: prompts composed from the genome's species name and the biotypes present in it (chance 1/K), the
    way a user would ask "is this window CDS, intron, intergenic or a promoter?"; rRNA, a biotype absent from training,
    is included where present. (iii) The same two questions for 4-mer logistic regression trained on the atlas
    (restricted to the present biotypes it knows)."""
    target = out / "transfer.json"
    if target.exists():
        return json.loads(target.read_text())
    pipe = Pipeline.load(out / "runs" / "main" / "model.pt", device)
    trained = dict(pipe.labels)
    trained_biotypes = sorted({n.split("|")[1] for n in trained})
    atlas_train = [r for r in parse_fasta(Path(ARGS.atlas)) if r.split == "train"]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(atlas_train), min(20000, len(atlas_train)), replace=False)
    kmer = KmerLogistic(4, C=0.5).fit([atlas_train[i].sequence for i in pick], [atlas_train[i].biotype for i in pick])
    results = {}
    for path in sorted(transfer_dir.glob("*.fa")):
        records = parse_fasta(path)
        species = records[0].species
        emb = pipe.embed([r.sequence for r in records])
        truth = np.array([r.biotype for r in records])
        present = sorted(set(truth.tolist()))
        gc = float(np.mean([(r.sequence.count("G") + r.sequence.count("C")) / max(len(r.sequence), 1) for r in records]))
        entry = {"species": species, "n": len(records), "gc": gc, "biotypes": dict(Counter(truth.tolist()))}
        pipe.set_labels(trained)  # (i)
        called = np.array(pipe.label_names)[pipe.scores(emb).argmax(1)]
        entry["called_species"] = dict(Counter(c.split("|")[0] for c in called).most_common())
        seen = np.isin(truth, trained_biotypes)
        open_biotype = np.array([c.split("|")[1] for c in called])
        entry["open_biotype_accuracy"] = float((open_biotype[seen] == truth[seen]).mean())
        pipe.set_labels({f"{species}|{b}": caption(species, b) for b in present})  # (ii)
        closed = np.array([n.split("|")[1] for n in pipe.label_names])[pipe.scores(emb).argmax(1)]
        entry["closed_biotype_accuracy"] = float((closed == truth).mean())
        entry["closed_biotype_macro_f1"] = macro_f1(truth, closed)
        entry["closed_confusion"] = {b: dict(Counter(closed[truth == b].tolist()).most_common()) for b in present}
        # (ii-b) species-agnostic prompts: each biotype's training prompts pooled over species (biotype words alone for
        # biotypes absent from training, e.g. rRNA)
        pipe.set_labels(trained)
        pipe.pool_labels("biotype", present)
        keep = [i for i, n in enumerate(pipe.label_names) if n in present]
        pooled_call = np.array(pipe.label_names)[keep][(emb @ pipe.prototypes[keep].T).argmax(1)]
        entry["pooled_biotype_accuracy"] = float((pooled_call == truth).mean())
        entry["pooled_confusion"] = {b: dict(Counter(pooled_call[truth == b].tolist()).most_common()) for b in present}
        entry["closed_chance"] = 1.0 / len(present)
        entry["retrieval_closed"] = retrieval_metrics(emb, [r.cell for r in records], pipe.prototypes, pipe.label_names)
        proba = kmer.predict_proba([r.sequence for r in records])  # (iii)
        known = [b for b in present if b in kmer.classes_]
        cols = [list(kmer.classes_).index(b) for b in known]
        kmer_closed = np.array(known)[proba[:, cols].argmax(1)]
        entry["kmer_closed_biotype_accuracy"] = float((kmer_closed[np.isin(truth, known)] == truth[np.isin(truth, known)]).mean())
        entry["closed_biotype_accuracy_known"] = float((closed[np.isin(truth, known)] == truth[np.isin(truth, known)]).mean())
        entry["pooled_biotype_accuracy_known"] = float(
            (pooled_call[np.isin(truth, known)] == truth[np.isin(truth, known)]).mean()
        )
        entry["kmer_open_biotype_accuracy"] = float((kmer.classes_[proba.argmax(1)][seen] == truth[seen]).mean())
        results[path.stem] = entry
        logger.info(
            "transfer %s: GC %.2f, called %s; closed-set biotype accuracy: composed %.3f, pooled %.3f, 4-mer %.3f (chance %.2f)",
            path.stem,
            gc,
            list(entry["called_species"].items())[:2],
            entry["closed_biotype_accuracy_known"],
            entry["pooled_biotype_accuracy_known"],
            entry["kmer_closed_biotype_accuracy"],
            entry["closed_chance"],
        )
    target.write_text(json.dumps(results, indent=1, default=float))
    return results


def length_leakage(transfer_dir: Path, sources: Path, out: Path) -> dict:
    """How well sequence length alone names the biotype when features are taken whole (as CLOP v1 did) rather than
    as fixed windows: a gradient-boosted classifier on log length, 5-fold cross-validated."""
    target = out / "length_leakage.json"
    if target.exists():
        return json.loads(target.read_text())
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import cross_val_predict

    sets = {
        "Actinidia chinensis (GFF3)": load_records(
            sources / "actinidia_chinensis" / "genome.gff3.gz",
            "gff3",
            sources / "actinidia_chinensis" / "genome.fasta.gz",
            ["CDS", "intron"],
        ),
        "Bacteroides fragilis (GenBank)": load_records(
            sources / "bacteroides_fragilis" / "genome.gbff.gz", "genbank", feature_types=["CDS", "tRNA", "rRNA"]
        ),
        "A. thaliana chloroplast (GenBank)": load_records(
            sources / "NC_000932.gb", "genbank", feature_types=["CDS", "tRNA", "rRNA"]
        ),
    }
    results = {}
    for name, records in sets.items():
        y = np.array([r.biotype for r in records])
        x = np.log10(np.array([[max(len(r.sequence), 1)] for r in records]))
        keep = np.isin(y, [c for c, n in Counter(y).items() if n >= 10])
        pred = cross_val_predict(HistGradientBoostingClassifier(max_depth=3), x[keep], y[keep], cv=5)
        results[name] = {
            "n": int(keep.sum()),
            "classes": dict(Counter(y[keep].tolist())),
            "length_only_macro_f1": float(f1_score(y[keep], pred, average="macro")),
            "chance_macro_f1": float(f1_score(y[keep], np.random.default_rng(0).permutation(y[keep]), average="macro")),
            "median_length": {
                c: float(np.median([len(r.sequence) for r, k in zip(records, y) if k == c])) for c in np.unique(y[keep])
            },
        }
        logger.info("length leakage %s: %s", name, results[name]["length_only_macro_f1"])
    target.write_text(json.dumps(results, indent=1))
    return results


# name -> (run directory, model config, training crops)
SHORTCUT_ARMS = {
    "mean+max pooling": ("meanmax", dict(pooling="meanmax"), [200, 500]),
    "moment pooling (CLOP 2)": ("moments", dict(), [100, 500]),
}


def shortcut(sources: Path, out: Path, device: str, probe: int = 100, seeds=(0, 1, 2)) -> dict:
    """Train on whole features, where length alone separates the classes (kiwifruit CDS vs intron), then test on whole
    held-out features and on equal-length ``probe`` bp centres of the same features, where length carries nothing. A
    model that learned the length shortcut loses on the second test what it gained on the first, and its equal-length
    calls drift towards the class that owned that length. CLOP arms are trained with several seeds."""
    target = out / "shortcut.json"
    if target.exists():
        return json.loads(target.read_text())
    from sklearn.ensemble import HistGradientBoostingClassifier

    records = load_records(
        sources / "actinidia_chinensis" / "genome.gff3.gz",
        "gff3",
        sources / "actinidia_chinensis" / "genome.fasta.gz",
        ["CDS", "intron"],
    )
    parts = prepare_records(records, "chromosome")
    test = [r for r in parts["test"] if len(r.sequence) >= probe]
    inputs = {
        "whole": [r.sequence for r in test],
        "equal_length": [r.sequence[(len(r.sequence) - probe) // 2 :][:probe] for r in test],
    }
    truth = [r.cell for r in test]

    def score(calls: dict) -> dict:  # macro F1 on both tests, and how far equal-length calls drift from the truth mix
        return {key: macro_f1(truth, pred) for key, pred in calls.items()} | {
            "equal_length_bias": label_bias(truth, calls["equal_length"])
        }

    length_model = HistGradientBoostingClassifier(max_depth=3).fit(
        np.log10([[len(r.sequence)] for r in parts["train"]]), [r.cell for r in parts["train"]]
    )
    kmer = KmerLogistic(4, C=0.5).fit([r.sequence for r in parts["train"]], [r.cell for r in parts["train"]])
    results = {
        "n": {s: len(v) for s, v in parts.items()} | {"test_scored": len(test)},
        "median_length": {b: float(np.median([len(r.sequence) for r in records if r.biotype == b])) for b in ("CDS", "intron")},
        "length only": score({k: length_model.predict(np.log10([[len(x)] for x in v])) for k, v in inputs.items()}),
        "4-mer logistic": score({k: kmer.classes_[kmer.predict_proba(v).argmax(1)] for k, v in inputs.items()}),
    }
    for name, (directory, model_kw, crops) in SHORTCUT_ARMS.items():
        runs = []
        for seed in seeds:
            run = out / "shortcut" / f"{directory}-s{seed}"
            if not (run / "model.pt").exists():
                train_cfg = TrainConfig(epochs=10, patience=3, crop_lengths=crops, seed=seed)
                train_model(parts, ModelConfig(dna_encoder="cnn", width=96, depth=3, **model_kw), train_cfg, run, device)
            pipe = Pipeline.load(run / "model.pt", device)
            names = np.array(pipe.label_names)
            runs.append(score({k: names[pipe.scores(pipe.embed(v)).argmax(1)] for k, v in inputs.items()}))
        results[name] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]} | {
            f"{k}_sd": float(np.std([r[k] for r in runs])) for k in runs[0]
        }
        logger.info("shortcut %s: %s", name, results[name])
    target.write_text(json.dumps(results, indent=1))
    return results


def main() -> None:
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--transfer", required=True, help="directory of transfer FASTA files")
    parser.add_argument("--sources", help="raw transfer sources (for the length-leakage check)")
    parser.add_argument("--out", default="results")
    parser.add_argument("--only", nargs="*", help="subset of steps: arm names, baselines, transfer, leakage, shortcut")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    ARGS = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("fontTools", "matplotlib", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    torch.set_num_threads(ARGS.threads)
    out = Path(ARGS.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = ARGS.only or ["leakage", "main", "transfer", "baselines", "shortcut", *[a for a in ARMS if a != "main"]]
    records = parse_fasta(ARGS.atlas)
    summary_path = out / "benchmark.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for step in steps:
        if step == "baselines":
            summary["baselines"] = baselines(records, out)
        elif step == "transfer":
            summary["transfer"] = transfer(out, Path(ARGS.transfer), ARGS.device)
        elif step == "leakage" and ARGS.sources:
            summary["length_leakage"] = length_leakage(Path(ARGS.transfer), Path(ARGS.sources), out)
        elif step == "shortcut" and ARGS.sources:
            summary["shortcut"] = shortcut(Path(ARGS.sources), out, ARGS.device)
        elif step in ARMS:
            summary.setdefault("arms", {})[step] = run_arm(step, records, out, ARGS.device)
        summary_path.write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
