#!/usr/bin/env python3
"""Paper and poster figures from a benchmark results directory.

python scripts/make_figures.py results paper/figures --atlas data/atlas6.fa.gz --transfer data/transfer
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from clop import viz
from clop.io import parse_fasta
from clop.pipeline import Pipeline

SPECIES_ORDER = [
    "Escherichia coli",
    "Saccharomyces cerevisiae",
    "Arabidopsis thaliana",
    "Caenorhabditis elegans",
    "Drosophila melanogaster",
    "Homo sapiens",
]
SPECIES_COLOUR = dict(zip(SPECIES_ORDER, ("#868e96", "#fab005", "#2f9e44", "#15aabf", "#7950f2", "#ff5a36")))
FAMILY = {
    "CDS": "protein-coding",
    "five_prime_UTR": "UTRs",
    "three_prime_UTR": "UTRs",
    "intron": "introns & intergenic",
    "intergenic": "introns & intergenic",
    "promoter": "regulatory",
    "enhancer": "regulatory",
    "open_chromatin": "regulatory",
    "replication_origin": "regulatory",
    "tRNA": "RNA genes",
    "snRNA": "RNA genes",
    "snoRNA": "RNA genes",
    "miRNA": "RNA genes",
    "lncRNA": "RNA genes",
    "transposable_element": "repeats & pseudogenes",
    "pseudogene": "repeats & pseudogenes",
}
FAMILY_COLOUR = {
    "protein-coding": "#3b5bdb",
    "UTRs": "#12b886",
    "introns & intergenic": "#adb5bd",
    "regulatory": "#ff5a36",
    "RNA genes": "#be4bdb",
    "repeats & pseudogenes": "#fab005",
}
SHORT = {
    "Escherichia coli": "E. coli",
    "Saccharomyces cerevisiae": "S. cerevisiae",
    "Arabidopsis thaliana": "A. thaliana",
    "Caenorhabditis elegans": "C. elegans",
    "Drosophila melanogaster": "D. melanogaster",
    "Homo sapiens": "H. sapiens",
}
ARM_LABELS = {
    "main": "CLOP 2 (CNN, multi-positive, full data)",
    "cnn": "CNN, multi-positive",
    "cnn-clip-loss": "CNN, CLIP loss",
    "cnn-sigmoid-loss": "CNN, sigmoid loss",
    "cnn-no-rc": "CNN, strand augmentation only",
    "cnn-no-crop": "CNN, no random crops",
    "cnn-meanmax": "CNN, mean+max pooling",
    "fcgr": "FCGR image CNN",
    "transformer": "k-mer Transformer",
    "lstm": "k-mer BiLSTM, multi-positive",
    "lstm-clip-loss": "k-mer BiLSTM, CLIP loss (v1)",
}


def load(results: Path) -> dict:
    """Benchmark results from the per-step files (each written once its step finishes)."""
    read = lambda name: json.loads((results / name).read_text()) if (results / name).exists() else {}
    arms = {}
    for metrics in sorted((results / "runs").glob("*/metrics.json")):
        arms[metrics.parent.name] = json.loads(metrics.read_text())
    return {
        "arms": arms,
        "baselines": read("baselines.json"),
        "transfer": read("transfer.json"),
        "length_leakage": read("length_leakage.json"),
        "shortcut": read("shortcut.json"),
    }


def overview(out: Path, atlas_xyz: np.ndarray, atlas_colour: np.ndarray) -> None:
    """Fig. 1: the dual encoder, the caption, the multi-positive objective and the resulting 3D atlas."""
    fig = plt.figure(figsize=(7.1, 2.75))
    ax = fig.add_axes([0, 0, 0.62, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    def box(x, y, w, h, text, colour, size=6.5, weight="bold", text_colour="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=colour, ec="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=text_colour, weight=weight)

    def arrow(a, b, colour=viz.MUTED):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=colour, lw=0.9))

    seq = "ATGGCGTTCGAATCCGCA"
    for i, base in enumerate(seq):
        ax.text(
            0.15 + i * 0.13,
            3.55,
            base,
            fontsize=6.2,
            family="DejaVu Sans Mono",
            weight="bold",
            color={"A": "#2f9e44", "C": "#3b5bdb", "G": "#f08c00", "T": "#e03131"}[base],
        )
    ax.text(0.15, 3.95, "500 bp window (fixed length)", fontsize=6, color=viz.MUTED)
    ax.text(0.15, 3.18, "+ reverse complement", fontsize=5.5, color=viz.MUTED, style="italic")
    box(0.1, 2.25, 2.5, 0.7, "DNA encoder\ndilated CNN,\nboth strands", "#1f2a5c", 5.6)
    arrow((1.35, 3.1), (1.35, 2.98))
    ax.text(
        0.15,
        1.35,
        "“homo sapiens human primate\nmammal … small nuclear\nrna snrna spliceosomal”",
        fontsize=5.6,
        color=viz.INK,
        style="italic",
        va="center",
    )
    ax.text(1.55, 0.62, "caption = lineage\n+ biotype words", fontsize=5.0, color=viz.MUTED)
    box(0.1, 0.0, 2.5, 0.5, "text encoder\n(bag of words)", "#5f3dc4", 5.6)
    arrow((1.35, 0.95), (1.35, 0.52))
    # similarity matrix with multi-positive targets
    rng = np.random.default_rng(3)
    owner = np.array([0, 0, 1, 2, 2, 2, 3, 1])
    sims = 0.15 * rng.random((8, 4))
    sims[np.arange(8), owner] += 0.8
    mx, my, cell = 3.6, 0.55, 0.36
    for i in range(8):
        for j in range(4):
            colour = viz.SEQUENTIAL(sims[i, j])
            ax.add_patch(plt.Rectangle((mx + j * cell, my + (7 - i) * cell), cell * 0.94, cell * 0.94, fc=colour, ec="none"))
            if owner[i] == j:
                ax.add_patch(
                    plt.Rectangle((mx + j * cell, my + (7 - i) * cell), cell * 0.94, cell * 0.94, fill=False, ec=viz.INK, lw=0.8)
                )
    ax.text(mx + 2 * cell, my + 8 * cell + 0.12, "distinct captions in the batch", ha="center", fontsize=5.8, color=viz.MUTED)
    ax.text(mx - 0.15, my + 4 * cell, "sequences", rotation=90, ha="right", va="center", fontsize=5.8, color=viz.MUTED)
    ax.text(
        mx + 2 * cell,
        my - 0.28,
        "multi-positive InfoNCE:\nsame caption is never a negative",
        ha="center",
        va="top",
        fontsize=5.6,
        color=viz.INK,
    )
    arrow((2.65, 2.6), (mx - 0.05, my + 6 * cell))
    arrow((2.65, 0.25), (mx - 0.05, my + 1.5 * cell))
    # zero-shot prompt
    box(
        6.0,
        2.85,
        3.9,
        0.95,
        "zero-shot: any caption\nis a classifier",
        "#ff5a36",
        6,
    )
    box(
        6.0,
        1.55,
        3.9,
        0.95,
        "search: text → DNA\nONNX + in-browser reader",
        "#e9ecef",
        6,
        text_colour=viz.INK,
    )
    box(
        6.0,
        0.25,
        3.9,
        0.95,
        "leakage control: fixed\nwindows, held-out chromosomes",
        "#e9ecef",
        6,
        text_colour=viz.INK,
    )
    arrow((mx + 4 * cell + 0.1, my + 4 * cell), (5.95, 2.0))
    ax3 = fig.add_axes([0.6, 0.02, 0.4, 0.96], projection="3d")
    viz.scatter_3d(ax3, atlas_xyz, atlas_colour, palette=SPECIES_COLOUR, size=1.6, dark=True)
    ax3.text2D(0.04, 0.93, "held-out windows, 3D UMAP", transform=ax3.transAxes, fontsize=6, color="#dfe3ff")
    for k, (name, colour) in enumerate(SPECIES_COLOUR.items()):
        ax3.text2D(0.04, 0.05 + 0.055 * k, SHORT[name], transform=ax3.transAxes, fontsize=5.6, color=colour, style="italic")
    viz.save(fig, out / "fig1_overview")


def pitfalls(bench: dict, collisions: dict, out: Path) -> None:
    """Fig. 2: length names the biotype of whole features; CLIP batches are full of identical captions."""
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), layout="constrained", gridspec_kw={"width_ratios": [1.25, 1]})
    leak = bench["length_leakage"]
    names = list(leak)
    short = {
        n: n.split(" (")[0]
        .replace("Actinidia chinensis", "kiwifruit")
        .replace("A. thaliana chloroplast", "chloroplast")
        .replace("Bacteroides fragilis", "B. fragilis")
        for n in names
    }
    y = np.arange(len(names))
    axes[0].barh(y, [leak[n]["length_only_macro_f1"] for n in names], color=viz.ACCENT, height=0.6, label="length only")
    axes[0].barh(y, [leak[n]["chance_macro_f1"] for n in names], color="#adb5bd", height=0.6, label="shuffled labels")
    for i, n in enumerate(names):
        axes[0].text(
            leak[n]["length_only_macro_f1"] + 0.02, i, f"{leak[n]['length_only_macro_f1']:.2f}", va="center", fontsize=5.5
        )
    axes[0].set_yticks(y, [short[n] for n in names], fontsize=6)
    axes[0].set_xlim(0, 1.1)
    axes[0].set_xlabel("biotype macro F1, whole features", fontsize=6)
    axes[0].set_title("a  Length gives the label away", fontsize=7)
    axes[0].legend(fontsize=5, loc="upper center", bbox_to_anchor=(0.45, -0.32), ncols=2)
    axes[0].invert_yaxis()
    ax = axes[1]
    ax.bar(
        [0, 1],
        [collisions["rows_with_duplicate"] * 100, 100 * collisions["distinct_captions"] / collisions["batch"]],
        color=[viz.ACCENT, viz.PALETTE[0]],
        width=0.6,
    )
    ax.set_xticks([0, 1], ["rows with an\nidentical caption", "distinct\ncaptions"], fontsize=5.5)
    for x, v in enumerate([collisions["rows_with_duplicate"] * 100, 100 * collisions["distinct_captions"] / collisions["batch"]]):
        ax.text(x, v + 2, f"{v:.0f}%", ha="center", fontsize=5.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of a 256-row batch", fontsize=6)
    ax.set_title("b  CLIP batches", fontsize=7)
    viz.save(fig, out / "fig2_pitfalls")


def benchmark(bench: dict, out: Path) -> None:
    """Fig. 3: test macro F1 of every arm (prompts + k-NN, and prompts alone) against k-mer baselines at both budgets."""
    arms, base = bench["arms"], bench["baselines"]
    rows, errors, baselines = {}, {}, []

    def add_arm(a):
        t = arms[a]["test"]
        rows[ARM_LABELS[a]] = {
            "species": t["ensemble"]["species_macro_f1"],
            "biotype": t["ensemble"]["biotype_macro_f1"],
            "cell": t["ensemble"]["cell_macro_f1"],
            "zs": t["zero_shot"]["cell_macro_f1"],
        }
        errors[ARM_LABELS[a]] = {"cell": tuple(t["ensemble"]["cell_macro_f1_ci"])}

    def add_baseline(name, tag):
        b = base[f"{name}/{tag}"]
        label = f"{name} ({'full data' if tag == 'main' else '300 per cell'})"
        rows[label] = {"species": b["species_macro_f1"], "biotype": b["biotype_macro_f1"], "cell": b["cell_macro_f1"]}
        if "cell_macro_f1_ci" in b:
            errors[label] = {"cell": tuple(b["cell_macro_f1_ci"])}
        baselines.append(label)

    add_arm("main")
    add_baseline("4-mer logistic", "main")
    add_baseline("6-mer logistic", "main")
    for a in ARM_LABELS:
        if a != "main" and a in arms:
            add_arm(a)
    for name in ("4-mer logistic", "6-mer logistic", "4-mer kNN", "majority"):
        add_baseline(name, "ablation")
    viz.metric_bars(
        rows,
        out / "fig3_benchmark",
        (("species", "Species"), ("biotype", "Biotype | species"), ("cell", "Species × biotype"), ("zs", "Cell, prompts only")),
        highlight=[ARM_LABELS["main"]],
        secondary=baselines,
        errors=errors,
        width=7.1,
    )


def atlas_figure(pipe: Pipeline, records, out: Path, n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fig. 4: DNA and prompt embeddings in one 3D UMAP, coloured by species and by biotype family."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(records), min(n, len(records)), replace=False)
    chosen = [records[i] for i in pick]
    dna = pipe.embed([r.sequence for r in chosen])
    text = pipe.prototypes
    xyz = viz.project(np.vstack([dna, text]), 3, seed=seed)
    d, t = xyz[: len(dna)], xyz[len(dna) :]
    species = np.array([r.species for r in chosen])
    family = np.array([FAMILY.get(r.biotype, "regulatory") for r in chosen])
    fig = plt.figure(figsize=(7.1, 3.2), facecolor="#0d1020")
    for k, (labels, palette, title) in enumerate(
        ((species, SPECIES_COLOUR, "a  coloured by species"), (family, FAMILY_COLOUR, "b  coloured by biotype family"))
    ):
        ax = fig.add_axes([0.5 * k, 0.0, 0.5, 1.0], projection="3d")
        viz.scatter_3d(ax, d, labels, palette=palette, size=1.5, dark=True)
        c = np.clip((t - np.median(d, 0)) / np.percentile(np.abs(d - np.median(d, 0)), 98), -1.25, 1.25)
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], marker="*", s=22, c="white", edgecolors="#0d1020", linewidths=0.3, depthshade=False)
        ax.text2D(0.02, 0.97, title, transform=ax.transAxes, fontsize=7, color="white", weight="bold")
        for j, (name, colour) in enumerate(palette.items()):
            ax.text2D(0.02, 0.04 + 0.05 * j, SHORT.get(name, name), transform=ax.transAxes, fontsize=5.4, color=colour)
    fig.text(0.97, 0.02, "★ prompt embeddings (one per species × biotype)", ha="right", fontsize=5, color="#dfe3ff")
    viz.save(fig, out / "fig4_atlas")
    return d, species


def transfer_figure(bench: dict, out: Path) -> None:
    """Fig. 5: genomes never seen in training: which training species their windows resemble, and closed-set biotype
    calls from composed prompts, species-agnostic pooled prompts and 4-mer logistic regression."""
    tr = bench["transfer"]
    genomes = [g for g in ("kiwifruit", "bacteroides", "chloroplast") if g in tr]
    labels = {"kiwifruit": "kiwifruit", "bacteroides": "B. fragilis", "chloroplast": "A. thaliana\nchloroplast"}
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.1), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    for x, g in enumerate(genomes):
        called = tr[g]["called_species"]
        total, bottom = sum(called.values()), 0.0
        for species in SPECIES_ORDER:
            share = 100 * called.get(species, 0) / total
            ax.bar(x, share, bottom=bottom, color=SPECIES_COLOUR[species], width=0.7)
            bottom += share
        ax.text(x, 102, f"GC {100 * tr[g]['gc']:.0f}%", ha="center", fontsize=5.2, color=viz.MUTED)
    ax.set_xticks(range(len(genomes)), [labels[g] for g in genomes], fontsize=5)
    ax.set_ylim(0, 110)
    for species in SPECIES_ORDER:
        ax.bar(0, 0, color=SPECIES_COLOUR[species], label=SHORT[species])
    ax.legend(fontsize=4.3, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.22), handlelength=0.8, columnspacing=0.5)
    ax.set_ylabel("% of windows, nearest training species", fontsize=5.5)
    ax.set_title("a  Reads like", fontsize=7)
    ax = axes[1]
    x = np.arange(len(genomes))
    w = 0.27
    for k, (key, name, colour) in enumerate(
        (
            ("closed_biotype_accuracy_known", "composed prompts", viz.ACCENT),
            ("pooled_biotype_accuracy_known", "pooled prompts", "#ffa94d"),
            ("kmer_closed_biotype_accuracy", "4-mer logistic", viz.PALETTE[0]),
        )
    ):
        ax.bar(x + (k - 1) * w, [tr[g][key] for g in genomes], w, color=colour, label=name)
    for i, g in enumerate(genomes):
        ax.plot([i - 1.5 * w, i + 1.5 * w], [tr[g]["closed_chance"]] * 2, color=viz.INK, ls=":", lw=0.8)
    ax.set_xticks(x, [labels[g] for g in genomes], fontsize=5.5)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("closed-set biotype accuracy", fontsize=5.5)
    ax.set_title("b  Naming biotypes", fontsize=7)
    ax.legend(fontsize=4.8, loc="upper right")
    viz.save(fig, out / "fig5_transfer")


def length_figure(bench: dict, out: Path) -> None:
    """Fig. 7: (a) zero-shot cell F1 against input length, with the cross-validated R^2 of log length (100-500 bp)
    read off the pooled features of shuffled windows; (b) the whole-feature shortcut test: trained on whole kiwifruit
    CDS and introns, tested on whole features and on their equal-length centres."""
    arms, sc = bench["arms"], bench["shortcut"]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.85), layout="constrained", gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    for arm, name, colour, style in (
        ("main", "CLOP 2, full data", viz.ACCENT, "-"),
        ("cnn", "CLOP 2, 300/cell", "#ffa94d", "-"),
        ("cnn-meanmax", "mean+max, 300/cell", viz.PALETTE[0], "--"),
    ):
        if arm not in arms:
            continue
        length = arms[arm]["length"]
        sizes = sorted(length["by_length"], key=int)
        ax.plot(
            [int(x) for x in sizes],
            [length["by_length"][x]["cell_macro_f1"] for x in sizes],
            style,
            marker="o",
            ms=2.5,
            lw=1.1,
            color=colour,
            label=f"{name}  $R^2_{{pool}}$ {max(length['pooled_probe_r2']['100-500'], 0):.2f}",
        )
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xticks([50, 100, 200, 500], ["50", "100", "200", "500"], fontsize=5.5)
    ax.tick_params(labelsize=5.5)
    ax.set_xlabel("input length (bp)", fontsize=5.5)
    ax.set_ylabel("cell macro F1, prompts", fontsize=5.5)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    ax.legend(fontsize=4.3, loc="upper left", handlelength=1.6)
    ax.set_title("a  Accuracy against length", fontsize=7)
    ax = axes[1]
    methods = [m for m in ("length only", "4-mer logistic", "mean+max pooling", "moment pooling (CLOP 2)") if m in sc]
    labels = {
        "length only": "length\nonly",
        "4-mer logistic": "4-mer",
        "mean+max pooling": "max\npooling",
        "moment pooling (CLOP 2)": "CLOP 2",
    }
    x, w = np.arange(len(methods)), 0.38
    for k, (key, name, colour) in enumerate(
        (("whole", "whole features", viz.PALETTE[0]), ("equal_length", "100 bp centres", viz.ACCENT))
    ):
        ax.bar(
            x + (k - 0.5) * w,
            [sc[m][key] for m in methods],
            w,
            yerr=[sc[m].get(f"{key}_sd", 0) for m in methods],
            color=colour,
            label=name,
            error_kw={"lw": 0.6, "capsize": 1.2},
        )
    ax.set_xticks(x, [labels[m] for m in methods], fontsize=5.2)
    ax.set_ylim(0.0, 1.22)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=5.5)
    ax.set_ylabel("biotype macro F1", fontsize=5.5)
    ax.legend(fontsize=4.6, loc="upper center", ncols=2, handlelength=1.0, columnspacing=0.8)
    ax.set_title("b  Trained on whole features", fontsize=7)
    viz.save(fig, out / "fig7_length")


def prompts_figure(pipe: Pipeline, out: Path) -> None:
    """Cosine similarity between the 44 label prompts, ordered by average-linkage clustering."""
    names = [f"{SHORT[n.split('|')[0]]} · {n.split('|')[1].replace('_', ' ')}" for n in pipe.label_names]
    viz.similarity_heatmap(pipe.prototypes, names, out / "fig6_prompts", "")


def poster_atlas(xyz: np.ndarray, species: np.ndarray, out: Path) -> None:
    """Poster hero: the held-out atlas in 3D on a night background, large and legend-free."""
    fig = plt.figure(figsize=(9.0, 6.4), facecolor="#0d1020")
    ax = fig.add_axes([-0.08, -0.12, 1.16, 1.24], projection="3d")
    viz.scatter_3d(ax, xyz, species, palette=SPECIES_COLOUR, size=4.5, elev=18, azim=-62, dark=True)
    fig.savefig(out / "fig_poster_atlas.png", dpi=300, facecolor="#0d1020")
    plt.close(fig)


def curves(results: Path, out: Path) -> None:
    history = json.loads((results / "runs" / "main" / "history.json").read_text())
    viz.training_curves(history, out / "figS_training")


def confusion(results: Path, out: Path) -> None:
    data = np.load(results / "runs" / "main" / "embeddings.npz")
    cells, pred = data["cells"], data["predictions"]
    biotype = [c.split("|")[1] for c in cells]
    viz.confusion(biotype, [p.split("|")[1] for p in pred], out / "figS_confusion_biotype", "Biotype calls, test (%)")
    viz.confusion(
        [SHORT[c.split("|")[0]] for c in cells],
        [SHORT[p.split("|")[0]] for p in pred],
        out / "figS_confusion_species",
        "Species calls, test (%)",
    )


def collision_stats(atlas: Path, cap: int = 1200, batch: int = 256, draws: int = 200, seed: int = 0) -> dict:
    from clop.data import cap_per_cell

    records = cap_per_cell([r for r in parse_fasta(atlas) if r.split == "train"], cap)
    _, ids = np.unique([r.cell for r in records], return_inverse=True)
    rng = np.random.default_rng(seed)
    pairs, rows, distinct = [], [], []
    for _ in range(draws):
        b = ids[rng.choice(len(ids), batch, replace=False)]
        same = b[:, None] == b[None, :]
        np.fill_diagonal(same, False)
        pairs.append(same.sum(1).mean())
        rows.append(same.any(1).mean())
        distinct.append(len(set(b.tolist())))
    return {
        "batch": batch,
        "false_negatives_per_row": float(np.mean(pairs)),
        "rows_with_duplicate": float(np.mean(rows)),
        "distinct_captions": float(np.mean(distinct)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("out")
    parser.add_argument("--atlas", required=True)
    args = parser.parse_args()
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bench = load(results)
    collisions = collision_stats(Path(args.atlas))
    (results / "collisions.json").write_text(json.dumps(collisions, indent=1))
    pipe = Pipeline.load(results / "runs" / "main" / "model.pt")
    test = [r for r in parse_fasta(args.atlas) if r.split == "test" and r.cell in pipe.labels]
    xyz, species = atlas_figure(pipe, test, out)
    overview(out, xyz, species)
    prompts_figure(pipe, out)
    poster_atlas(xyz, species, out)
    pitfalls(bench, collisions, out)
    benchmark(bench, out)
    transfer_figure(bench, out)
    length_figure(bench, out)
    curves(results, out)
    confusion(results, out)
    print(json.dumps(collisions), Counter(species).most_common(2))


if __name__ == "__main__":
    main()
