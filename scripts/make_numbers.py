#!/usr/bin/env python3
"""LaTeX macros (\\newcommand) for every number quoted in the paper and poster, from a benchmark results directory.

python scripts/make_numbers.py results paper/numbers.tex
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ARMS = {
    "main": "Main",
    "cnn": "Cnn",
    "cnn-clip-loss": "CnnClip",
    "cnn-sigmoid-loss": "CnnSig",
    "cnn-no-rc": "CnnNoRc",
    "cnn-no-crop": "CnnNoCrop",
    "cnn-meanmax": "CnnMeanmax",
    "fcgr": "Fcgr",
    "transformer": "Tf",
    "lstm": "Lstm",
    "lstm-clip-loss": "LstmClip",
}
BASELINES = {"6-mer logistic": "KSix", "4-mer logistic": "KFour", "4-mer kNN": "KnnFour", "majority": "Major"}
LENGTHS = {"50": "Fifty", "100": "Hundred", "200": "TwoHundred", "500": "FiveHundred"}
SHORTCUT = {
    "length only": "Len",
    "4-mer logistic": "Kmer",
    "mean+max pooling": "Old",
    "moment pooling (CLOP 2)": "New",
}
GENOMES = {"kiwifruit": "Kiwi", "bacteroides": "Bfrag", "chloroplast": "Chloro"}


def f3(x: float) -> str:
    return f"{x:.3f}"


def pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}"


def hierarchical(prefix: str, m: dict, out: dict) -> None:
    out[f"{prefix}Species"] = f3(m["species_macro_f1"])
    out[f"{prefix}Biotype"] = f3(m["biotype_macro_f1"])
    out[f"{prefix}Cell"] = f3(m["cell_macro_f1"])
    out[f"{prefix}SpeciesAcc"] = pct(m["species_accuracy"], 1)
    out[f"{prefix}CellAcc"] = pct(m["cell_accuracy"], 1)
    for key, name in (("cell_macro_f1_ci", "Cell"), ("species_macro_f1_ci", "Species")):
        if key in m:
            out[f"{prefix}{name}Lo"], out[f"{prefix}{name}Hi"] = f3(m[key][0]), f3(m[key][1])
    for species, value in m.get("biotype_macro_f1_by_species", {}).items():
        out[f"{prefix}Biotype{species.split()[0]}"] = f3(value)


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


def main(results: str, target: str) -> None:
    root = Path(results)
    bench = load(root)
    v: dict[str, str] = {}
    for arm, tag in ARMS.items():
        if arm not in bench.get("arms", {}):
            continue
        m = bench["arms"][arm]
        t = m["test"]
        for method, name in (("zero_shot", "Zs"), ("knn", "Knn"), ("ensemble", "Ens")):
            hierarchical(f"{tag}{name}", t[method], v)
        r, g = t["retrieval"], t["geometry"]
        v |= {
            f"{tag}Map": f3(r["text_to_dna_map"]),
            f"{tag}PTen": f3(r["text_to_dna_p@10"]),
            f"{tag}PHundred": f3(r["text_to_dna_p@100"]),
            f"{tag}Mrr": f3(r["dna_to_text_mrr"]),
            f"{tag}TopOne": pct(r["dna_to_text_top1"], 1),
            f"{tag}Gap": f"{g['modality_gap']:.2f}",
            f"{tag}Sil": f3(g["silhouette"]),
            f"{tag}Ari": f3(g["kmeans_ari"]),
            f"{tag}OwnCos": f3(g["mean_cosine_to_own_prompt"]),
            f"{tag}Params": f"{m['parameters']['total']:,}".replace(",", "{,}"),
            f"{tag}DnaParams": f"{m['parameters']['dna']:,}".replace(",", "{,}"),
            f"{tag}Epochs": str(m["epochs_run"]),
            f"{tag}Best": str(m["best_epoch"]),
            f"{tag}Minutes": f"{m.get('wall_seconds', 0) / 60:.0f}",
        }
        if "length" in m:
            for key, name in (("length_probe_r2", "LenProbe"), ("pooled_probe_r2", "PoolProbe")):
                for span, r2 in m["length"].get(key, {}).items():
                    v[f"{tag}{name}{LENGTHS[span.split('-')[0]]}"] = f"{max(r2, 0.0):.2f}"
            for size, name in LENGTHS.items():
                v[f"{tag}Len{name}"] = f3(m["length"]["by_length"][size]["cell_macro_f1"])
                v[f"{tag}Bias{name}"] = f3(m["length"]["by_length"][size]["label_bias"])
        v[f"{tag}Train"] = f"{m['data']['train']['n']:,}".replace(",", "{,}")
        v[f"{tag}Test"] = f"{t['zero_shot']['n']:,}".replace(",", "{,}")
    for (name, scope), m in ((k.split("/"), x) for k, x in bench.get("baselines", {}).items()):
        hierarchical(f"{BASELINES[name]}{'Full' if scope == 'main' else 'Small'}", m, v)
    for genome, tag in GENOMES.items():
        g = bench.get("transfer", {}).get(genome)
        if not g:
            continue
        called = g["called_species"]
        total = sum(called.values())
        v |= {
            f"Tr{tag}N": str(g["n"]),
            f"Tr{tag}Gc": pct(g["gc"]),
            f"Tr{tag}Open": pct(g["open_biotype_accuracy"], 1),
            f"Tr{tag}KmerOpen": pct(g["kmer_open_biotype_accuracy"], 1),
            f"Tr{tag}Composed": pct(g["closed_biotype_accuracy_known"], 1),
            f"Tr{tag}Pooled": pct(g["pooled_biotype_accuracy_known"], 1),
            f"Tr{tag}Kmer": pct(g["kmer_closed_biotype_accuracy"], 1),
            f"Tr{tag}Chance": pct(g["closed_chance"]),
            f"Tr{tag}K": str(round(1 / g["closed_chance"])),
        }
        for species, count in called.items():
            v[f"Tr{tag}Called{species.split()[0]}"] = pct(count / total, 1)
        for biotype, row in g["pooled_confusion"].items():
            n = sum(row.values())
            v[f"Tr{tag}{biotype.replace('_', '')}PooledRight"] = pct(row.get(biotype, 0) / n, 0)
            v[f"Tr{tag}{biotype.replace('_', '')}N"] = str(n)
    for name, m in bench.get("length_leakage", {}).items():
        tag = {"Actinidia": "Kiwi", "Bacteroides": "Bfrag", "A.": "Chloro"}[name.split()[0]]
        v[f"Leak{tag}"], v[f"Leak{tag}Chance"] = f3(m["length_only_macro_f1"]), f3(m["chance_macro_f1"])
        v[f"Leak{tag}N"] = f"{m['n']:,}".replace(",", "{,}")
    shortcut = bench.get("shortcut", {})
    for name, tag in SHORTCUT.items():
        if name in shortcut:
            m = shortcut[name]
            v |= {
                f"Sc{tag}Whole": f3(m["whole"]),
                f"Sc{tag}Equal": f3(m["equal_length"]),
                f"Sc{tag}Drop": f3(m["whole"] - m["equal_length"]),
                f"Sc{tag}Bias": f3(m["equal_length_bias"]),
            }
    if shortcut:
        v |= {f"ScN{k.capitalize()}": f"{n:,}".replace(",", "{,}") for k, n in shortcut["n"].items()}
        v |= {f"ScMedian{b}": f"{x:.0f}" for b, x in shortcut["median_length"].items()}
    speed = json.loads((root / "speed.json").read_text()) if (root / "speed.json").exists() else {}
    for arm, tag in ARMS.items():
        if arm in speed:
            v[f"{tag}Speed"] = f"{speed[arm]['windows_per_second']:,.0f}".replace(",", "{,}")
    collisions = json.loads((root / "collisions.json").read_text()) if (root / "collisions.json").exists() else None
    if collisions:
        v |= {
            "CollideRows": pct(collisions["rows_with_duplicate"], 1),
            "CollideDistinct": f"{collisions['distinct_captions']:.0f}",
            "CollideFalseNeg": f"{collisions['false_negatives_per_row']:.1f}",
            "CollideBatch": str(collisions["batch"]),
        }
    if "main" in bench["arms"]:
        from clop.models import CLOP, ModelConfig

        margin = CLOP(ModelConfig.from_dict(bench["arms"]["main"]["model_config"])).dna.margin
        v["Margin"], v["MarginBp"] = str(margin), str(2 * margin)
    if "MainEnsCell" in v and "KFourFullCell" in v:
        v["MainGainCell"] = f3(float(v["MainEnsCell"]) - float(v["KFourFullCell"]))
        v["MainGainSpecies"] = f3(float(v["MainEnsSpecies"]) - float(v["KFourFullSpecies"]))
        v["MainGainBiotype"] = f3(float(v["MainEnsBiotype"]) - float(v["KFourFullBiotype"]))
    if "CnnEnsCell" in v and "KFourSmallCell" in v:
        v["CnnGainCell"] = f3(float(v["CnnEnsCell"]) - float(v["KFourSmallCell"]))
    lines = ["% generated by scripts/make_numbers.py; do not edit"]
    for key in sorted(v):
        name = re.sub(r"[^A-Za-z]", "", key)
        lines.append(f"\\newcommand{{\\n{name}}}{{{v[key]}}}")
    Path(target).write_text("\n".join(lines) + "\n")
    print(f"{len(v)} macros -> {target}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
