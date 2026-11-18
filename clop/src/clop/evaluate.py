"""Evaluation: zero-shot and k-NN classification of species, biotype within species and species x biotype cell;
cross-modal retrieval; embedding geometry (clustering, modality gap); bootstrap intervals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score, silhouette_score

from .baselines import knn_predict
from .io import Record
from .pipeline import Pipeline


def macro_f1(truth: Sequence[str], pred: Sequence[str]) -> float:
    return float(f1_score(truth, pred, average="macro", zero_division=0))


def bootstrap_f1(truth: Sequence[str], pred: Sequence[str], n: int = 500, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of macro F1 over window resamples (confusion counts, vectorised per resample)."""
    labels, t = np.unique(np.concatenate([truth, pred]), return_inverse=True)
    t, p = t[: len(truth)], t[len(truth) :]
    k = len(labels)
    rng = np.random.default_rng(seed)
    present = np.bincount(t, minlength=k) > 0
    scores = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(t), len(t))
        tt, pp = t[idx], p[idx]
        tp = np.bincount(tt[tt == pp], minlength=k)
        f1 = 2 * tp / np.maximum(np.bincount(tt, minlength=k) + np.bincount(pp, minlength=k), 1)
        scores[i] = f1[present].mean()
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def hierarchical_metrics(truth_cells: Sequence[str], scores: np.ndarray, label_names: Sequence[str], ci: bool = False) -> dict:
    """Species, biotype-given-species and cell metrics from (n, n_labels) label scores over ``species|biotype``
    labels. Biotype is decided among the true species' labels only, so it measures class recognition alone."""
    names = np.asarray(label_names)
    species_of = np.array([n.split("|")[0] for n in names])
    truth_cells = np.asarray(truth_cells)
    truth_species = np.array([c.split("|")[0] for c in truth_cells])
    truth_biotype = np.array([c.split("|")[1] for c in truth_cells])
    cell_pred = names[scores.argmax(1)]
    sp_names, sp_index = np.unique(species_of, return_inverse=True)
    species_scores = np.zeros((len(scores), len(sp_names)))
    np.add.at(species_scores.T, sp_index, scores.T)
    species_pred = sp_names[species_scores.argmax(1)]
    restricted = np.where(species_of[None, :] == truth_species[:, None], scores, -np.inf)
    biotype_pred = np.array([n.split("|")[1] for n in names[restricted.argmax(1)]])
    within = {s: macro_f1(truth_biotype[truth_species == s], biotype_pred[truth_species == s]) for s in np.unique(truth_species)}
    out = {
        "n": int(len(truth_cells)),
        "species_accuracy": float((species_pred == truth_species).mean()),
        "species_macro_f1": macro_f1(truth_species, species_pred),
        "biotype_macro_f1_by_species": within,
        "biotype_macro_f1": float(np.mean(list(within.values()))),
        "cell_accuracy": float((cell_pred == truth_cells).mean()),
        "cell_macro_f1": macro_f1(truth_cells, cell_pred),
    }
    if ci:
        out["species_macro_f1_ci"] = bootstrap_f1(truth_species, species_pred)
        out["cell_macro_f1_ci"] = bootstrap_f1(truth_cells, cell_pred)
    out["predictions"] = {"cell": cell_pred.tolist(), "species": species_pred.tolist(), "biotype": biotype_pred.tolist()}
    return out


def align_scores(class_names: Sequence[str], votes: np.ndarray, label_names: Sequence[str]) -> np.ndarray:
    """Re-index (n, n_classes) scores onto ``label_names`` (absent classes score 0)."""
    position = {c: i for i, c in enumerate(class_names)}
    out = np.zeros((len(votes), len(label_names)), dtype=np.float32)
    for j, name in enumerate(label_names):
        if name in position:
            out[:, j] = votes[:, position[name]]
    return out


def retrieval_metrics(
    dna: np.ndarray, cells: Sequence[str], prototypes: np.ndarray, label_names: Sequence[str], ks: Sequence[int] = (1, 10, 100)
) -> dict:
    """Text -> DNA: rank all query windows for each label prompt; mean average precision and precision@k with every
    window of that label relevant. DNA -> text: mean reciprocal rank of the true label among all prompts."""
    cells = np.asarray(cells)
    names = np.asarray(label_names)
    sims = prototypes @ dna.T  # (labels, n)
    order = np.argsort(-sims, axis=1)
    relevant = cells[order] == names[:, None]
    hits = np.cumsum(relevant, axis=1)
    n_rel = relevant.sum(1)
    ranks = np.arange(1, relevant.shape[1] + 1)
    ap = np.where(n_rel > 0, (relevant * hits / ranks).sum(1) / np.maximum(n_rel, 1), np.nan)
    present = n_rel > 0
    out = {
        "text_to_dna_map": float(np.nanmean(ap)),
        **{f"text_to_dna_p@{k}": float((hits[present, min(k, hits.shape[1]) - 1] / min(k, hits.shape[1])).mean()) for k in ks},
    }
    true = np.array([np.flatnonzero(names == c)[0] if c in set(names) else -1 for c in cells])
    keep = true >= 0
    true_sims = sims.T[np.flatnonzero(keep), true[keep]]
    rank = (sims.T[keep] > true_sims[:, None]).sum(1) + 1
    out["dna_to_text_mrr"] = float((1.0 / rank).mean())
    out["dna_to_text_top1"] = float((rank == 1).mean())
    return out


def geometry_metrics(
    dna: np.ndarray, cells: Sequence[str], text: np.ndarray | None = None, sample: int = 4000, seed: int = 0
) -> dict:
    """Silhouette (cosine) and k-means ARI against cells on a sample; modality gap (distance between the DNA and
    text embedding centroids, Liang et al. 2022) when text embeddings are given."""
    rng = np.random.default_rng(seed)
    cells = np.asarray(cells)
    idx = rng.choice(len(dna), min(sample, len(dna)), replace=False)
    x, y = dna[idx], cells[idx]
    out = {}
    if len(np.unique(y)) > 1:
        out["silhouette"] = float(silhouette_score(x, y, metric="cosine"))
        clusters = KMeans(len(np.unique(y)), n_init=4, random_state=seed).fit_predict(x)
        out["kmeans_ari"] = float(adjusted_rand_score(y, clusters))
    if text is not None and len(text):
        out["modality_gap"] = float(np.linalg.norm(dna.mean(0) - text.mean(0)))
    return out


def evaluate(
    pipeline: Pipeline,
    records: Sequence[Record],
    reference: Sequence[Record] | None = None,
    reference_embeddings: np.ndarray | None = None,
    k: int = 15,
    ci: bool = False,
    embeddings: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    """All metrics for ``records`` (zero-shot from prompts; k-NN and their average when a reference set is given)."""
    cells = [r.cell for r in records]
    dna = pipeline.embed([r.sequence for r in records]) if embeddings is None else embeddings
    names = pipeline.label_names
    zero = pipeline.scores(dna)
    result = {
        "zero_shot": hierarchical_metrics(cells, zero, names, ci),
        "retrieval": retrieval_metrics(dna, cells, pipeline.prototypes, names),
        "geometry": geometry_metrics(dna, cells, pipeline.prototypes),
    }
    own = {c: i for i, c in enumerate(names)}
    rows = [i for i, c in enumerate(cells) if c in own]
    if rows:
        result["geometry"]["mean_cosine_to_own_prompt"] = float(
            np.mean((dna[rows] * pipeline.prototypes[[own[cells[i]] for i in rows]]).sum(1))
        )
    if reference is not None:
        ref = pipeline.embed([r.sequence for r in reference]) if reference_embeddings is None else reference_embeddings
        classes, votes = knn_predict(ref, [r.cell for r in reference], dna, k)
        knn = align_scores(classes, votes, names)
        result["knn"] = hierarchical_metrics(cells, knn, names, ci)
        result["ensemble"] = hierarchical_metrics(cells, 0.5 * (knn + zero), names, ci)
    return result, dna


def label_bias(truth: Sequence[str], called: Sequence[str]) -> float:
    """Total-variation distance between the called and the true label frequencies (0: calls in the true proportions)."""
    labels, index = np.unique(np.concatenate([truth, called]), return_inverse=True)
    n = len(truth)
    return float(
        np.abs(np.bincount(index[:n], minlength=len(labels)) - np.bincount(index[n:], minlength=len(labels))).sum() / (2 * n)
    )


def length_metrics(
    pipeline: Pipeline,
    records: Sequence[Record],
    lengths: Sequence[int] = (50, 100, 200, 500),
    probe_ranges: Sequence[tuple[int, int]] = ((50, 500), (100, 500), (200, 500)),
    n: int = 3000,
    seed: int = 0,
) -> dict:
    """How much the embedding depends on input length, from random crops of ``n`` labelled windows.

    * ``by_length``: zero-shot metrics at each crop length, plus ``label_bias`` (a length-dependent model drifts towards
      the labels it ties to short or long inputs, beyond losing accuracy).
    * ``length_probe_r2``: for each length range, the cross-validated R^2 of a ridge regression of log crop length on
      the embedding, lengths drawn log-uniformly and independently of the label: 0 means the embedding does not encode
      length at all. Crops of real windows also differ in content with length (a short crop of a feature-centred window
      may miss the feature), and short inputs give noisier statistics, which the projection head and normalisation turn
      into a readable signal.
    * ``pooled_probe_r2``: the same regression on the encoder's pooled features (before the projection head) of
      composition-preserving shuffles of the windows, which are stationary: it isolates what the pooling itself senses
      of length. Moment pooling should give ~0; a max grows with length, and edge-affected positions weigh ~1/length."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import cross_val_predict

    rng = np.random.default_rng(seed)
    records = [records[i] for i in rng.choice(len(records), min(n, len(records)), replace=False)]
    cells = [r.cell for r in records]
    names = pipeline.label_names
    full = np.array([len(r.sequence) for r in records])

    def crops(sizes: np.ndarray) -> list[str]:
        starts = (rng.random(len(records)) * (full - sizes + 1)).astype(int)
        return [r.sequence[a : a + b] for r, a, b in zip(records, starts, sizes)]

    by_length = {}
    for size in lengths:
        scores = pipeline.scores(pipeline.embed(crops(np.minimum(size, full))))
        metrics = hierarchical_metrics(cells, scores, names)
        by_length[str(size)] = {
            "species_macro_f1": metrics["species_macro_f1"],
            "biotype_macro_f1": metrics["biotype_macro_f1"],
            "cell_macro_f1": metrics["cell_macro_f1"],
            "label_bias": label_bias(cells, np.asarray(names)[scores.argmax(1)]),
        }
    shuffled = [rng.permutation(np.frombuffer(r.sequence.encode(), np.uint8)).tobytes().decode() for r in records]
    probe = lambda x, y: float(r2_score(y, cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), x, y, cv=5)))
    probes, pooled = {}, {}
    for lo, hi in probe_ranges:
        sizes = np.minimum(np.exp(rng.uniform(np.log(lo), np.log(hi), len(records))).astype(int), full)
        probes[f"{lo}-{hi}"] = probe(pipeline.embed(crops(sizes)), np.log(sizes))
        stationary = [s[:n] for s, n in zip(shuffled, sizes)]
        pooled[f"{lo}-{hi}"] = probe(pipeline.embed(stationary, pooled=True), np.log(sizes))
    return {"by_length": by_length, "length_probe_r2": probes, "pooled_probe_r2": pooled, "n": len(records)}


def strip_predictions(result: dict) -> dict:
    """A copy without per-window prediction lists (for compact JSON summaries)."""
    if isinstance(result, dict):
        return {k: strip_predictions(v) for k, v in result.items() if k != "predictions"}
    return result
