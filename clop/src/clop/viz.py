"""Figures: a consistent palette and typography, 2D and 3D embedding maps, training curves, confusion matrices and
benchmark bars. Everything renders headless (Agg)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

INK, MUTED, GRID, PAPER = "#14161f", "#5b5e6b", "#e6e4df", "#fcfcfa"
ACCENT = "#ff5a36"
PALETTE = (
    "#3b5bdb",
    "#ff5a36",
    "#12b886",
    "#fab005",
    "#be4bdb",
    "#15aabf",
    "#e64980",
    "#82c91e",
    "#7950f2",
    "#fd7e14",
    "#0ca678",
    "#495057",
    "#4dabf7",
    "#f783ac",
    "#a9e34b",
    "#9775fa",
    "#ffa94d",
    "#63e6be",
)
SEQUENTIAL = LinearSegmentedColormap.from_list("clop", ["#f7f6f2", "#ffd8c8", "#ff8a65", "#e8432a", "#7a1f0f"])

plt.rcParams.update(
    {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "font.size": 8,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def colours(labels: Sequence[str]) -> dict[str, str]:
    """Stable colour per label (sorted order through the palette, cycling)."""
    return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(sorted(set(labels)))}


def save(fig, path: str | Path, formats: Sequence[str] = ("png", "pdf")) -> list[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for f in formats:
        target = path.with_suffix(f".{f}")
        fig.savefig(target, dpi=220 if f == "png" else None, bbox_inches="tight")
        out.append(target)
    plt.close(fig)
    return out


def project(embeddings: np.ndarray, dims: int = 2, method: str = "auto", seed: int = 0) -> np.ndarray:
    """Low-dimensional coordinates: UMAP when installed (``auto``/``umap``), else t-SNE, or PCA."""
    x = np.asarray(embeddings, dtype=np.float32)
    if method in ("auto", "umap"):
        try:
            import inspect
            import warnings

            import umap

            options = dict(n_components=dims, n_neighbors=20, min_dist=0.15, metric="cosine", random_state=seed)
            if "compatibility_layout" in inspect.signature(umap.UMAP).parameters:
                options["compatibility_layout"] = True
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return umap.UMAP(**options).fit_transform(x)
        except ImportError:
            if method == "umap":
                raise
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=dims, perplexity=35, init="pca", metric="cosine", random_state=seed).fit_transform(x)
    from sklearn.decomposition import PCA

    return PCA(n_components=dims, random_state=seed).fit_transform(x)


def scatter_2d(
    ax,
    coords: np.ndarray,
    labels: Sequence[str],
    palette: dict[str, str] | None = None,
    size: float = 2.0,
    legend: bool = True,
    label_points: dict[str, tuple[float, float]] | None = None,
    ncol: int = 2,
) -> None:
    labels = np.asarray(labels)
    palette = palette or colours(labels)
    for name in sorted(set(labels), key=lambda n: -(labels == n).sum()):
        m = labels == name
        ax.scatter(coords[m, 0], coords[m, 1], s=size, color=palette[name], lw=0, alpha=0.8, label=name, rasterized=True)
    for name, (x, y) in (label_points or {}).items():
        ax.annotate(
            name,
            (x, y),
            fontsize=6,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc=PAPER, ec=GRID, lw=0.4, alpha=0.9),
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    if legend:
        ax.legend(
            fontsize=6,
            markerscale=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            ncols=ncol,
            handletextpad=0.2,
            columnspacing=0.8,
        )


def scatter_3d(
    ax,
    coords: np.ndarray,
    labels: Sequence[str],
    palette: dict[str, str] | None = None,
    size: float = 3.0,
    elev: float = 22,
    azim: float = -58,
    dark: bool = False,
) -> None:
    """Depth-cued 3D scatter: points fade and shrink with distance from the viewer, drawn back to front."""
    labels = np.asarray(labels)
    palette = palette or colours(labels)
    ax.view_init(elev=elev, azim=azim)
    c = coords - np.median(coords, 0)
    c = np.clip(c / np.percentile(np.abs(c), 98), -1.25, 1.25)  # robust scale: a few outlying islands cannot shrink the cloud
    e, a = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    depth = c @ view
    near = (depth - depth.min()) / max(np.ptp(depth), 1e-9)
    order = np.argsort(near)
    rgb = np.array([to_rgb(palette[name]) for name in labels])
    background = np.array(to_rgb("#0d1020" if dark else PAPER))
    fade = (0.35 + 0.65 * near)[:, None]
    rgba = np.concatenate([rgb * fade + background * (1 - fade), (0.35 + 0.6 * near)[:, None]], 1)
    ax.scatter(
        c[order, 0],
        c[order, 1],
        c[order, 2],
        s=size * (0.4 + 1.2 * near[order]),
        c=rgba[order],
        lw=0,
        depthshade=False,
        rasterized=True,
    )
    ax.set_axis_off()
    lim = 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    if dark:
        ax.set_facecolor("#0d1020")


def training_curves(history: list[dict], path: str | Path) -> list[Path]:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), layout="constrained")
    axes[0].plot(epochs, [h["train_loss"] for h in history], color=INK, lw=1.6, marker="o", ms=2.5)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    for key, label, colour in (
        ("val_species_f1", "species", PALETTE[0]),
        ("val_biotype_f1", "biotype within species", PALETTE[2]),
        ("val_cell_f1", "species x biotype", ACCENT),
    ):
        if key in history[0]:
            axes[1].plot(epochs, [h[key] for h in history], color=colour, lw=1.6, marker="o", ms=2.5, label=label)
    axes[1].set_title("Validation zero-shot macro F1")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=6.5)
    for ax in axes:
        ax.grid(axis="y")
    return save(fig, path)


def confusion(
    truth: Sequence[str],
    pred: Sequence[str],
    path: str | Path,
    title: str = "",
    order: Sequence[str] | None = None,
    annotate: bool = True,
) -> list[Path]:
    """Row-normalised confusion matrix (%)."""
    names = list(order or sorted(set(truth)))
    index = {n: i for i, n in enumerate(names)}
    m = np.zeros((len(names), len(names)))
    for t, p in zip(truth, pred):
        if t in index and p in index:
            m[index[t], index[p]] += 1
    m = 100 * m / np.maximum(m.sum(1, keepdims=True), 1)
    size = max(2.6, 0.22 * len(names) + 1.2)
    fig, ax = plt.subplots(figsize=(size, size), layout="constrained")
    ax.imshow(m, cmap=SEQUENTIAL, vmin=0, vmax=100)
    if annotate and len(names) <= 24:
        for (i, j), v in np.ndenumerate(m):
            if v >= 5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5, color="white" if v > 55 else INK)
    ax.set_xticks(range(len(names)), names, rotation=50, ha="right", fontsize=6, rotation_mode="anchor")
    ax.set_yticks(range(len(names)), names, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_title(title)
    ax.spines[:].set_visible(False)
    return save(fig, path)


def metric_bars(
    rows: dict[str, dict[str, float]],
    path: str | Path,
    metrics: Sequence[tuple[str, str]],
    highlight: Sequence[str] = (),
    secondary: Sequence[str] = (),
    errors: dict[str, dict[str, tuple[float, float]]] | None = None,
    width: float = 7.0,
) -> list[Path]:
    """Horizontal bars, one panel per metric; ``rows`` maps an arm name to its metric values."""
    arms = list(rows)
    fig, axes = plt.subplots(1, len(metrics), figsize=(width, 0.24 * len(arms) + 0.9), sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, (key, title) in zip(axes, metrics):
        values = [rows[a].get(key, np.nan) for a in arms]
        colour = [
            ACCENT if a in highlight else "#20c997" if a in secondary else "#adb5bd" if "majority" in a.lower() else PALETTE[0]
            for a in arms
        ]
        ax.barh(range(len(arms)), values, color=colour, height=0.68)
        for y, v in enumerate(values):
            if np.isfinite(v):
                lo_hi = (errors or {}).get(arms[y], {}).get(key)
                if lo_hi:
                    ax.plot(lo_hi, [y, y], color=INK, lw=0.8)
                ax.text((lo_hi[1] if lo_hi else v) + 0.015, y, f"{v:.2f}", va="center", fontsize=5.5, color=MUTED)
        ax.set_xlim(0, 1.1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(title, fontsize=8)
        ax.grid(axis="x")
    axes[0].set_yticks(range(len(arms)), arms, fontsize=6.5)
    axes[0].invert_yaxis()
    return save(fig, path)


def similarity_heatmap(prototypes: np.ndarray, labels: Sequence[str], path: str | Path, title: str = "") -> list[Path]:
    """Cosine similarity between label prompts, ordered by average-linkage clustering."""
    from scipy.cluster.hierarchy import leaves_list, linkage

    sims = prototypes @ prototypes.T
    order = leaves_list(linkage(prototypes, "average", metric="cosine")) if len(labels) > 2 else np.arange(len(labels))
    size = max(3.0, 0.14 * len(labels) + 1.6)
    fig, ax = plt.subplots(figsize=(size, size), layout="constrained")
    ax.imshow(sims[np.ix_(order, order)], cmap=SEQUENTIAL, vmin=np.percentile(sims, 5), vmax=1)
    names = [labels[i] for i in order]
    ax.set_yticks(range(len(names)), names, fontsize=5)
    ax.set_xticks([])
    ax.set_title(title)
    ax.spines[:].set_visible(False)
    return save(fig, path)
