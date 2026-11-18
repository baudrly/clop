"""Exports: ONNX encoders (any architecture) and a script bundle for the dependency-free in-browser reader in ``web/``
(CNN DNA encoder + bag-of-words text encoder, batch-norm folded to per-channel affine maps)."""

from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .io import Record
from .pipeline import Pipeline

logger = logging.getLogger(__name__)


class _DNA(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, codes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.model.encode_dna(codes, lengths)


class _Text(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(tokens, lengths)


def export_onnx(pipe: Pipeline, out_dir: str | Path, opset: int = 17, verify: bool = True) -> list[Path]:
    """``dna_encoder.onnx`` (codes uint8 [B, L], lengths int64 [B]) and ``text_encoder.onnx`` (tokens, lengths), both
    returning L2-normalised embeddings, plus ``onnx_meta.json`` (vocabulary, labels, prompt embeddings)."""
    out = Path(out_dir)
    model = pipe.model.cpu().eval()
    codes = torch.randint(0, 4, (2, 300), dtype=torch.uint8)
    lengths = torch.tensor([300, 250])
    tokens, token_lengths = (torch.from_numpy(a) for a in pipe.tokenizer.batch(["homo sapiens promoter", "trna"]))
    paths = [out / "dna_encoder.onnx", out / "text_encoder.onnx"]
    with torch.no_grad():
        torch.onnx.export(
            _DNA(model).eval(),
            (codes, lengths),
            paths[0],
            input_names=["codes", "lengths"],
            output_names=["embedding"],
            dynamic_axes={"codes": {0: "batch", 1: "length"}, "lengths": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=opset,
            dynamo=False,
        )
        torch.onnx.export(
            _Text(model).eval(),
            (tokens, token_lengths),
            paths[1],
            input_names=["tokens", "lengths"],
            output_names=["embedding"],
            dynamic_axes={"tokens": {0: "batch", 1: "length"}, "lengths": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=opset,
            dynamo=False,
        )
    meta = {
        "vocab": pipe.tokenizer.to_dict(),
        "labels": pipe.labels,
        "logit_scale": model.scale().item(),
        "prototypes": np.round(pipe.prototypes, 6).tolist(),
        "model_config": model.cfg.to_dict(),
        "codes": "A=0 C=1 G=2 T=3 other/padding=4",
    }
    (out / "onnx_meta.json").write_text(json.dumps(meta))
    model.eval()  # the exporter restores the wrappers' (training) mode recursively
    if verify:
        import onnxruntime as ort

        session = ort.InferenceSession(str(paths[0]))
        test_codes = torch.randint(0, 5, (3, 411), dtype=torch.uint8)
        test_lengths = torch.tensor([411, 300, 120])
        got = session.run(None, {"codes": test_codes.numpy(), "lengths": test_lengths.numpy()})[0]
        want = model.encode_dna(test_codes, test_lengths).detach().numpy()
        error = float(np.abs(got - want).max())
        logger.info("ONNX DNA encoder max abs error %.2e", error)
        if error > 1e-4:
            raise RuntimeError(f"ONNX export mismatch ({error:.2e})")
    return paths


def _b64(array) -> dict:
    a = np.ascontiguousarray(array.detach().cpu().numpy() if isinstance(array, torch.Tensor) else array, dtype=np.float32)
    return {"shape": list(a.shape), "data": base64.b64encode(a.tobytes()).decode()}


def _affine(bn: nn.BatchNorm1d) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return scale.numpy(), (bn.bias - bn.running_mean * scale).numpy()


def web_weights(pipe: Pipeline) -> dict:
    """Weights of a cnn + bow model in the layout ``web/js/clop.js`` reads."""
    model = pipe.model.cpu().eval()
    cfg = model.cfg
    if cfg.dna_encoder != "cnn" or cfg.text_encoder != "bow":
        raise ValueError("the web reader supports cnn DNA and bow text encoders; use --onnx for other models")
    dna = model.dna
    stem_conv, stem_bn = dna.stem[0], dna.stem[2]
    s, b = _affine(stem_bn)
    weights = {"stem.w": _b64(stem_conv.weight), "stem.b": _b64(stem_conv.bias), "stem.s": _b64(s), "stem.t": _b64(b)}
    for i, block in enumerate(dna.blocks):
        s, b = _affine(block.norm)
        weights |= {
            f"block{i}.w": _b64(block.conv.weight),
            f"block{i}.b": _b64(block.conv.bias),
            f"block{i}.s": _b64(s),
            f"block{i}.t": _b64(b),
        }
    for prefix, head in (("dna", dna.head), ("text", model.text.head)):
        weights |= {
            f"{prefix}.head0.w": _b64(head[0].weight),
            f"{prefix}.head0.b": _b64(head[0].bias),
            f"{prefix}.head3.w": _b64(head[3].weight),
            f"{prefix}.head3.b": _b64(head[3].bias),
        }
    weights["text.embed"] = _b64(model.text.embed.weight)
    return {
        "config": {
            "width": cfg.width,
            "depth": cfg.depth,
            "kernel": cfg.kernel,
            "embed_dim": cfg.embed_dim,
            "rc": cfg.rc,
            "pooling": cfg.pooling,
            "margin": dna.margin,
            "window": cfg.window,
        },
        "weights": weights,
    }


def _script(name: str, payload: dict) -> str:
    """``payload`` as a script that sets ``globalThis[name]``: unlike fetch(), a <script> tag also loads from file://."""
    text = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")  # never closes an inline <script>
    return f"globalThis.{name}={text};\n"


SITE = ("index.html", "style.css", "app.js", "clop.js", "atlas3d.js")


def export_web(
    pipe: Pipeline,
    out_dir: str | Path,
    atlas: Sequence[Record] | None = None,
    atlas_size: int = 6000,
    seed: int = 0,
    examples_per_cell: int = 3,
) -> list[Path]:
    """A complete static site that works from disk (double-click ``index.html``) or from any web server:

    * ``clop-model.js``: weights, vocabulary, label prompts and their embeddings;
    * ``clop-atlas.js`` when annotated sequences are given: 3D layout, labels and float16 embeddings of a sample (for
      neighbour search and the map), and up to ``examples_per_cell`` example sequences per species x biotype cell;
    * the reader (index.html, app.js, clop.js, atlas3d.js, style.css);
    * ``clop-reader.html``: the same reader with everything inlined, a single file to share."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = web_weights(pipe)
    from .text import BIOTYPE_WORDS, LINEAGE

    bundle |= {
        "format": "clop-web-2",
        "vocab": pipe.tokenizer.to_dict(),
        "logit_scale": pipe.model.scale().item(),
        "lineage": LINEAGE,
        "biotype_words": BIOTYPE_WORDS,
        "labels": list(pipe.labels),
        "captions": list(pipe.labels.values()),
        "prototypes": _b64(pipe.prototypes),
    }
    data = {"clop-model.js": _script("CLOP_MODEL", bundle)}
    if atlas:
        from collections import defaultdict

        from .viz import project

        rng = np.random.default_rng(seed)
        chosen = [atlas[i] for i in np.sort(rng.choice(len(atlas), min(atlas_size, len(atlas)), replace=False))]
        emb = pipe.embed([r.sequence for r in chosen])
        xyz = project(emb, 3)
        scores = pipe.scores(emb)
        examples = defaultdict(list)
        for i in rng.permutation(len(atlas)):
            cell = examples[atlas[i].cell]
            if len(cell) < examples_per_cell:
                cell.append({"id": atlas[i].id, "sequence": atlas[i].sequence})
        payload = {
            "format": "clop-atlas-2",
            "positions": np.round(xyz, 4).ravel().tolist(),
            "species": [r.species for r in chosen],
            "biotypes": [r.biotype for r in chosen],
            "ids": [r.id for r in chosen],
            "called": [pipe.label_names[j] for j in scores.argmax(1)],
            "embeddings": {"shape": list(emb.shape), "data": base64.b64encode(emb.astype(np.float16).tobytes()).decode()},
            "examples": dict(sorted(examples.items())),
        }
        data["clop-atlas.js"] = _script("CLOP_ATLAS", payload)
    from importlib import resources

    site = resources.files("clop").joinpath("assets/web")
    files = {name: site.joinpath(name).read_text() for name in SITE} | data
    paths = []
    for name, text in files.items():
        paths.append(out / name)
        paths[-1].write_text(text)
    # one pass over the page's own tags, so inlined code is never rescanned; tags of files not exported are dropped
    inline = lambda m: "<script>\n" + files[m[1]].replace("</script", "<\\/script") + "</script>" if m[1] in files else ""
    page = re.sub(r'<script src="([^"]+)"></script>', inline, files["index.html"])
    page = page.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{files['style.css']}</style>")
    paths.append(out / "clop-reader.html")
    paths[-1].write_text(page)
    logger.info("web reader in %s: %s", out, ", ".join(f"{p.name} ({p.stat().st_size / 1e6:.2f} MB)" for p in paths))
    return paths
