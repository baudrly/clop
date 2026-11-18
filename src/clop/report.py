"""Run reports: figures plus one self-contained interactive HTML page (3D embedding atlas in WebGL, metrics, plots).
The page has no external dependencies, so it opens offline and can be attached or published as is."""

from __future__ import annotations

import base64
import html
import json
from importlib import resources
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb

from . import viz

ATLAS_POINTS = 5000


def _png(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def atlas_payload(
    embeddings: np.ndarray,
    cells: np.ndarray,
    predictions: np.ndarray | None,
    ids: np.ndarray | None,
    limit: int = ATLAS_POINTS,
    seed: int = 0,
) -> dict:
    """3D coordinates and colourings (species, biotype, call correctness) for a sample of embedded windows."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(embeddings), min(limit, len(embeddings)), replace=False))
    coords = viz.project(embeddings[idx].astype(np.float32), 3)
    cells = np.asarray(cells)[idx]
    species = np.array([c.split("|")[0] for c in cells])
    biotype = np.array([c.split("|")[1] for c in cells])
    modes = {}
    for name, labels in (("species", species), ("biotype", biotype)):
        palette = viz.colours(labels)
        modes[name] = {
            "legend": {k: v for k, v in palette.items()},
            "colors": np.round(np.array([to_rgb(palette[name]) for name in labels]), 3).ravel().tolist(),
        }
    if predictions is not None:
        correct = np.asarray(predictions)[idx] == cells
        palette = {"correct": "#12b886", "wrong": viz.ACCENT}
        modes["call"] = {
            "legend": palette,
            "colors": np.array([to_rgb(palette["correct" if c else "wrong"]) for c in correct]).ravel().round(3).tolist(),
        }
    return {
        "positions": np.round(coords, 4).ravel().tolist(),
        "modes": modes,
        "labels": [f"{s} · {b}" for s, b in zip(species, biotype)],
        "ids": (np.asarray(ids)[idx].tolist() if ids is not None else []),
        "predictions": (np.asarray(predictions)[idx].tolist() if predictions is not None else []),
    }


def write_figures(run: Path) -> dict[str, Path]:
    metrics = json.loads((run / "metrics.json").read_text())
    history = json.loads((run / "history.json").read_text())
    figures = run / "figures"
    out = {"curves": viz.training_curves(history, figures / "training_curves")[0]}
    data = np.load(run / "embeddings.npz", allow_pickle=False)
    cells, predictions = data["cells"], data["predictions"]
    out["confusion_cell"] = viz.confusion(cells, predictions, figures / "confusion_cells", "Species x biotype calls (test, %)")[0]
    species = [c.split("|")[0] for c in cells]
    out["confusion_species"] = viz.confusion(
        species, [p.split("|")[0] for p in predictions], figures / "confusion_species", "Species calls (test, %)"
    )[0]
    out["prompts"] = viz.similarity_heatmap(
        data["prototypes"], list(data["labels"]), figures / "prompt_similarity", "Prompt embeddings: cosine similarity"
    )[0]
    test = metrics.get("test", {})
    rows = {
        arm.replace("_", "-"): {
            "species": test[arm]["species_macro_f1"],
            "biotype": test[arm]["biotype_macro_f1"],
            "cell": test[arm]["cell_macro_f1"],
        }
        for arm in ("zero_shot", "knn", "ensemble")
        if arm in test
    }
    if rows:
        out["bars"] = viz.metric_bars(
            rows,
            figures / "test_metrics",
            (("species", "Species"), ("biotype", "Biotype | species"), ("cell", "Species x biotype")),
            highlight=list(rows),
        )[0]
    return out


def _card(label: str, value: str, note: str = "") -> str:
    return f'<div class="card"><div class="v">{html.escape(value)}</div><div class="l">{html.escape(label)}</div><div class="n">{html.escape(note)}</div></div>'


def write_report(run: str | Path) -> Path:
    run = Path(run)
    metrics = json.loads((run / "metrics.json").read_text())
    figures = write_figures(run)
    data = np.load(run / "embeddings.npz", allow_pickle=False)
    payload = atlas_payload(data["embeddings"].astype(np.float32), data["cells"], data["predictions"], data["ids"])
    test = metrics.get("test", {})
    cfg, params = metrics["model_config"], metrics["parameters"]
    cards = [
        _card("parameters", f"{params['total']:,}", f"{cfg['dna_encoder']} DNA + {cfg['text_encoder']} text"),
        _card("epochs", str(metrics["epochs_run"]), f"best {metrics['best_epoch']}"),
    ]
    for arm, title in (("zero_shot", "zero-shot"), ("ensemble", "zero-shot + k-NN")):
        if arm in test:
            cards.append(_card(f"{title} species F1", f"{test[arm]['species_macro_f1']:.3f}", "macro, test"))
            cards.append(_card(f"{title} cell F1", f"{test[arm]['cell_macro_f1']:.3f}", "species x biotype"))
    if "retrieval" in test:
        cards.append(_card("text → DNA mAP", f"{test['retrieval']['text_to_dna_map']:.3f}", "prompt retrieval"))
    if "geometry" in test and "modality_gap" in test["geometry"]:
        cards.append(_card("modality gap", f"{test['geometry']['modality_gap']:.2f}", "centroid distance"))
    table = ""
    for arm in ("zero_shot", "knn", "ensemble"):
        if arm in test:
            m = test[arm]
            table += (
                f"<tr><td>{arm.replace('_', '-')}</td><td>{m['species_accuracy']:.3f}</td><td>{m['species_macro_f1']:.3f}</td>"
                f"<td>{m['biotype_macro_f1']:.3f}</td><td>{m['cell_accuracy']:.3f}</td><td>{m['cell_macro_f1']:.3f}</td></tr>"
            )
    images = "".join(
        f'<figure><img src="{_png(p)}" alt="{k}"><figcaption>{k.replace("_", " ")}</figcaption></figure>'
        for k, p in figures.items()
    )
    script = resources.files("clop").joinpath("assets/web/atlas3d.js").read_text()
    page = TEMPLATE.format(
        title=html.escape(run.name),
        cards="".join(cards),
        table=table,
        images=images,
        payload=json.dumps(payload, separators=(",", ":")),
        script=script,
        n=len(payload["labels"]),
        config=html.escape(json.dumps(cfg, indent=1)),
    )
    target = run / "report.html"
    target.write_text(page)
    return target


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLOP report · {title}</title>
<style>
:root{{--ink:#14161f;--muted:#5b5e6b;--paper:#fcfcfa;--line:#e6e4df;--accent:#ff5a36;--night:#0d1020}}
*{{box-sizing:border-box}} body{{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Inter,sans-serif;color:var(--ink);background:var(--paper)}}
.hero{{position:relative;height:min(78vh,720px);background:radial-gradient(ellipse at 30% 20%,#1b2350,var(--night) 70%);overflow:hidden}}
.hero canvas{{position:absolute;inset:0;width:100%;height:100%;cursor:grab}}
.overlay{{position:absolute;left:28px;top:24px;color:#eef;pointer-events:none;max-width:520px}}
.overlay h1{{margin:0;font-size:30px;letter-spacing:-.02em}} .overlay p{{margin:.3em 0 0;color:#aab;font-size:14px}}
.controls{{position:absolute;left:28px;bottom:86px;display:flex;gap:8px;flex-wrap:wrap}}
.controls button{{background:#ffffff14;color:#eef;border:1px solid #ffffff30;border-radius:999px;padding:6px 14px;font:inherit;font-size:13px;cursor:pointer}}
.controls button.on{{background:var(--accent);border-color:var(--accent)}}
.legend{{position:absolute;right:20px;top:20px;bottom:20px;overflow:auto;color:#dde;font-size:12px;max-width:260px}}
.legend div{{display:flex;align-items:center;gap:7px;padding:2px 6px;border-radius:6px;cursor:pointer}} .legend div:hover{{background:#ffffff14}}
.legend i{{width:10px;height:10px;border-radius:50%;flex:none}}
.tip{{position:absolute;pointer-events:none;background:#000c;color:#fff;padding:5px 9px;border-radius:7px;font-size:12px;display:none;white-space:nowrap}}
main{{max-width:1120px;margin:0 auto;padding:28px 20px 60px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-top:-64px;position:relative}}
.card{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 16px;box-shadow:0 6px 24px #14161f12}}
.card .v{{font-size:26px;font-weight:700;letter-spacing:-.02em}} .card .l{{font-size:13px;color:var(--muted)}} .card .n{{font-size:11px;color:#999}}
h2{{font-size:19px;margin:36px 0 10px}} table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right}} td:first-child,th:first-child{{text-align:left}}
.figs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}} figure{{margin:0;background:#fff;border:1px solid var(--line);border-radius:12px;padding:10px}}
figure img{{width:100%;height:auto}} figcaption{{font-size:12px;color:var(--muted);margin-top:4px}}
pre{{background:#f3f2ee;border-radius:10px;padding:12px;font-size:12px;overflow:auto}}
@media (max-width:640px){{.legend{{display:none}} .overlay h1{{font-size:22px}}}}
</style></head><body>
<section class="hero"><canvas id="atlas"></canvas>
<div class="overlay"><h1>CLOP embedding atlas</h1><p>{n} held-out sequences, embedded by the trained DNA encoder and laid out in 3D. Drag to rotate, scroll to zoom, hover a point, click a legend entry to isolate it.</p></div>
<div class="controls" id="modes"></div><div class="legend" id="legend"></div><div class="tip" id="tip"></div></section>
<main><div class="cards">{cards}</div>
<h2>Test metrics</h2><table><tr><th>method</th><th>species acc.</th><th>species F1</th><th>biotype F1 (species given)</th><th>cell acc.</th><th>cell F1</th></tr>{table}</table>
<h2>Figures</h2><div class="figs">{images}</div>
<h2>Model configuration</h2><pre>{config}</pre></main>
<script>{script}</script>
<script>
const D={payload};
const canvas=document.getElementById("atlas"),tip=document.getElementById("tip"),legend=document.getElementById("legend"),modes=document.getElementById("modes");
let mode="species";
const view=new Atlas3D(canvas,{{positions:new Float32Array(D.positions),colors:new Float32Array(D.modes[mode].colors),onHover:(i,x,y)=>{{
  if(i<0){{tip.style.display="none";return;}}
  tip.style.display="block";tip.style.left=(x+14)+"px";tip.style.top=(y+10)+"px";
  tip.textContent=D.labels[i]+(D.predictions.length?"  →  called "+D.predictions[i].replace("|"," · "):"")+(D.ids.length?"  ("+D.ids[i]+")":"");}}}});
function groupOf(i){{const [s,b]=D.labels[i].split(" · ");return mode==="species"?s:mode==="biotype"?b:(D.predictions[i]===D.labels[i].replace(" · ","|")?"correct":"wrong");}}
function draw(){{
  view.setColors(new Float32Array(D.modes[mode].colors));view.setHighlight(null);legend.innerHTML="";
  for(const [name,colour] of Object.entries(D.modes[mode].legend)){{const row=document.createElement("div");row.innerHTML=`<i style="background:${{colour}}"></i>${{name}}`;
    row.onclick=()=>{{const s=new Set();for(let i=0;i<D.labels.length;i++)if(groupOf(i)===name)s.add(i);view.setHighlight(view.highlight&&row.classList.contains("on")?null:s);
      [...legend.children].forEach(c=>c.classList.remove("on"));if(view.highlight)row.classList.add("on");}};legend.appendChild(row);}}
  [...modes.children].forEach(b=>b.classList.toggle("on",b.dataset.mode===mode));}}
for(const m of Object.keys(D.modes)){{const b=document.createElement("button");b.textContent="colour by "+m;b.dataset.mode=m;b.onclick=()=>{{mode=m;draw();}};modes.appendChild(b);}}
draw();
</script></body></html>
"""
