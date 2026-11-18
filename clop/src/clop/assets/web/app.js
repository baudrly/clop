/* CLOP web reader: read DNA (one or many records, with a scan along long sequences and robustness checks), search
 * the atlas with words, build zero-shot classifiers from your own prompts, and explore the model. Data come from
 * clop-model.js and clop-atlas.js (globals set by <script> tags, so the page also works when opened from disk). */
(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const PALETTE = ["#4c6ef5", "#ff5a36", "#12b886", "#fab005", "#be4bdb", "#15aabf", "#e64980", "#82c91e", "#7950f2",
    "#fd7e14", "#0ca678", "#adb5bd", "#4dabf7", "#f783ac", "#a9e34b", "#9775fa", "#ffa94d", "#63e6be"];
  const BIOTYPE_PALETTE = ["#4c6ef5", "#ff5a36", "#12b886", "#fab005", "#be4bdb", "#15aabf", "#e64980", "#a9e34b", "#7950f2",
    "#fd7e14", "#868e96", "#99e9f2", "#ffc9c9", "#5c940d", "#ffe066", "#d0bfff", "#c92a2a", "#1864ab"];
  const CONFIDENCE = [["above 80%", "#12b886"], ["60–80%", "#82c91e"], ["40–60%", "#fab005"], ["below 40%", "#ff5a36"]];
  const QUERIES = ["budding yeast transfer rna", "human promoter", "fruit fly coding sequence", "small nuclear rna spliceosomal",
    "thale cress intron", "bacterium coding sequence", "nematode worm three prime utr", "open chromatin accessible region",
    "transposable element transposon repeat", "replication origin ars"];
  const PROMPT_SETS = {
    "Kingdoms": ["bacterium prokaryote", "budding yeast fungus", "flowering plant", "fruit fly insect arthropod", "nematode worm", "human primate mammal"],
    "Gene structure": ["coding sequence cds protein coding exon", "intron spliced", "five prime untranslated region utr",
      "three prime untranslated region utr", "promoter transcription start upstream", "intergenic spacer non genic"],
    "RNA genes": ["transfer rna trna", "small nuclear rna snrna spliceosomal", "small nucleolar rna snorna", "microrna mirna precursor",
      "long non coding rna lncrna", "coding sequence cds"],
    "Regulation": ["enhancer distal regulatory element", "open chromatin accessible regulatory region",
      "promoter transcription start upstream regulatory", "replication origin ars", "intron spliced"],
    "Repeats": ["transposable element transposon repeat", "pseudogene", "coding sequence cds protein coding", "intergenic spacer non genic"],
  };
  const sleep = () => new Promise(r => setTimeout(r, 0));
  const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255);
  const pct = x => `${(100 * x).toFixed(1)}%`;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const pretty = s => s.replace(/_/g, " ");

  if (!globalThis.CLOP_MODEL) {
    $("stats").textContent = "clop-model.js is missing: export the reader with `clop export -m model.pt -o site --web`.";
    return;
  }
  const model = new CLOPModel(CLOP_MODEL), atlas = globalThis.CLOP_ATLAS || null;
  const labels = model.labels, L = labels.length, d = model.cfg.embed_dim, P = model.prototypes.data;
  const labelSpecies = labels.map(l => l.split("|")[0]), labelBiotype = labels.map(l => l.split("|")[1]);
  const sorted = values => [...new Set(values)].sort();
  const paletteOf = (values, colours) => Object.fromEntries(sorted(values).map((v, i) => [v, colours[i % colours.length]]));
  const palettes = {species: paletteOf(labelSpecies, PALETTE), biotype: paletteOf(labelBiotype, BIOTYPE_PALETTE)};
  const short = species => species.replace(/^(\w)\w+ /, "$1. ");
  palettes.called = {"called correctly": "#12b886", "called otherwise": "#ff5a36"};
  palettes.confidence = Object.fromEntries(CONFIDENCE);
  const colourOf = (part, v) => palettes[part][v] || "#868e96";

  // ------------------------------------------------------------------------------------------------ maths
  const dot = (a, b, offset = 0) => { let s = 0; for (let j = 0; j < d; j++) s += a[offset + j] * b[j]; return s; };
  function softmax(logits) {
    const m = Math.max(...logits), e = logits.map(v => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / z);
  }
  const labelProbs = e => softmax(Array.from({length: L}, (_, j) => model.scale * dot(P, e, j * d)));
  function marginal(probs, keys) {
    const totals = {};
    probs.forEach((p, j) => (totals[keys[j]] = (totals[keys[j]] || 0) + p));
    return Object.entries(totals).map(([label, score]) => ({label, score})).sort((a, b) => b.score - a.score);
  }
  const normalise = v => { const n = Math.hypot(...v) || 1; return v.map(x => x / n); };
  const revcomp = s => s.split("").reverse().map(c => ({A: "T", C: "G", G: "C", T: "A"}[c] || "N")).join("");
  const shuffle = s => { const a = s.split(""); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a.join(""); };
  const randomDNA = (n, gc) => Array.from({length: n}, () => (Math.random() < gc ? "GC" : "AT")[Math.random() < 0.5 ? 0 : 1]).join("");
  const gcOf = s => (s.match(/[GC]/g) || []).length / Math.max(1, (s.match(/[ACGT]/g) || []).length);

  // ------------------------------------------------------------------------------------------------ canvas helper
  function canvas2d(c, height) {
    const ratio = window.devicePixelRatio || 1, w = c.clientWidth, h = height || c.clientHeight;
    c.width = Math.round(w * ratio); c.height = Math.round(h * ratio); c.style.height = `${h}px`;
    const ctx = c.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px Inter, system-ui, sans-serif";
    return {ctx, w, h};
  }
  function bars(container, rows, colour, top = 5) {
    rows = rows.slice(0, top);
    container.innerHTML = rows.map(r => `<div class="bar"><div class="track"><div class="fill" style="width:0;background:${colour(r.label)}"></div>
      <span class="name">${esc(pretty(r.label).replace("|", " · "))}</span></div><span class="value">${pct(r.score)}</span></div>`).join("");
    requestAnimationFrame(() => container.querySelectorAll(".fill").forEach((f, i) => (f.style.width = `${Math.max(1.5, 100 * rows[i].score)}%`)));
  }

  // ------------------------------------------------------------------------------------------------ atlas
  let view = null, n = 0, emb = null, groups = {}, mode = "species", isolated = null, anchors = [];
  if (atlas) {
    const bytes = Uint8Array.from(atob(atlas.embeddings.data), c => c.charCodeAt(0)), u16 = new Uint16Array(bytes.buffer);
    emb = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {  // float16 -> float32
      const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      emb[i] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
    }
    n = atlas.species.length;
    const confidence = Array.from({length: n}, (_, i) => {
      const top = Math.max(...labelProbs(emb.subarray(i * d, (i + 1) * d)));
      return CONFIDENCE[top > 0.8 ? 0 : top > 0.6 ? 1 : top > 0.4 ? 2 : 3][0];
    });
    groups = {
      species: atlas.species, biotype: atlas.biotypes, confidence,
      called: atlas.called.map((c, i) => (c === `${atlas.species[i]}|${atlas.biotypes[i]}` ? "called correctly" : "called otherwise")),
    };
    view = new Atlas3D($("atlas"), {
      positions: new Float32Array(atlas.positions), colors: colorsFor(mode), background: null,
      offset: () => (window.innerWidth > 900 ? 0.24 : 0),
      onHover: (i, x, y) => {
        const tip = $("tip");
        if (i < 0) { tip.style.display = "none"; return; }
        const rect = $("atlas").getBoundingClientRect();
        Object.assign(tip.style, {display: "block", left: `${rect.left + x + 14}px`, top: `${rect.top + y + 12}px`});
        tip.innerHTML = `<b>${esc(atlas.species[i])}</b> · ${esc(pretty(atlas.biotypes[i]))}<br><span class="muted">called ${esc(pretty(atlas.called[i]).replace("|", " · "))} · ${esc(atlas.ids[i])}</span>`;
      },
      onFrame: placeLabels,
    });
  } else {
    $("legendPanel").hidden = true;
    $("atlas").hidden = true;
  }
  function colorsFor(name) {
    const out = new Float32Array(3 * n);
    groups[name].forEach((v, i) => out.set(hex(colourOf(name, v)), 3 * i));
    return out;
  }
  const cosines = vector => { const s = new Float32Array(n); for (let i = 0; i < n; i++) s[i] = dot(emb, vector, i * d); return s; };
  const topK = (sims, k) => Array.from(sims.keys()).sort((a, b) => sims[b] - sims[a]).slice(0, k);
  function placeAmong(sims, near) {  // softmax-weighted 3D position of the nearest windows
    let w = 0; const p = [0, 0, 0];
    near.forEach(i => { const wi = Math.exp(25 * (sims[i] - sims[near[0]])); w += wi; for (let k = 0; k < 3; k++) p[k] += wi * atlas.positions[3 * i + k]; });
    return p.map(v => v / w);
  }

  // labels: each group's prompt (pooled over its label prompts, or a custom prompt) placed where it points in the map
  function setAnchors(entries) {
    const box = $("labels");
    box.innerHTML = "";
    anchors = entries.map(([name, vector, colour]) => {
      const sims = cosines(vector), div = document.createElement("div");
      div.textContent = pretty(name);
      div.style.color = colour;
      box.appendChild(div);
      return {xyz: placeAmong(sims, topK(sims, 25)), div};
    });
  }
  function promptAnchors(name) {
    if (name !== "species" && name !== "biotype") return setAnchors(name === "prompts" ? customAnchors : []);
    const keys = name === "species" ? labelSpecies : labelBiotype;
    setAnchors(sorted(keys).map(g => {
      const v = new Array(d).fill(0);
      keys.forEach((k, j) => { if (k === g) for (let t = 0; t < d; t++) v[t] += P[j * d + t]; });
      return [g, normalise(v), colourOf(name, g)];
    }));
  }
  function placeLabels(v) {  // project, then push overlapping labels apart vertically
    const show = $("showLabels").checked, placed = [];
    for (const a of anchors) {
      a.div.style.display = show ? "block" : "none";
      if (!show) continue;
      const [x, y, depth] = v.projectPoint(a.xyz), w = a.width || (a.width = a.div.offsetWidth);
      let top = y;
      for (const p of placed.sort((m, k) => m.top - k.top)) if (Math.abs(p.x - x) < (p.w + w) / 2 && Math.abs(p.top - top) < 14) top = p.top + 14;
      placed.push({x, top, w});
      a.div.style.transform = `translate(${x}px, ${top}px) translate(-50%, -50%)`;
      a.div.style.opacity = Math.max(0.3, Math.min(1, (3.9 - depth) / 1.6));
    }
  }

  function drawLegend() {
    const legend = $("legend"), counts = {};
    groups[mode].forEach(v => (counts[v] = (counts[v] || 0) + 1));
    legend.innerHTML = "";
    const names = mode === "confidence" ? CONFIDENCE.map(c => c[0]) : Object.keys(palettes[mode]).filter(k => counts[k]);
    for (const name of names) {
      const row = document.createElement("div");
      row.innerHTML = `<i style="background:${colourOf(mode, name)};color:${colourOf(mode, name)}"></i><span>${esc(pretty(name))}</span><span class="count">${counts[name] || 0}</span>`;
      row.onclick = () => {
        isolated = isolated === name ? null : name;
        view.setHighlight(isolated ? new Set(groups[mode].flatMap((v, i) => (v === isolated ? [i] : []))) : null);
        [...legend.children].forEach(c => c.classList.toggle("on", c === row && isolated !== null));
      };
      legend.appendChild(row);
    }
  }
  function setMode(name) {
    mode = name; isolated = null;
    view.setColors(colorsFor(mode)); view.setHighlight(null); drawLegend(); promptAnchors(mode);
    [...$("modes").children].forEach(c => c.classList.toggle("on", c.dataset.mode === name));
  }
  function addModeButton(name) {
    const b = document.createElement("button");
    b.textContent = name; b.dataset.mode = name;
    b.onclick = () => setMode(name);
    $("modes").appendChild(b);
  }
  if (atlas) { Object.keys(groups).forEach(addModeButton); setMode("species"); }

  // ------------------------------------------------------------------------------------------------ header stats, tabs
  const params = Object.values(model.bundle.weights).reduce((s, w) => s + w.shape.reduce((a, b) => a * b, 1), 0);
  $("stats").textContent = atlas ? `${n.toLocaleString()} held-out windows · ${L} label prompts · ${params.toLocaleString()} parameters`
    : `${L} label prompts · ${params.toLocaleString()} parameters · no atlas in this export`;
  document.querySelectorAll(".tabs button").forEach(tab => (tab.onclick = () => {
    document.querySelectorAll(".tabs button").forEach(t => t.classList.toggle("on", t === tab));
    document.querySelectorAll("[data-pane]").forEach(p => (p.hidden = p.dataset.pane !== tab.dataset.tab));
    if (tab.dataset.tab === "model") drawHeatmap();
  }));
  $("showLabels").onchange = () => placeLabels(view);

  // ------------------------------------------------------------------------------------------------ input and examples
  const wrap = s => s.match(/.{1,60}/g).join("\n");
  const fasta = (header, seq) => `>${header}\n${wrap(seq)}`;
  function parseFasta(text) {
    const records = [];
    let current = null;
    for (const line of text.split(/\r?\n/)) {
      if (line.startsWith(">")) { current = {header: line.slice(1).trim(), seq: ""}; records.push(current); }
      else if (line.trim()) { if (!current) { current = {header: "query", seq: ""}; records.push(current); } current.seq += line; }
    }
    return records.map(r => ({...r, seq: r.seq.toUpperCase().replace(/U/g, "T").replace(/[^ACGTN]/g, "")})).filter(r => r.seq.length);
  }
  const examples = atlas ? Object.entries(atlas.examples) : [];
  const pick = list => list[Math.floor(Math.random() * list.length)];
  function exampleOf(predicate) {
    const pool = examples.filter(([cell]) => predicate(cell));
    if (!pool.length) return null;
    const [cell, list] = pick(pool), ex = pick(list);
    return {cell, id: ex.id, seq: ex.sequence};
  }
  const select = $("exampleSelect");
  select.innerHTML = `<option value="">${examples.length ? "Choose an annotated example…" : "No annotated examples in this export"}</option>` +
    sorted(examples.map(([c]) => c.split("|")[0])).map(sp => `<optgroup label="${esc(sp)}">${examples.filter(([c]) => c.startsWith(sp + "|"))
      .flatMap(([c, list]) => list.map((ex, k) => `<option value="${esc(c)}#${k}">${esc(pretty(c.split("|")[1]))} · ${esc(ex.id)}</option>`)).join("")}</optgroup>`).join("");
  select.onchange = () => {
    if (!select.value) return;
    const [cell, k] = select.value.split("#"), ex = atlas.examples[cell][+k];
    $("sequence").value = fasta(`${ex.id} species=${cell.split("|")[0]} biotype=${cell.split("|")[1]}`, ex.sequence);
    read();
  };
  const current = () => parseFasta($("sequence").value)[0];
  const PROBES = [
    ["Chimera: genes end to end", () => {
      const counts = {};
      examples.forEach(([c]) => (counts[c.split("|")[0]] = (counts[c.split("|")[0]] || 0) + 1));
      const species = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
      const parts = ["CDS", "intron", "tRNA", "promoter", "three_prime_UTR", "snRNA", "intergenic"]
        .map(b => exampleOf(c => c === `${species}|${b}`)).filter(Boolean).slice(0, 5);
      let at = 0;
      const segments = parts.map(p => `${p.cell.split("|")[1]}:${at}-${(at += p.seq.length)}`);
      return fasta(`chimera species=${species} segments=${segments.join(",")}`, parts.map(p => p.seq).join(""));
    }, () => examples.length],
    ["Six records (batch)", () => sorted(examples.map(([c]) => c.split("|")[0])).map(sp => {
      const ex = exampleOf(c => c.startsWith(sp + "|"));
      return fasta(`${ex.id} species=${sp} biotype=${ex.cell.split("|")[1]}`, ex.seq);
    }).join("\n"), () => examples.length],
    ["Reverse complement", () => { const r = current(); return r && fasta(`${r.header} (reverse complement)`, revcomp(r.seq)); }, () => true],
    ["Shuffled (same composition)", () => { const r = current(); return r && fasta(`${r.header} (shuffled)`, shuffle(r.seq)); }, () => true],
    ["Random, 50% GC", () => fasta("random 50% GC", randomDNA(500, 0.5)), () => true],
    ["Random, 30% GC", () => fasta("random 30% GC", randomDNA(500, 0.3)), () => true],
    ["Telomeric repeat", () => fasta("telomeric repeat (TTAGGG)n", "TTAGGG".repeat(84)), () => true],
    ["CpG-rich", () => fasta("CpG-rich random", Array.from({length: 250}, () => pick(["CG", "GC", "CC", "GG", "CA", "TG"])).join("")), () => true],
  ];
  for (const [name, make, available] of PROBES) {
    if (!available()) continue;
    const b = document.createElement("button");
    b.textContent = name;
    b.onclick = () => { const text = make(); if (!text) { $("readStatus").textContent = "Load a sequence first."; return; } $("sequence").value = text; read(); };
    $("probes").appendChild(b);
  }
  const loadFile = file => file && file.text().then(t => { $("sequence").value = t; read(); });
  $("file").onchange = e => loadFile(e.target.files[0]);
  const drop = $("drop");
  ["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => loadFile(e.dataTransfer.files[0]));

  // ------------------------------------------------------------------------------------------------ reading
  let generation = 0, lastEmbedding = null, batchRows = [];
  async function read() {
    const records = parseFasta($("sequence").value), mine = ++generation;
    if (!records.length) { $("readStatus").textContent = "Paste a DNA sequence first."; return; }
    $("readResult").hidden = true; $("batch").hidden = records.length < 2;
    if (records.length > 1) await readBatch(records.slice(0, 500), mine);
    else await detail(records[0], mine);
  }
  $("read").onclick = read;

  async function readBatch(records, mine) {
    batchRows = [];
    const table = $("batchTable");
    table.innerHTML = "<tr><th>record</th><th>bp</th><th>species</th><th>biotype</th></tr>";
    $("batchHint").textContent = "";
    if (view) { view.clearMarkers(); view.setHighlight(new Set()); }
    for (const [k, r] of records.entries()) {
      if (mine !== generation) return;
      $("readStatus").textContent = `record ${k + 1} of ${records.length}…`;
      await sleep();
      const e = model.embedDNA(r.seq.slice(0, 50000)), probs = labelProbs(e);
      const sp = marginal(probs, labelSpecies)[0], bt = marginal(probs, labelBiotype)[0];
      batchRows.push([r.header.split(/\s/)[0], r.seq.length, sp.label, sp.score, bt.label, bt.score]);
      const row = table.insertRow(), truth = key => (r.header.match(new RegExp(`${key}=(\\S+(?: [a-z]\\S*)?)`)) || [])[1];
      const mark = (key, call) => (truth(key) ? (truth(key) === call ? " <span class='ok'>✓</span>" : ` <span class='bad' title="annotated ${esc(truth(key))}">✗</span>`) : "");
      row.innerHTML = `<td title="${esc(r.header)}">${esc(r.header.split(/\s/)[0])}</td><td>${r.seq.length}</td>
        <td><i style="background:${colourOf("species", sp.label)}"></i>${esc(short(sp.label))} <span class="muted">${pct(sp.score)}</span>${mark("species", sp.label)}</td>
        <td><i style="background:${colourOf("biotype", bt.label)}"></i>${esc(pretty(bt.label))} <span class="muted">${pct(bt.score)}</span>${mark("biotype", bt.label)}</td>`;
      row.onclick = () => detail(r, ++generation);
      if (view) { const sims = cosines(e); view.addMarker(placeAmong(sims, topK(sims, 12)), hex(colourOf("species", sp.label))); }
    }
    $("batchHint").textContent = `${records.length} read · click a row for details`;
    $("readStatus").textContent = "";
  }
  $("download").onclick = () => {
    const tsv = ["record\tlength\tspecies\tspecies_score\tbiotype\tbiotype_score", ...batchRows.map(r => r.map(v => (typeof v === "number" && v < 1 ? v.toFixed(4) : v)).join("\t"))].join("\n");
    const a = Object.assign(document.createElement("a"), {href: URL.createObjectURL(new Blob([tsv], {type: "text/tab-separated-values"})), download: "clop_calls.tsv"});
    a.click(); URL.revokeObjectURL(a.href);
  };

  async function detail(record, mine) {
    const seq = record.seq.slice(0, 50000);
    if (seq.length < 20) { $("readStatus").textContent = "Paste at least 20 bp (100 bp or more reads best)."; return; }
    $("read").disabled = true; $("readStatus").textContent = "embedding…";
    await sleep();
    const t0 = performance.now(), e = model.embedDNA(seq), probs = labelProbs(e), ms = performance.now() - t0;
    lastEmbedding = e;
    const species = marginal(probs, labelSpecies), biotype = marginal(probs, labelBiotype);
    const cells = probs.map((score, j) => ({label: labels[j], score})).sort((a, b) => b.score - a.score);
    const windows = model.tileStarts(seq.length).length;
    $("summary").innerHTML = `<b>${esc(record.header)}</b><span>${seq.length.toLocaleString()} bp · GC ${pct(gcOf(seq))}${windows > 1 ? ` · read as ${windows} overlapping windows` : ""} · ${ms.toFixed(0)} ms</span>`;
    bars($("speciesBars"), species, l => colourOf("species", l));
    bars($("biotypeBars"), biotype, l => colourOf("biotype", l));
    bars($("cellBars"), cells, l => colourOf("species", l.split("|")[0]), 4);
    if (view) {
      const sims = cosines(e), near = topK(sims, 12);
      $("neighbours").innerHTML = near.slice(0, 6).map(i => `<li><i style="background:${colourOf("species", atlas.species[i])}"></i>${esc(atlas.species[i])} · ${esc(pretty(atlas.biotypes[i]))} <span class="muted">${esc(atlas.ids[i])} · ${sims[i].toFixed(3)}</span></li>`).join("");
      view.clearMarkers(); view.addMarker(placeAmong(sims, near), [1, 1, 1]); view.setHighlight(new Set(near));
    } else {
      $("neighbours").innerHTML = "<li class='muted'>no atlas in this export</li>";
    }
    $("readResult").hidden = false; $("read").disabled = false;
    $("readStatus").textContent = "checking robustness…";
    await robustness(seq, e, species[0], biotype[0], mine);
    if (mine === generation) await scan(record, seq, mine);
    if (mine === generation) $("readStatus").textContent = "";
  }

  async function robustness(seq, e, topSpecies, topBiotype, mine) {
    await sleep();
    const rc = model.embedDNA(revcomp(seq)), shuffled = model.embedDNA(shuffle(seq));
    const shuffledBiotype = marginal(labelProbs(shuffled), labelBiotype)[0];
    $("checks").innerHTML = `<div><b>${dot(rc, e).toFixed(6)}</b><span>cosine to the reverse complement: the encoder reads both strands</span></div>
      <div><b>${dot(shuffled, e).toFixed(3)}</b><span>cosine to a shuffle with the same composition, called ${esc(pretty(shuffledBiotype.label))} (${pct(shuffledBiotype.score)}): how much the base order matters</span></div>`;
    const sizes = [60, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000].filter(s => s < seq.length).concat([seq.length]);
    const rows = [];
    for (const size of sizes) {
      if (mine !== generation) return;
      await sleep();
      const start = Math.floor((seq.length - size) / 2), probs = labelProbs(model.embedDNA(seq.slice(start, start + size)));
      rows.push([size, marginal(probs, labelSpecies).find(r => r.label === topSpecies.label).score, marginal(probs, labelBiotype).find(r => r.label === topBiotype.label).score]);
    }
    lengthChart(rows, topSpecies.label, topBiotype.label);
  }

  function lengthChart(rows, species, biotype) {
    const {ctx, w, h} = canvas2d($("lengthChart"), 130), left = 30, right = 8, top = 26, bottom = 20;
    const lo = Math.log(rows[0][0]), hi = Math.log(Math.max(rows[rows.length - 1][0], rows[0][0] + 1));
    const x = s => left + (w - left - right) * (Math.log(s) - lo) / (hi - lo || 1), y = p => top + (h - top - bottom) * (1 - p);
    ctx.strokeStyle = "rgba(255,255,255,0.1)"; ctx.fillStyle = "#a3a8c3";
    [0, 0.5, 1].forEach(p => { ctx.beginPath(); ctx.moveTo(left, y(p)); ctx.lineTo(w - right, y(p)); ctx.stroke(); ctx.fillText(`${p * 100}%`, 2, y(p) + 4); });
    rows.forEach(([s], k) => (k === 0 || k === rows.length - 1 || k % 2 === 0) && ctx.fillText(`${s}`, x(s) - 8, h - 5));
    [[1, colourOf("species", species), species], [2, colourOf("biotype", biotype), pretty(biotype)]].forEach(([col, colour, name], k) => {
      ctx.strokeStyle = colour; ctx.fillStyle = colour; ctx.lineWidth = 2; ctx.beginPath();
      rows.forEach((r, i) => (i ? ctx.lineTo(x(r[0]), y(r[col])) : ctx.moveTo(x(r[0]), y(r[col]))));
      ctx.stroke();
      rows.forEach(r => { ctx.beginPath(); ctx.arc(x(r[0]), y(r[col]), 2.5, 0, 7); ctx.fill(); });
      ctx.fillText(`P(${name})`, left + 4 + k * (w / 2.3), 11);
    });
    ctx.fillStyle = "#a3a8c3"; ctx.textAlign = "right"; ctx.fillText("centred crop, bp →", w - right, h - 5 - 12); ctx.textAlign = "left";
  }

  // scan: biotype probabilities in sliding windows along sequences longer than the model's window
  async function scan(record, seq, mine) {
    const size = Math.min(300, model.cfg.window || 500);
    if (seq.length < 2 * size) { $("scanBlock").hidden = true; return; }
    const stride = Math.max(size / 3, Math.ceil((seq.length - size) / 200)), starts = [];
    for (let s = 0; s + size <= seq.length; s += stride) starts.push(Math.round(s));
    const truth = ((record.header.match(/segments=(\S+)/) || [])[1] || "").split(",").filter(Boolean)
      .map(t => { const [b, r] = t.split(":"); const [a, z] = r.split("-").map(Number); return {b, a, z}; });
    const columns = [];
    $("scanBlock").hidden = false;
    $("scanHint").textContent = `${size} bp windows every ${stride} bp`;
    for (const [k, s] of starts.entries()) {
      if (mine !== generation) return;
      const probs = labelProbs(model.embedDNA(seq.slice(s, s + size)));
      columns.push({s, biotype: marginal(probs, labelBiotype), species: marginal(probs, labelSpecies)[0]});
      if (k % 4 === 3 || k === starts.length - 1) { drawScan(columns, seq.length, size, truth); await sleep(); }
    }
    const seen = new Set(columns.flatMap(c => c.biotype.filter(r => r.score > 0.15).map(r => r.label)));
    const speciesSeen = new Set(columns.map(c => c.species.label));
    $("scanLegend").innerHTML = [...seen].map(b => `<span><i style="background:${colourOf("biotype", b)}"></i>${esc(pretty(b))}</span>`).join("") +
      [...speciesSeen].map(sp => `<span><i style="background:${colourOf("species", sp)}"></i><em>${esc(short(sp))}</em></span>`).join("");
  }
  function drawScan(columns, length, size, truth) {
    const band = 10, strips = truth.length ? 3 : 2, top = strips * (band + 3) + 3, bottom = 16;
    const {ctx, w, h} = canvas2d($("scan"), 190), left = 64, right = 4;
    const x = pos => left + (w - left - right) * pos / length, plotH = h - top - bottom, order = sorted(labelBiotype);
    const names = [...(truth.length ? ["annotation"] : []), "biotype call", "species call"];
    ctx.fillStyle = "#a3a8c3";
    names.forEach((name, k) => ctx.fillText(name, 0, k * (band + 3) + band - 1));
    ctx.fillText("biotype P", 0, top + 10);
    if (truth.length) truth.forEach(t => { ctx.fillStyle = colourOf("biotype", t.b); ctx.fillRect(x(t.a), 0, x(t.z) - x(t.a) - 1, band); });
    columns.forEach((c, k) => {
      const a = x(c.s + size / 2 - (k ? (c.s - columns[k - 1].s) / 2 : size / 2)), z = x(c.s + size / 2 + (k < columns.length - 1 ? (columns[k + 1].s - c.s) / 2 : size / 2));
      const row = truth.length ? 1 : 0;
      ctx.fillStyle = colourOf("biotype", c.biotype[0].label);
      ctx.fillRect(a, row * (band + 3), z - a + 0.5, band);
      ctx.fillStyle = colourOf("species", c.species.label);
      ctx.fillRect(a, (row + 1) * (band + 3), z - a + 0.5, band);
      let y = top;
      const scores = Object.fromEntries(c.biotype.map(r => [r.label, r.score]));
      for (const b of order) { const hgt = plotH * (scores[b] || 0); ctx.fillStyle = colourOf("biotype", b); ctx.fillRect(a, y, z - a + 0.5, hgt); y += hgt; }
    });
    ctx.fillStyle = "#a3a8c3";
    [0, 0.25, 0.5, 0.75, 1].forEach(f => ctx.fillText(`${Math.round(f * length)}`, Math.min(w - 34, Math.max(left, x(f * length) - 10)), h - 3));
  }

  // ------------------------------------------------------------------------------------------------ search
  const speciesList = sorted(labelSpecies), biotypeList = sorted(labelBiotype);
  $("composeSpecies").innerHTML = `<option value="">any species</option>` + speciesList.map(s => `<option>${esc(s)}</option>`).join("");
  $("composeBiotype").innerHTML = `<option value="">any biotype</option>` + biotypeList.map(b => `<option value="${esc(b)}">${esc(pretty(b))}</option>`).join("");
  $("composeSpecies").onchange = $("composeBiotype").onchange = () => {
    $("query").value = model.caption($("composeSpecies").value, $("composeBiotype").value || ""); search();
  };
  QUERIES.forEach(q => { const b = document.createElement("button"); b.textContent = q; b.onclick = () => { $("query").value = q; search(); }; $("queries").appendChild(b); });
  function unknownWords(text) { return (text.toLowerCase().match(/[a-z0-9]+(?:'[a-z]+)?/g) || []).filter(w => !(w in model.vocab)); }
  function search() {
    const text = $("query").value.trim();
    if (!text) return;
    if (!atlas) { $("searchSummary").textContent = "This export has no atlas to search."; $("searchResult").hidden = false; return; }
    const sims = cosines(model.embedText(text)), k = 150, top = topK(sims, k), counts = {};
    top.forEach(i => { const key = `${atlas.species[i]}|${atlas.biotypes[i]}`; counts[key] = (counts[key] || 0) + 1; });
    const rows = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([label, c]) => ({label, score: c / k}));
    const unknown = unknownWords(text);
    $("searchSummary").innerHTML = `The ${k} windows closest to <i>“${esc(text)}”</i> are mostly:` +
      (unknown.length ? `<br><span class="muted">words the model never saw (ignored): ${esc(unknown.join(", "))}</span>` : "");
    $("searchResult").hidden = false;  // visible before drawing, so the canvas has a width
    bars($("searchBars"), rows, l => colourOf("species", l.split("|")[0]), 6);
    histogram(sims, sims[top[k - 1]]);
    view.clearMarkers(); view.setHighlight(new Set(top)); isolated = null;
  }
  function histogram(sims, threshold) {
    const {ctx, w, h} = canvas2d($("simHist"), 90), bins = 60;
    let lo = Infinity, hi = -Infinity;
    sims.forEach(s => { lo = Math.min(lo, s); hi = Math.max(hi, s); });
    const counts = new Array(bins).fill(0);
    sims.forEach(s => counts[Math.min(bins - 1, Math.floor((s - lo) / (hi - lo + 1e-9) * bins))]++);
    const max = Math.max(...counts), bw = w / bins;
    counts.forEach((c, b) => { ctx.fillStyle = lo + (b + 1) * (hi - lo) / bins > threshold ? "#ff5a36" : "rgba(163,168,195,0.45)"; ctx.fillRect(b * bw, h - 14 - (h - 18) * c / max, bw - 1, (h - 18) * c / max); });
    ctx.fillStyle = "#a3a8c3"; ctx.fillText(lo.toFixed(2), 0, h - 2); ctx.fillText(`cosine ${hi.toFixed(2)}`, w - 70, h - 2);
  }
  $("search").onclick = search;
  $("query").addEventListener("keydown", e => e.key === "Enter" && search());

  // ------------------------------------------------------------------------------------------------ custom prompts
  let customAnchors = [];
  const promptList = () => $("customPrompts").value.split("\n").map(s => s.trim()).filter(Boolean);
  $("customPrompts").value = PROMPT_SETS["Gene structure"].join("\n");
  Object.entries(PROMPT_SETS).forEach(([name, list]) => {
    const b = document.createElement("button");
    b.textContent = name; b.onclick = () => { $("customPrompts").value = list.join("\n"); if (atlas) paint(); };
    $("promptSets").appendChild(b);
  });
  function paint() {
    const prompts = promptList();
    if (prompts.length < 2) { $("customTitle").textContent = "Write at least two prompts."; $("customResult").hidden = false; return; }
    if (!atlas) return;
    const vectors = prompts.map(p => model.embedText(p));
    palettes.prompts = Object.fromEntries(prompts.map((p, k) => [p, PALETTE[k % PALETTE.length]]));
    groups.prompts = Array.from({length: n}, (_, i) => {
      let best = 0, bestS = -Infinity;
      vectors.forEach((v, k) => { const s = dot(emb, v, i * d); if (s > bestS) { bestS = s; best = k; } });
      return prompts[best];
    });
    customAnchors = prompts.map((p, k) => [p, vectors[k], palettes.prompts[p]]);
    if (!$("modes").querySelector('[data-mode="prompts"]')) addModeButton("prompts");
    setMode("prompts");
    const counts = {};
    groups.prompts.forEach(p => (counts[p] = (counts[p] || 0) + 1));
    $("customTitle").textContent = "Share of atlas windows called by each prompt";
    bars($("customBars"), prompts.map(p => ({label: p, score: (counts[p] || 0) / n})).sort((a, b) => b.score - a.score), l => palettes.prompts[l], prompts.length);
    $("customResult").hidden = false;
  }
  $("paintMap").onclick = paint;
  $("classifyCustom").onclick = () => {
    const prompts = promptList(), record = current();
    if (!record || prompts.length < 2) { $("customTitle").textContent = "Load a sequence on the Read tab and write at least two prompts."; $("customResult").hidden = false; return; }
    const e = lastEmbedding && $("readResult").hidden === false ? lastEmbedding : model.embedDNA(record.seq.slice(0, 50000));
    const probs = softmax(prompts.map(p => model.scale * dot(model.embedText(p), e)));
    const unknown = [...new Set(prompts.flatMap(unknownWords))];
    $("customTitle").textContent = `Zero-shot call of “${record.header.split(/\s/)[0]}” among your prompts` + (unknown.length ? ` (unknown words ignored: ${unknown.join(", ")})` : "");
    bars($("customBars"), prompts.map((p, k) => ({label: p, score: probs[k]})).sort((a, b) => b.score - a.score), l => PALETTE[prompts.indexOf(l) % PALETTE.length], prompts.length);
    $("customResult").hidden = false;
  };

  // ------------------------------------------------------------------------------------------------ model tab
  const stat = (value, name) => `<div><b>${value}</b><span>${name}</span></div>`;
  $("modelStats").innerHTML = [
    stat(params.toLocaleString(), "parameters"), stat(d, "embedding dimensions"), stat(L, "label prompts"),
    stat(`${model.cfg.window || "–"} bp`, "training window"), stat(model.cfg.pooling || "moments", "pooling"),
    stat(atlas ? n.toLocaleString() : "–", "atlas windows"),
  ].join("");
  let heatmapDrawn = false;
  function drawHeatmap() {
    if (heatmapDrawn) return;
    heatmapDrawn = true;
    const c = $("heatmap"), size = c.clientWidth, {ctx} = canvas2d(c, size), cell = size / L;
    for (let a = 0; a < L; a++) for (let b = 0; b < L; b++) {
      const s = dot(P, P.subarray(b * d, (b + 1) * d), a * d), t = Math.max(0, Math.min(1, (s + 0.2) / 1.2));
      ctx.fillStyle = `rgb(${Math.round(20 + 235 * t)},${Math.round(22 + 68 * t * t)},${Math.round(40 + 14 * t)})`;
      ctx.fillRect(b * cell, a * cell, Math.ceil(cell), Math.ceil(cell));
    }
    c.onmousemove = ev => {
      const r = c.getBoundingClientRect(), a = Math.floor((ev.clientY - r.top) / cell), b = Math.floor((ev.clientX - r.left) / cell), tip = $("tip");
      if (a < 0 || b < 0 || a >= L || b >= L) { tip.style.display = "none"; return; }
      Object.assign(tip.style, {display: "block", left: `${ev.clientX + 14}px`, top: `${ev.clientY + 12}px`});
      tip.innerHTML = `${esc(pretty(labels[a]).replace("|", " · "))}<br>${esc(pretty(labels[b]).replace("|", " · "))}<br><span class="muted">cosine ${dot(P, P.subarray(b * d, (b + 1) * d), a * d).toFixed(3)}</span>`;
    };
    c.onmouseleave = () => ($("tip").style.display = "none");
  }
})();
