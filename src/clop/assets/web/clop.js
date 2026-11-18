/* CLOP in the browser: the CNN DNA encoder and bag-of-words text encoder of a `clop export --web` bundle, in plain
 * JavaScript (no runtime, no WebAssembly). Numerically matches PyTorch to ~1e-7.
 *   load clop-model.js with a script tag first (it sets globalThis.CLOP_MODEL; works from file:// as well as http)
 *   const model = new CLOPModel(CLOP_MODEL);
 *   model.embedDNA("ACGT...")  -> Float32Array (unit norm);  model.embedText("homo sapiens promoter")
 *   model.classify(embedding) -> [{label, score}] over the bundle's labels;  model.caption(species, biotype)
 */
(function (root) {
  "use strict";
  const CODE = new Uint8Array(256).fill(4);
  "ACGT".split("").forEach((b, i) => { CODE[b.charCodeAt(0)] = i; CODE[b.toLowerCase().charCodeAt(0)] = i; });
  CODE["U".charCodeAt(0)] = CODE["u".charCodeAt(0)] = 3;

  function decode(entry) {
    const bytes = Uint8Array.from(atob(entry.data), c => c.charCodeAt(0));
    return {shape: entry.shape, data: new Float32Array(bytes.buffer)};
  }

  // erf with |error| < 1.2e-7 (Numerical Recipes erfc), so GELU matches torch's exact (erf) form
  function erf(x) {
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
      t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? 1 - r : r - 1;
  }
  const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2));

  function encode(sequence) {
    const s = sequence.replace(/[^A-Za-z]/g, ""), codes = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) codes[i] = CODE[s.charCodeAt(i)];
    return codes;
  }

  function reverseComplement(codes) {
    const n = codes.length, out = new Uint8Array(n);
    for (let i = 0; i < n; i++) { const c = codes[n - 1 - i]; out[i] = c < 4 ? 3 - c : 4; }
    return out;
  }

  function linear(x, w, b, act) {  // w: (out, in)
    const [nOut, nIn] = w.shape, out = new Float32Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let acc = b.data[o];
      const row = o * nIn;
      for (let i = 0; i < nIn; i++) acc += w.data[row + i] * x[i];
      out[o] = act ? gelu(acc) : acc;
    }
    return out;
  }

  function normalise(v) {
    let n = 0;
    for (const x of v) n += x * x;
    n = Math.sqrt(n) || 1;
    return v.map(x => x / n);
  }

  class CLOPModel {
    constructor(bundle) {
      this.bundle = bundle;
      this.cfg = bundle.config;
      this.w = {};
      for (const [k, v] of Object.entries(bundle.weights)) this.w[k] = decode(v);
      this.vocab = bundle.vocab;
      this.labels = bundle.labels;
      this.scale = bundle.logit_scale;
      this.prototypes = decode(bundle.prototypes);
    }

    /* Strand features: stem conv (one-hot input) -> GELU -> BN -> maxpool 2 -> dilated residual blocks -> pooling:
     * mean and mean square over interior positions ("moments", length-invariant), or mean and max ("meanmax"). */
    features(codes) {
      const W = this.cfg.width, K = this.cfg.kernel, pad = K >> 1, L = codes.length, w = this.w;
      const stemW = w["stem.w"].data, stemB = w["stem.b"].data, stemS = w["stem.s"].data, stemT = w["stem.t"].data;
      const T = Math.ceil(L / 2);
      let x = new Float32Array(W * T);
      const tmp = new Float32Array(L);
      for (let c = 0; c < W; c++) {
        for (let t = 0; t < L; t++) {
          let acc = stemB[c];
          for (let k = 0; k < K; k++) {
            const p = t + k - pad;
            if (p >= 0 && p < L && codes[p] < 4) acc += stemW[(c * 4 + codes[p]) * K + k];
          }
          tmp[t] = gelu(acc) * stemS[c] + stemT[c];
        }
        for (let t = 0; t < T; t++) x[c * T + t] = 2 * t + 1 < L ? Math.max(tmp[2 * t], tmp[2 * t + 1]) : tmp[2 * t];
      }
      const y = new Float32Array(W * T);
      for (let b = 0; b < this.cfg.depth; b++) {
        const d = 1 << b, cw = w[`block${b}.w`].data, cb = w[`block${b}.b`].data, cs = w[`block${b}.s`].data, ct = w[`block${b}.t`].data;
        y.fill(0);
        for (let o = 0; o < W; o++) {
          const yo = y.subarray(o * T, (o + 1) * T);
          yo.fill(cb[o]);
          for (let i = 0; i < W; i++) {
            const xi = x.subarray(i * T, (i + 1) * T);
            for (let k = 0; k < 5; k++) {
              const wv = cw[(o * W + i) * 5 + k], shift = (k - 2) * d;
              const lo = Math.max(0, -shift), hi = Math.min(T, T - shift);
              for (let t = lo; t < hi; t++) yo[t] += wv * xi[t + shift];
            }
          }
          for (let t = 0; t < T; t++) yo[t] = gelu(yo[t]) * cs[o] + ct[o];
        }
        for (let j = 0; j < x.length; j++) x[j] += y[j];
      }
      const pooled = new Float32Array(2 * W), meanmax = this.cfg.pooling === "meanmax", m = this.cfg.margin;
      const [lo, hi] = meanmax || T <= 2 * m ? [0, T] : [m, T - m];
      for (let c = 0; c < W; c++) {
        let sum = 0, second = meanmax ? -Infinity : 0;
        for (let t = lo; t < hi; t++) {
          const v = x[c * T + t];
          sum += v;
          second = meanmax ? Math.max(second, v) : second + v * v;
        }
        pooled[c] = sum / (hi - lo);
        pooled[W + c] = meanmax ? second : second / (hi - lo);
      }
      return pooled;
    }

    embedCodes(codes) {
      let pooled = this.features(codes);
      if (this.cfg.rc) {
        const other = this.features(reverseComplement(codes));
        pooled = pooled.map((v, i) => v + other[i]);
      }
      const h = linear(pooled, this.w["dna.head0.w"], this.w["dna.head0.b"], true);
      return normalise(linear(h, this.w["dna.head3.w"], this.w["dna.head3.b"], false));
    }

    /* Starts of the half-overlapping training-size windows that cover a sequence of L bases (as Python tile_starts). */
    tileStarts(L) {
      const size = this.cfg.window;
      if (!size || L <= size) return [0];
      const stride = Math.max(size >> 1, 1), count = Math.ceil((L - size) / stride) + 1;
      return Array.from({length: count}, (_, i) => Math.min(i * stride, L - size));
    }

    /* Inputs longer than the training window: normalised mean over half-overlapping windows (as Pipeline.embed). */
    embedDNA(sequence) {
      const codes = encode(sequence), size = this.cfg.window, starts = this.tileStarts(codes.length);
      if (!codes.length) throw new Error("empty sequence");
      if (starts.length === 1) return this.embedCodes(codes);
      const mean = new Float32Array(this.cfg.embed_dim);
      for (const start of starts) {
        const e = this.embedCodes(codes.subarray(start, start + size));
        for (let j = 0; j < mean.length; j++) mean[j] += e[j];
      }
      return normalise(mean);
    }

    tokens(text) {
      const words = text.toLowerCase().match(/[a-z0-9]+(?:'[a-z]+)?/g) || [];
      const ids = words.slice(0, 64).map(word => this.vocab[word] ?? 1);
      return ids.length ? ids : [1];
    }

    embedText(text) {
      const ids = this.tokens(text), E = this.w["text.embed"], width = E.shape[1], mean = new Float32Array(width);
      for (const id of ids) for (let j = 0; j < width; j++) mean[j] += E.data[id * width + j] / ids.length;
      const h = linear(mean, this.w["text.head0.w"], this.w["text.head0.b"], true);
      return normalise(linear(h, this.w["text.head3.w"], this.w["text.head3.b"], false));
    }

    caption(species, biotype) {
      const key = species.trim().toLowerCase().replace(/_/g, " ");
      const lineage = this.bundle.lineage[key] || "";
      const words = this.bundle.biotype_words[biotype.trim().toLowerCase()] || biotype.toLowerCase().replace(/_/g, " ");
      return [key, lineage, words].filter(Boolean).join(" ");
    }

    /* Softmax over the bundle's label prompts (or over custom prompt embeddings). */
    classify(embedding, prototypes = null, labels = null) {
      const P = prototypes || this.prototypes, names = labels || this.labels, d = embedding.length;
      const n = P.data ? P.shape[0] : P.length, logits = new Float32Array(n);
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let i = 0; i < d; i++) s += embedding[i] * (P.data ? P.data[j * d + i] : P[j][i]);
        logits[j] = this.scale * s;
      }
      const max = Math.max(...logits);
      let total = 0;
      const p = Array.from(logits, v => { const e = Math.exp(v - max); total += e; return e; });
      return p.map((v, j) => ({label: names[j], score: v / total})).sort((a, b) => b.score - a.score);
    }
  }

  root.CLOPModel = CLOPModel;
  root.CLOPEncode = encode;
  if (typeof module !== "undefined") module.exports = {CLOPModel, encode};
})(typeof window !== "undefined" ? window : globalThis);
