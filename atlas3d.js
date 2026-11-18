/* Atlas3D: a dependency-free WebGL point-cloud viewer for CLOP embeddings.
 * Drag to orbit, scroll or pinch to zoom, hover to inspect; auto-rotates when idle.
 *   const view = new Atlas3D(canvas, {positions, colors, onHover});
 *   view.setColors(rgbFloat32Array); view.setHighlight(indexSet | null); view.addMarker([x, y, z]);
 *   view.onFrame = view => {...}   (called every frame, e.g. to place HTML labels with view.projectPoint([x, y, z]))
 * positions: Float32Array of xyz (any scale; centred and scaled so that 98% of points fall in the unit ball),
 * colors: Float32Array of rgb in [0, 1].
 */
(function (root) {
  "use strict";
  const VERTEX = `
    attribute vec3 position; attribute vec3 color; attribute float weight;
    uniform mat4 view; uniform float scale; uniform float pixelRatio; uniform float offset;
    varying vec3 vColor; varying float vDepth; varying float vWeight;
    void main() {
      vec4 p = view * vec4(position, 1.0);
      float depth = -p.z;
      gl_Position = vec4(p.xy * scale / depth + vec2(offset, 0.0), (depth - 0.1) / 20.0, 1.0);
      gl_PointSize = pixelRatio * (3.5 + 9.0 * weight) * scale / depth;
      vColor = color; vDepth = clamp((4.6 - depth) / 2.4, 0.0, 1.0); vWeight = weight;
    }`;
  const FRAGMENT = `
    precision mediump float;
    varying vec3 vColor; varying float vDepth; varying float vWeight; uniform float dim;
    void main() {
      vec2 d = gl_PointCoord - 0.5; float r = dot(d, d);
      if (r > 0.25) discard;
      float core = smoothstep(0.25, 0.0, r);
      float alpha = core * mix(0.25, 0.95, vDepth) * mix(dim, 1.0, vWeight);
      gl_FragColor = vec4(vColor * mix(0.55, 1.15, vDepth), alpha);
    }`;

  function compile(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  }

  function normalise(positions) {
    const n = positions.length / 3, centre = [0, 0, 0];
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) centre[k] += positions[3 * i + k] / n;
    const out = new Float32Array(positions.length), radii = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < 3; k++) { out[3 * i + k] = positions[3 * i + k] - centre[k]; radii[i] += out[3 * i + k] ** 2; }
    }
    radii.sort();  // robust scale: a few far outliers must not shrink the whole cloud
    const radius = Math.sqrt(radii[Math.min(n - 1, Math.floor(0.98 * n))]) || 1;
    for (let i = 0; i < out.length; i++) out[i] /= radius;
    return {out, centre, radius};
  }

  class Atlas3D {
    constructor(canvas, options) {
      this.canvas = canvas;
      this.gl = canvas.getContext("webgl", {antialias: true, premultipliedAlpha: false});
      if (!this.gl) throw new Error("WebGL unavailable");
      this.onHover = options.onHover || null;
      this.onFrame = options.onFrame || null;
      this.yaw = 0.6; this.pitch = 0.35; this.distance = canvas.clientWidth < canvas.clientHeight ? 4.4 : 2.6;  // farther on portrait screens this.autoRotate = options.autoRotate !== false;
      this.markers = [];
      this.background = options.background === undefined ? [0.05, 0.06, 0.12] : options.background;  // null: transparent
      this.offset = options.offset || 0;  // horizontal shift of the cloud in clip space (e.g. beside a side panel)
      this._init();
      this.setData(options.positions, options.colors);
      this._bind();
      this._loop = this._loop.bind(this);
      requestAnimationFrame(this._loop);
    }

    _init() {
      const gl = this.gl;
      const program = gl.createProgram();
      gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX));
      gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT));
      gl.linkProgram(program);
      this.program = program;
      this.loc = {};
      for (const a of ["position", "color", "weight"]) this.loc[a] = gl.getAttribLocation(program, a);
      for (const u of ["view", "scale", "pixelRatio", "dim", "offset"]) this.loc[u] = gl.getUniformLocation(program, u);
      this.buffers = {position: gl.createBuffer(), color: gl.createBuffer(), weight: gl.createBuffer()};
    }

    setData(positions, colors) {
      const {out, centre, radius} = normalise(positions);
      this.positions = out; this.centre = centre; this.radius = radius; this.n = out.length / 3;
      this.base = new Float32Array(this.n).fill(0.35);
      this._upload("position", out, 3);
      this.setColors(colors);
      this.setHighlight(null);
    }

    /* Map a point given in the original (unnormalised) coordinates into the viewer's space. */
    toView(xyz) { return xyz.map((v, k) => (v - this.centre[k]) / this.radius); }

    _upload(name, data, size) {
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers[name]);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      this[`_${name}Size`] = size;
    }

    setColors(colors) { this.colors = colors; this._upload("color", colors, 3); }

    setHighlight(indices) {
      this.highlight = indices;
      const w = new Float32Array(this.n + this.markers.length);
      for (let i = 0; i < this.n; i++) w[i] = indices ? (indices.has(i) ? 1.0 : 0.12) : this.base[i];
      for (let j = 0; j < this.markers.length; j++) w[this.n + j] = 1.0;
      this.weights = w;
      this._upload("weight", w, 1);
    }

    addMarker(xyz, rgb = [1, 1, 1]) {
      this.markers.push({xyz: this.toView(xyz), rgb});
      const p = new Float32Array(this.positions.length + 3 * this.markers.length);
      p.set(this.positions);
      const c = new Float32Array(p.length);
      c.set(this.colors);
      this.markers.forEach((m, j) => { p.set(m.xyz, this.positions.length + 3 * j); c.set(m.rgb, this.positions.length + 3 * j); });
      this._upload("position", p, 3);
      this._upload("color", c, 3);
      this.setHighlight(this.highlight);
    }

    clearMarkers() { this.markers = []; this._upload("position", this.positions, 3); this.setColors(this.colors); this.setHighlight(this.highlight); }

    _matrix() {
      const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
      // rotation (yaw about y, then pitch about x), then translation along -z by the viewing distance
      return new Float32Array([cy, sy * sp, -sy * cp, 0, 0, cp, sp, 0, sy, -cy * sp, cy * cp, 0, 0, 0, -this.distance, 1]);
    }

    project(i) { return this._project(this.positions[3 * i], this.positions[3 * i + 1], this.positions[3 * i + 2]); }

    /* Screen position (CSS pixels) and depth of a point in the original coordinates. */
    projectPoint(xyz) { return this._project(...this.toView(xyz)); }

    _project(x, y, z) {
      const m = this._matrix();
      const px = m[0] * x + m[4] * y + m[8] * z + m[12], py = m[1] * x + m[5] * y + m[9] * z + m[13];
      const pz = m[2] * x + m[6] * y + m[10] * z + m[14], depth = -pz, s = this._scale();
      const aspect = this.canvas.width / this.canvas.height;
      const off = typeof this.offset === "function" ? this.offset() : this.offset;
      return [(px * s / depth / aspect + off + 1) * 0.5 * this.canvas.clientWidth, (1 - py * s / depth) * 0.5 * this.canvas.clientHeight, depth];
    }

    _scale() { return 2.2; }

    _bind() {
      const c = this.canvas;
      let dragging = false, last = null, idle = 0;
      const pointer = e => (e.touches ? e.touches[0] : e);
      const down = e => { dragging = true; last = pointer(e); this.autoRotateHeld = true; };
      const move = e => {
        const p = pointer(e), rect = c.getBoundingClientRect();
        if (dragging && last) {
          this.yaw += (p.clientX - last.clientX) * 0.008;
          this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch + (p.clientY - last.clientY) * 0.008));
          last = p;
          if (e.cancelable) e.preventDefault();
        } else if (this.onHover && !e.touches) {
          this._hover(p.clientX - rect.left, p.clientY - rect.top);
        }
      };
      const up = () => { dragging = false; clearTimeout(idle); idle = setTimeout(() => (this.autoRotateHeld = false), 2500); };
      c.addEventListener("mousedown", down);
      c.addEventListener("touchstart", down, {passive: true});
      window.addEventListener("mousemove", move);
      c.addEventListener("touchmove", move, {passive: false});
      window.addEventListener("mouseup", up);
      c.addEventListener("touchend", up);
      c.addEventListener("mouseleave", () => this.onHover && this.onHover(-1));
      c.addEventListener("wheel", e => { this.distance = Math.max(1.4, Math.min(8, this.distance * Math.exp(e.deltaY * 0.001))); e.preventDefault(); }, {passive: false});
    }

    _hover(x, y) {
      let best = -1, bestD = 64;
      for (let i = 0; i < this.n; i++) {
        if (this.highlight && !this.highlight.has(i)) continue;
        const [px, py] = this.project(i), d = (px - x) ** 2 + (py - y) ** 2;
        if (d < bestD) { bestD = d; best = i; }
      }
      if (best !== this._hovered) { this._hovered = best; this.onHover(best, x, y); }
    }

    _resize() {
      const ratio = window.devicePixelRatio || 1, c = this.canvas;
      const w = Math.round(c.clientWidth * ratio), h = Math.round(c.clientHeight * ratio);
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
      return ratio;
    }

    _loop(time) {
      const gl = this.gl, ratio = this._resize();
      if (this.autoRotate && !this.autoRotateHeld) this.yaw += 0.0016;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      if (this.background) gl.clearColor(...this.background, 1); else gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.BLEND);
      if (this.background) gl.blendFunc(gl.SRC_ALPHA, gl.ONE);  // additive glow on an opaque night sky
      else gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);  // true colours over the page
      gl.disable(gl.DEPTH_TEST);
      gl.useProgram(this.program);
      const aspect = this.canvas.width / this.canvas.height, m = this._matrix();
      // fold the aspect ratio into x of the view matrix
      for (const k of [0, 4, 8, 12]) m[k] /= aspect;
      gl.uniformMatrix4fv(this.loc.view, false, m);
      gl.uniform1f(this.loc.scale, this._scale());
      gl.uniform1f(this.loc.pixelRatio, ratio);
      gl.uniform1f(this.loc.dim, this.highlight ? 0.22 : 1.0);
      gl.uniform1f(this.loc.offset, typeof this.offset === "function" ? this.offset() : this.offset);
      for (const [name, size] of [["position", 3], ["color", 3], ["weight", 1]]) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers[name]);
        gl.enableVertexAttribArray(this.loc[name]);
        gl.vertexAttribPointer(this.loc[name], size, gl.FLOAT, false, 0, 0);
      }
      gl.drawArrays(gl.POINTS, 0, this.n + this.markers.length);
      if (this.onFrame) this.onFrame(this);
      requestAnimationFrame(this._loop);
    }
  }

  root.Atlas3D = Atlas3D;
})(typeof window !== "undefined" ? window : this);
