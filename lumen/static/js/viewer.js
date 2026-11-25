// Pan/zoom image viewer with a metric overlay.
//
// Coordinates
// -----------
// "Frame" coordinates are the pixel coordinates of the analysed image on the
// server, pixel-centre convention: integer (u, v) is the centre of a pixel.
// The bitmap shown may be a reduced preview; it is always stretched over the
// full frame rectangle, so overlay geometry (which arrives in frame
// coordinates) lands on the right pixels regardless of preview size.
//
//   view_x = tx + (u + 0.5) * s        view_y = ty + (v + 0.5) * s
//
// where (tx, ty) is the CSS-pixel position of the frame's top-left corner and
// s the CSS pixels per frame pixel.  These three functions are the *only*
// place the mapping is defined; tests/browser/test_pixel_mapping.py checks them.

export function fitTransform(W, H, cw, ch, pad = 0) {
  const s = Math.min((cw - 2 * pad) / W, (ch - 2 * pad) / H);
  return { s, tx: (cw - W * s) / 2, ty: (ch - H * s) / 2 };
}
export function frameToView(t, u, v) {
  return [t.tx + (u + 0.5) * t.s, t.ty + (v + 0.5) * t.s];
}
export function viewToFrame(t, x, y) {
  return [(x - t.tx) / t.s - 0.5, (y - t.ty) / t.s - 0.5];
}

const css = (name, fallback) => {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
};

export class Viewer {
  constructor(host, { label = 'Image viewer', onLine = null, onTap = null } = {}) {
    this.host = host;
    this.canvas = document.createElement('canvas');
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute('role', 'img');
    this.canvas.setAttribute('aria-label', label);
    host.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.W = 0; this.H = 0; this.bitmap = null;
    this.t = { s: 1, tx: 0, ty: 0 };
    this.overlay = null;
    this.tool = 'pan';
    this.onLine = onLine; this.onTap = onTap;
    this.pointers = new Map();
    this.draft = null;            // line being drawn, frame coords
    this.extra = [];              // caller-supplied primitives drawn on top
    this.layers = { grid: true, tags: true, dims: true, candidates: false };
    this.userMoved = false;
    this._raf = 0;
    this._bind();
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(host);
    this._resize();
  }

  destroy() {
    this._ro.disconnect();
    cancelAnimationFrame(this._raf);
    this.canvas.remove();
    if (this.bitmap && this.bitmap.close) { try { this.bitmap.close(); } catch (_) { /* ignore */ } }
    this.bitmap = null;
  }

  // ------------------------------------------------------------ content
  setFrame(bitmap, W, H) {
    const same = this.W === W && this.H === H;
    if (this.bitmap && this.bitmap !== bitmap && this.bitmap.close) { try { this.bitmap.close(); } catch (_) { /* ignore */ } }
    this.bitmap = bitmap; this.W = W; this.H = H;
    if (!same || !this.userMoved) this.fit();
    this.draw();
  }
  setOverlay(overlay) { this.overlay = overlay; this.draw(); }
  setExtra(prims) { this.extra = prims || []; this.draw(); }
  setTool(tool) {
    this.tool = tool; this.draft = null;
    this.canvas.classList.toggle('drawing', tool === 'line');
    this.draw();
  }
  setLayer(name, on) { this.layers[name] = on; this.draw(); }

  // The drawing surface is the canvas's own content box.  (The container has
  // a border; sizing from it shifted every overlay by the border width — a
  // defect found by tests/browser/test_pixel_mapping.py.)
  _box() {
    return { width: this.canvas.clientWidth, height: this.canvas.clientHeight };
  }

  // ----------------------------------------------------------- transform
  toView(u, v) { return frameToView(this.t, u, v); }
  toFrame(x, y) { return viewToFrame(this.t, x, y); }
  fit() {
    const r = this._box();
    if (!this.W || !r.width) return;
    this.t = fitTransform(this.W, this.H, r.width, r.height, 0);
    this.minScale = this.t.s * 0.6;
    this.userMoved = false;
    this.draw();
  }
  zoomAt(x, y, factor) {
    const s0 = this.t.s;
    const s1 = Math.min(Math.max(s0 * factor, this.minScale || s0 * 0.2), 40);
    const k = s1 / s0;
    this.t = { s: s1, tx: x - (x - this.t.tx) * k, ty: y - (y - this.t.ty) * k };
    this.userMoved = true;
    this.draw();
  }
  centerOn(u, v, scaleMul = null) {
    const r = this._box();
    if (!r.width || !r.height) return;      // not laid out yet: keep the fit that the next resize will make
    if (scaleMul) this.t.s = Math.min(this.t.s * scaleMul, 40);
    this.t.tx = r.width / 2 - (u + 0.5) * this.t.s;
    this.t.ty = r.height / 2 - (v + 0.5) * this.t.s;
    this.userMoved = true;
    this.draw();
  }

  _resize() {
    const r = this._box();
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const w = Math.max(1, Math.round(r.width * dpr));
    const hgt = Math.max(1, Math.round(r.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== hgt) {
      this.canvas.width = w; this.canvas.height = hgt;
    }
    this.dpr = dpr;
    if (!this.userMoved) this.fit(); else this.draw();
  }

  // -------------------------------------------------------------- input
  _pos(ev) {
    const r = this.canvas.getBoundingClientRect();
    return [ev.clientX - r.left, ev.clientY - r.top];
  }
  _bind() {
    const c = this.canvas;
    c.addEventListener('pointerdown', (ev) => {
      c.setPointerCapture(ev.pointerId);
      const p = this._pos(ev);
      this.pointers.set(ev.pointerId, { x: p[0], y: p[1], x0: p[0], y0: p[1], t0: performance.now() });
      if (this.pointers.size === 1 && this.tool === 'line') {
        const f = this.toFrame(p[0], p[1]);
        this.draft = { p0: f, p1: f, cursor: p };
      } else if (this.pointers.size === 2) {
        this.draft = null;   // a second finger turns the gesture into a pinch
        this._pinch = this._pinchState();
      }
      this.draw();
    });
    c.addEventListener('pointermove', (ev) => {
      const ptr = this.pointers.get(ev.pointerId);
      if (!ptr) return;
      const p = this._pos(ev);
      const dx = p[0] - ptr.x, dy = p[1] - ptr.y;
      ptr.x = p[0]; ptr.y = p[1];
      if (this.pointers.size === 1) {
        if (this.draft) {
          this.draft.p1 = this.toFrame(p[0], p[1]);
          this.draft.cursor = p;
        } else {
          this.t.tx += dx; this.t.ty += dy; this.userMoved = true;
        }
      } else if (this.pointers.size === 2 && this._pinch) {
        const now = this._pinchState();
        const k = now.d / Math.max(this._pinch.d, 1);
        this.zoomAt(now.cx, now.cy, k);
        this.t.tx += now.cx - this._pinch.cx; this.t.ty += now.cy - this._pinch.cy;
        this._pinch = now;
      }
      this.draw();
    });
    const up = (ev) => {
      const ptr = this.pointers.get(ev.pointerId);
      this.pointers.delete(ev.pointerId);
      if (this.pointers.size < 2) this._pinch = null;
      if (!ptr) return;
      const moved = Math.hypot(ptr.x - ptr.x0, ptr.y - ptr.y0);
      if (this.draft && ev.type === 'pointerup') {
        const d = this.draft; this.draft = null;
        const [x0, y0] = this.toView(d.p0[0], d.p0[1]);
        const [x1, y1] = this.toView(d.p1[0], d.p1[1]);
        if (Math.hypot(x1 - x0, y1 - y0) >= 28 && this.onLine) this.onLine(d.p0, d.p1);
      } else if (ev.type === 'pointerup' && moved < 8 && performance.now() - ptr.t0 < 350) {
        const now = performance.now();
        if (this._lastTap && now - this._lastTap.t < 320 && Math.hypot(ptr.x - this._lastTap.x, ptr.y - this._lastTap.y) < 30) {
          this.zoomAt(ptr.x, ptr.y, 2.2); this._lastTap = null;
        } else {
          this._lastTap = { t: now, x: ptr.x, y: ptr.y };
          if (this.onTap) this.onTap(this.toFrame(ptr.x, ptr.y));
        }
      }
      if (ev.type === 'pointercancel') this.draft = null;
      this.draw();
    };
    c.addEventListener('pointerup', up);
    c.addEventListener('pointercancel', up);
    c.addEventListener('wheel', (ev) => {
      ev.preventDefault();
      const p = this._pos(ev);
      this.zoomAt(p[0], p[1], Math.exp(-ev.deltaY * 0.0015));
    }, { passive: false });
    c.addEventListener('keydown', (ev) => {
      const r = this._box();
      const step = 40;
      if (ev.key === '+' || ev.key === '=') this.zoomAt(r.width / 2, r.height / 2, 1.3);
      else if (ev.key === '-') this.zoomAt(r.width / 2, r.height / 2, 1 / 1.3);
      else if (ev.key === '0') this.fit();
      else if (ev.key === 'ArrowLeft') { this.t.tx += step; this.userMoved = true; this.draw(); }
      else if (ev.key === 'ArrowRight') { this.t.tx -= step; this.userMoved = true; this.draw(); }
      else if (ev.key === 'ArrowUp') { this.t.ty += step; this.userMoved = true; this.draw(); }
      else if (ev.key === 'ArrowDown') { this.t.ty -= step; this.userMoved = true; this.draw(); }
      else return;
      ev.preventDefault();
    });
  }
  _pinchState() {
    const [a, b] = [...this.pointers.values()];
    return { d: Math.hypot(a.x - b.x, a.y - b.y), cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2 };
  }

  // Nearest overlay primitive of a kind to a frame point, within `maxPx` on screen.
  nearest(kind, u, v, maxPx = 26) {
    const [x, y] = this.toView(u, v);
    let best = null, bestD = maxPx;
    for (const p of ((this.overlay && this.overlay.primitives) || [])) {
      if (p.kind !== kind || !p.points || p.points.length < 2) continue;
      for (let i = 1; i < p.points.length; i++) {
        const a = this.toView(p.points[i - 1][0], p.points[i - 1][1]), b = this.toView(p.points[i][0], p.points[i][1]);
        const dx = b[0] - a[0], dy = b[1] - a[1];
        const t = Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / Math.max(dx * dx + dy * dy, 1e-9)));
        const d = Math.hypot(x - a[0] - t * dx, y - a[1] - t * dy);
        if (d < bestD) { bestD = d; best = p; }
      }
    }
    return best;
  }

  // ------------------------------------------------------------ drawing
  draw() {
    cancelAnimationFrame(this._raf);
    this._raf = requestAnimationFrame(() => this._draw());
  }

  _draw() {
    const ctx = this.ctx, dpr = this.dpr || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cw = this.canvas.width / dpr, ch = this.canvas.height / dpr;
    ctx.clearRect(0, 0, cw, ch);
    if (!this.bitmap || !this.W) return;
    ctx.imageSmoothingEnabled = this.t.s < 3;       // show real pixels when zoomed in
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(this.bitmap, this.t.tx, this.t.ty, this.W * this.t.s, this.H * this.t.s);
    const prims = [...((this.overlay && this.overlay.primitives) || []), ...this.extra];
    const order = ['grid_minor', 'grid_major', 'grid_axis', 'tag', 'tag_rejected', 'candidate', 'datum_floor', 'datum_side', 'axis_x', 'axis_z', 'tick', 'origin', 'dim', 'target_preview', 'target', 'target_ref', 'measured', 'proposal', 'segment', 'handle', 'note'];
    // grid density on screen decides what is worth drawing
    const step = (this.overlay && this.overlay.grid_step_mm) || 0;
    const ppm = (this.overlay && this.overlay.px_per_mm) || 0;
    this._gridPx = step * ppm * this.t.s;
    this._labels = [];
    prims.slice().sort((a, b) => order.indexOf(a.kind) - order.indexOf(b.kind)).forEach((p) => this._prim(ctx, p));
    this._flushLabels(ctx);
    if (this.draft) this._draft(ctx, cw, ch);
  }

  _path(ctx, pts, closed) {
    ctx.beginPath();
    pts.forEach((p, i) => {
      const [x, y] = this.toView(p[0], p[1]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    if (closed) ctx.closePath();
  }
  _stroke(ctx, pts, closed, color, width, { dash = null, halo = true, alpha = 1 } = {}) {
    this._path(ctx, pts, closed);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.setLineDash(dash || []);
    if (halo) { ctx.globalAlpha = 0.75 * alpha; ctx.strokeStyle = '#05070a'; ctx.lineWidth = width + 3; ctx.stroke(); }
    ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
    ctx.globalAlpha = 1; ctx.setLineDash([]);
  }
  // Labels are queued while the geometry is drawn and placed afterwards in
  // priority order, so the target and its dimensions always win a spot and
  // low-priority labels (tag IDs, grid ticks) are dropped when they collide.
  _label(ctx, text, x, y, color, { align = 'left', size = 13, bold = true, avoid = false, prio = 5 } = {}) {
    if (!text) return;
    this._labels.push({ text, x, y, color, align, size, bold, avoid, prio });
  }
  // Screen areas covered by controls that sit on top of the canvas.
  _reserved() {
    const base = this.canvas.getBoundingClientRect();
    const out = [];
    this.host.querySelectorAll('.viewer-hud > *, .viewer-tools, .viewer-watermark, .live-coach > *').forEach((el) => {
      const b = el.getBoundingClientRect();
      if (b.width > 0 && b.height > 0) out.push({ x: b.left - base.left, y: b.top - base.top, w: b.width, h: b.height });
    });
    return out;
  }
  _flushLabels(ctx) {
    const r = this._box();
    const placed = [];
    const reserved = this._reserved();
    const hit = (q, bx, by, w, hgt) => bx < q.x + q.w + 3 && bx + w + 3 > q.x && by < q.y + q.h + 3 && by + hgt + 3 > q.y;
    const free = (bx, by, w, hgt) => !placed.some((q) => hit(q, bx, by, w, hgt)) && !reserved.some((q) => hit(q, bx, by, w, hgt));
    this._labels.sort((a, b) => b.prio - a.prio);
    for (const L of this._labels) {
      ctx.font = `${L.bold ? 700 : 500} ${L.size}px system-ui, -apple-system, sans-serif`;
      const w = ctx.measureText(L.text).width + 10, hgt = L.size + 8;
      const clampX = (v) => Math.min(Math.max(v, 2), r.width - w - 2);
      const clampY = (v) => Math.min(Math.max(v, 2), r.height - hgt - 2);
      const bx0 = clampX(L.align === 'center' ? L.x - w / 2 : L.align === 'right' ? L.x - w : L.x);
      const by0 = clampY(L.y - hgt / 2);
      let spot = null;
      if (free(bx0, by0, w, hgt)) spot = [bx0, by0];
      else if (!L.avoid) {
        const sy = hgt + 4, sx = w * 0.6 + 6;
        const tries = [[0, sy], [0, -sy], [sx, 0], [-sx, 0], [sx, sy], [-sx, sy], [sx, -sy], [-sx, -sy], [0, 2 * sy], [0, -2 * sy], [2 * sx, 0], [-2 * sx, 0], [0, 3 * sy], [0, -3 * sy], [-2 * sx, -sy], [-2 * sx, sy], [-3 * sx, 0], [0, 4 * sy], [0, -4 * sy]];
        for (const [dx, dy] of tries) {
          const bx = clampX(bx0 + dx), by = clampY(by0 + dy);
          if (free(bx, by, w, hgt)) { spot = [bx, by]; break; }
        }
        if (!spot) spot = [bx0, by0];        // nothing free: important labels are still shown
      }
      if (!spot) continue;
      const [bx, by] = spot;
      placed.push({ x: bx, y: by, w, h: hgt });
      ctx.fillStyle = 'rgba(5,7,10,0.84)';
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(bx, by, w, hgt, 5); else ctx.rect(bx, by, w, hgt);
      ctx.fill();
      ctx.fillStyle = L.color; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      ctx.fillText(L.text, bx + 5, by + hgt / 2 + 0.5);
    }
    this._placedLabels = placed.length;
  }
  // Point of a polyline at fraction f (0..1) of its drawn length, in view px.
  _along(pts, f) {
    const v = pts.map((p) => this.toView(p[0], p[1]));
    let total = 0;
    const seg = [];
    for (let i = 1; i < v.length; i++) { const d = Math.hypot(v[i][0] - v[i - 1][0], v[i][1] - v[i - 1][1]); seg.push(d); total += d; }
    let want = f * total;
    for (let i = 0; i < seg.length; i++) {
      if (want <= seg[i] || i === seg.length - 1) {
        const k = seg[i] > 0 ? Math.min(want / seg[i], 1) : 0;
        return [v[i][0] + (v[i + 1][0] - v[i][0]) * k, v[i][1] + (v[i + 1][1] - v[i][1]) * k];
      }
      want -= seg[i];
    }
    return v[0];
  }
  _arrowHead(ctx, a, b, color) {
    const [x0, y0] = this.toView(a[0], a[1]);
    const [x1, y1] = this.toView(b[0], b[1]);
    const ang = Math.atan2(y1 - y0, x1 - x0), L = 13;
    ctx.fillStyle = color; ctx.strokeStyle = '#05070a'; ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - L * Math.cos(ang - 0.42), y1 - L * Math.sin(ang - 0.42));
    ctx.lineTo(x1 - L * Math.cos(ang + 0.42), y1 - L * Math.sin(ang + 0.42));
    ctx.closePath(); ctx.fill(); ctx.stroke();
  }

  _prim(ctx, p) {
    const pts = p.points || [];
    const k = p.kind;
    const C = {
      grid: css('--ov-grid', '#5cc8ff'), floor: css('--ov-floor', '#ff9f1a'), side: css('--ov-side', '#b388ff'),
      tag: css('--ov-tag', '#3ddc97'), bad: css('--ov-tag-bad', '#ff5d5d'), target: css('--ov-target', '#ff4fd8'),
      preview: css('--ov-target-preview', '#ffb020'), ax: css('--ov-axis-x', '#ff7a7a'), az: css('--ov-axis-z', '#6fb4ff'),
    };
    if (k.startsWith('grid')) {
      if (!this.layers.grid || pts.length < 2) return;
      const major = k !== 'grid_minor';
      if (!major && this._gridPx > 0 && this._gridPx < 7) return;     // too dense to help
      this._stroke(ctx, pts, false, C.grid, k === 'grid_axis' ? 2.2 : major ? 1.4 : 0.8, { halo: k === 'grid_axis', alpha: k === 'grid_axis' ? 0.95 : major ? 0.75 : 0.42 });
    } else if (k === 'tick') {
      if (!this.layers.grid || pts.length < 2) return;
      if (this._gridPx > 0 && this._gridPx * 5 < 38) return;          // labels would collide
      const a = this.toView(pts[0][0], pts[0][1]), b = this.toView(pts[1][0], pts[1][1]);
      const L = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
      const ox = ((b[0] - a[0]) / L) * 15, oy = ((b[1] - a[1]) / L) * 15;
      this._label(ctx, p.label, a[0] + ox, a[1] + oy, C.grid, { size: 11, align: 'center', avoid: true, prio: 3 });
    } else if (k === 'tag' || k === 'tag_rejected') {
      if (!this.layers.tags || pts.length < 3) return;
      const col = k === 'tag' ? C.tag : C.bad;
      this._stroke(ctx, pts, true, col, 2, { dash: k === 'tag' ? null : [6, 5] });
      const [x, y] = this.toView(pts[0][0], pts[0][1]);
      this._label(ctx, p.label, x, y - 12, col, { size: 12, avoid: true, prio: 2 });
    } else if (k === 'datum_floor' || k === 'datum_side') {
      if (pts.length < 2) return;
      const col = k === 'datum_floor' ? C.floor : C.side;
      this._stroke(ctx, pts, false, col, 3);
      // label toward the far end of the visible stretch, clear of the origin
      const [x, y] = this._along(pts, 0.78);
      this._label(ctx, p.label, x, y - 16, col, { align: 'center', prio: 6 });
    } else if (k === 'axis_x' || k === 'axis_z') {
      if (pts.length < 2) return;
      const col = k === 'axis_x' ? C.ax : C.az;
      this._stroke(ctx, pts, false, col, 3.5);
      this._arrowHead(ctx, pts[pts.length - 2], pts[pts.length - 1], col);
      const e = pts[pts.length - 1];
      const [x, y] = this.toView(e[0], e[1]);
      this._label(ctx, p.label, x + 8, y, col, { size: 14, prio: 7 });
    } else if (k === 'origin') {
      const [x, y] = this.toView(pts[0][0], pts[0][1]);
      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#fff'; ctx.fill(); ctx.lineWidth = 2.5; ctx.strokeStyle = '#05070a'; ctx.stroke();
    } else if (k === 'dim') {
      if (!this.layers.dims || pts.length < 2) return;
      this._stroke(ctx, pts, false, '#ffffff', 1.4, { dash: [7, 5], alpha: 0.95 });
      // the leader runs from the datum to the target: label it near the datum end
      const [x, y] = this._along(pts, 0.3);
      this._label(ctx, p.label, x, y, '#ffffff', { align: 'center', prio: 8 });
    } else if (k === 'target' || k === 'target_preview' || k === 'measured' || k === 'proposal') {
      if (pts.length < 3) return;
      const col = k === 'target' ? C.target : k === 'target_preview' ? C.preview : k === 'measured' ? '#3ddc97' : '#ffffff';
      this._path(ctx, pts, true);
      ctx.fillStyle = col; ctx.globalAlpha = k === 'target' ? 0.16 : 0.10; ctx.fill(); ctx.globalAlpha = 1;
      this._stroke(ctx, pts, true, col, p.active === false ? 2 : 3.2, { dash: k === 'target_preview' || k === 'proposal' ? [10, 6] : null });
      (p.corners || []).forEach((c) => {
        const [x, y] = this.toView(c[0], c[1]);
        ctx.beginPath(); ctx.arc(x, y, 4.5, 0, Math.PI * 2); ctx.fillStyle = col; ctx.fill();
        ctx.lineWidth = 1.5; ctx.strokeStyle = '#05070a'; ctx.stroke();
      });
      // label above the outline as it appears on screen (whatever its rotation)
      const vv = (p.corners || pts).map((c) => this.toView(c[0], c[1]));
      const minY = Math.min(...vv.map((c) => c[1])), midX = vv.reduce((a, c) => a + c[0], 0) / vv.length;
      const tagText = k === 'target_preview' && p.active !== false ? `${p.label} · PREVIEW` : p.label;
      this._label(ctx, tagText, midX, minY - 17, col, { size: 13, align: 'center', prio: p.active === false ? 4 : 9, avoid: p.active === false });
    } else if (k === 'target_ref') {
      const [x, y] = this.toView(pts[0][0], pts[0][1]);
      ctx.strokeStyle = '#05070a'; ctx.lineWidth = 4;
      ctx.beginPath(); ctx.moveTo(x - 9, y); ctx.lineTo(x + 9, y); ctx.moveTo(x, y - 9); ctx.lineTo(x, y + 9); ctx.stroke();
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.stroke();
    } else if (k === 'handle') {
      const [x, y] = this.toView(pts[0][0], pts[0][1]);
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.fillStyle = p.color || '#ffffff'; ctx.fill(); ctx.lineWidth = 2.5; ctx.strokeStyle = '#05070a'; ctx.stroke();
      this._label(ctx, p.label, x + 14, y, p.color || '#ffffff', { size: 12 });
    } else if (k === 'note') {
      const [x, y] = this.toView(pts[0][0], pts[0][1]);
      this._label(ctx, p.label, x, y, p.color || '#ffffff', { align: 'center' });
    } else if (k === 'segment') {
      this._stroke(ctx, pts, false, p.color || '#ffffff', p.width || 2.5, { dash: p.dash || null });
    } else if (k === 'candidate') {
      if (!this.layers.candidates || pts.length < 2) return;
      const col = p.classification === 'on_plane' ? '#3ddc97' : p.classification === 'off_plane' ? '#ff5d5d' : p.classification === 'view_dependent' ? '#ffb020' : '#ffffff';
      this._stroke(ctx, pts, false, col, 3, { dash: [10, 7] });
      const [x, y] = this._along(pts, 0.5);
      const txt = p.classification === 'on_plane' ? 'on the face' : p.classification === 'off_plane' ? `off the face ${Math.abs(p.height_mm || 0).toFixed(0)} mm` : p.classification === 'view_dependent' ? 'silhouette / shadow' : 'not verified';
      this._label(ctx, txt, x, y - 14, col, { size: 12, align: 'center', prio: 5, avoid: true });
    }
  }

  _draft(ctx, cw, ch) {
    const d = this.draft;
    this._stroke(ctx, [d.p0, d.p1], false, '#ffffff', 2.5, { dash: [8, 6] });
    // loupe: the finger hides the point being placed, so show it magnified
    const [cx, cy] = d.cursor;
    const R = 62, mag = 3;
    const lx = cx < cw / 2 ? cw - R - 14 : R + 14;
    const ly = R + 14;
    ctx.save();
    ctx.beginPath(); ctx.arc(lx, ly, R, 0, Math.PI * 2); ctx.clip();
    ctx.fillStyle = '#05070a'; ctx.fillRect(lx - R, ly - R, 2 * R, 2 * R);
    const s = this.t.s * mag;
    const [u, v] = d.p1;
    ctx.imageSmoothingEnabled = s < 3;
    ctx.drawImage(this.bitmap, lx - (u + 0.5) * s, ly - (v + 0.5) * s, this.W * s, this.H * s);
    ctx.restore();
    ctx.beginPath(); ctx.arc(lx, ly, R, 0, Math.PI * 2);
    ctx.lineWidth = 3; ctx.strokeStyle = '#ffc83d'; ctx.stroke();
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(lx - 14, ly); ctx.lineTo(lx + 14, ly); ctx.moveTo(lx, ly - 14); ctx.lineTo(lx, ly + 14); ctx.stroke();
  }
}
