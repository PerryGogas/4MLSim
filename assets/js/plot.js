/* =====================================================================
   4MLSim plotting library (SVG, no dependencies)
   Real data coordinates, nice ticks, tooltips, crosshair, click/drag,
   heatmaps, contours, legends, PNG / SVG / XLSX export, table view.
   (c) 2026 Periklis Gogas, EMEF Lab. MIT License.
   ===================================================================== */
(function () {
  'use strict';
  const MLS = window.MLS;
  const NS = 'http://www.w3.org/2000/svg';

  MLS.color = name => MLS.css(name.startsWith('--') ? name : '--' + name);
  MLS.seriesColor = i => MLS.css('--series-' + ((i % 8) + 1));
  const SHAPES = ['circle', 'square', 'triangle', 'diamond', 'cross', 'star'];
  MLS.shapeFor = i => SHAPES[i % SHAPES.length];

  /* ---------- nice ticks ---------- */
  function niceStep(span, target) {
    const raw = span / Math.max(1, target);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const m = raw / p;
    return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
  }
  function ticks(min, max, target) {
    if (!(max > min)) return [min];
    const st = niceStep(max - min, target);
    const out = [];
    const start = Math.ceil(min / st - 1e-9) * st;
    for (let v = start; v <= max + st * 1e-9; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : v);
    return out;
  }
  function tickFmt(v, step) {
    const d = step >= 1 ? 0 : Math.min(6, Math.ceil(-Math.log10(step) - 1e-9));
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e6 || a < 1e-4)) return v.toExponential(1).replace('-', '−');
    const s = v.toFixed(d);
    return s.startsWith('-') ? '−' + s.slice(1) : s;
  }
  MLS.niceTicks = ticks;
  MLS.niceDomain = function (min, max, pad) {
    if (!(max > min)) { const c = Number.isFinite(min) ? min : 0; return [c - 1, c + 1]; }
    const p = (max - min) * (pad === undefined ? 0.06 : pad);
    return [min - p, max + p];
  };

  /* ---------- shapes ---------- */
  function shapePath(shape, cx, cy, r) {
    switch (shape) {
      case 'square': { const s = r * 0.9; return 'M' + (cx - s) + ',' + (cy - s) + 'h' + 2 * s + 'v' + 2 * s + 'h' + -2 * s + 'z'; }
      case 'triangle': { const h = r * 1.25; return 'M' + cx + ',' + (cy - h) + 'L' + (cx + h * 0.95) + ',' + (cy + h * 0.7) + 'L' + (cx - h * 0.95) + ',' + (cy + h * 0.7) + 'z'; }
      case 'diamond': { const h = r * 1.25; return 'M' + cx + ',' + (cy - h) + 'L' + (cx + h) + ',' + cy + 'L' + cx + ',' + (cy + h) + 'L' + (cx - h) + ',' + cy + 'z'; }
      case 'cross': { const a = r * 0.45, b = r * 1.15; return 'M' + (cx - a) + ',' + (cy - b) + 'h' + 2 * a + 'v' + (b - a) + 'h' + (b - a) + 'v' + 2 * a + 'h' + -(b - a) + 'v' + (b - a) + 'h' + -2 * a + 'v' + -(b - a) + 'h' + -(b - a) + 'v' + -2 * a + 'h' + (b - a) + 'z'; }
      case 'star': { let d = ''; for (let k = 0; k < 10; k++) { const rr = k % 2 ? r * 0.55 : r * 1.3, ang = -Math.PI / 2 + k * Math.PI / 5; d += (k ? 'L' : 'M') + (cx + rr * Math.cos(ang)).toFixed(2) + ',' + (cy + rr * Math.sin(ang)).toFixed(2); } return d + 'z'; }
      default: return null;
    }
  }
  MLS.shapeSVG = function (shape, color, size, stroke) {
    const s = size || 14, c = s / 2, r = s * 0.33;
    const sp = shapePath(shape, c, c, r);
    const st = stroke ? ' stroke="' + stroke + '" stroke-width="1.5"' : '';
    return '<svg width="' + s + '" height="' + s + '" viewBox="0 0 ' + s + ' ' + s + '" aria-hidden="true">' +
      (sp ? '<path d="' + sp + '" fill="' + color + '"' + st + '/>' : '<circle cx="' + c + '" cy="' + c + '" r="' + r * 1.15 + '" fill="' + color + '"' + st + '/>') + '</svg>';
  };

  /* ---------- sequential colour scale from theme tokens ---------- */
  MLS.seqScale = function () {
    const stops = [0, 1, 2, 3, 4, 5, 6, 7].map(i => hexToRgb(MLS.css('--seq-' + i)));
    return function (t) {
      t = Math.max(0, Math.min(1, t)) * (stops.length - 1);
      const i = Math.min(stops.length - 2, Math.floor(t)), f = t - i, a = stops[i], b = stops[i + 1];
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
    };
  };
  function hexToRgb(h) {
    h = (h || '#000').replace('#', '');
    if (h.length === 3) h = h.split('').map(c => c + c).join('');
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  MLS.hexToRgb = hexToRgb;
  MLS.rgba = (hex, a) => { const c = hexToRgb(hex); return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')'; };

  /* loss-surface colour scale: the same in light and dark mode,
     light at the minimum (t = 0) and dark where the loss is high (t = 1) */
  MLS.lossScale = function () {
    const stops = ['#F4F8FE', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'].map(hexToRgb);
    return function (t) {
      t = Math.max(0, Math.min(1, t)) * (stops.length - 1);
      const i = Math.min(stops.length - 2, Math.floor(t)), f = t - i, a = stops[i], b = stops[i + 1];
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
    };
  };

  /* grid: values[j][i] (j = row along y, i = column along x) -> data URL image */
  MLS.gridImage = function (values, colorOf) {
    const ny = values.length, nx = values[0].length;
    const cv = document.createElement('canvas'); cv.width = nx; cv.height = ny;
    const ctx = cv.getContext('2d'); const img = ctx.createImageData(nx, ny);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const c = colorOf(values[j][i], i, j); const k = ((ny - 1 - j) * nx + i) * 4;
      img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = c[3] === undefined ? 255 : c[3];
    }
    ctx.putImageData(img, 0, 0);
    return cv.toDataURL('image/png');
  };

  /* marching squares: returns segments [[x1,y1,x2,y2],...] in data units */
  MLS.contourSegments = function (values, xs, ys, level) {
    const segs = [], ny = ys.length, nx = xs.length;
    const lerp = (a, b, va, vb) => a + (b - a) * ((level - va) / (vb - va));
    for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
      const v0 = values[j][i], v1 = values[j][i + 1], v2 = values[j + 1][i + 1], v3 = values[j + 1][i];
      let idx = 0; if (v0 > level) idx |= 1; if (v1 > level) idx |= 2; if (v2 > level) idx |= 4; if (v3 > level) idx |= 8;
      if (idx === 0 || idx === 15) continue;
      const x0 = xs[i], x1 = xs[i + 1], y0 = ys[j], y1 = ys[j + 1];
      const e = {
        b: [lerp(x0, x1, v0, v1), y0], r: [x1, lerp(y0, y1, v1, v2)],
        t: [lerp(x0, x1, v3, v2), y1], l: [x0, lerp(y0, y1, v0, v3)]
      };
      const add = (p, q) => segs.push([e[p][0], e[p][1], e[q][0], e[q][1]]);
      switch (idx) {
        case 1: case 14: add('l', 'b'); break;
        case 2: case 13: add('b', 'r'); break;
        case 3: case 12: add('l', 'r'); break;
        case 4: case 11: add('r', 't'); break;
        case 5: add('l', 't'); add('b', 'r'); break;
        case 6: case 9: add('b', 't'); break;
        case 7: case 8: add('l', 't'); break;
        case 10: add('l', 'b'); add('r', 't'); break;
      }
    }
    return segs;
  };

  /* =================================================================
     plot object
     ================================================================= */
  MLS.plot = function (host, opts) {
    opts = opts || {};
    host.classList.add('plot');
    if (opts.onClick || opts.onPointerDown) host.classList.add('clickable');
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('role', 'img');
    if (opts.ariaLabel) svg.setAttribute('aria-label', opts.ariaLabel);
    host.appendChild(svg);
    const P = {
      host, svg, opts,
      m: Object.assign({ l: 60, r: 16, t: 14, b: 48 }, opts.margin || {}),
      xd: (opts.x && opts.x.domain) || [0, 1], yd: (opts.y && opts.y.domain) || [0, 1],
      W: 600, H: 380, tips: [], drawFn: null, hover: null
    };
    P.setX = (a, b) => { P.xd = [a, b]; return P; };
    P.setY = (a, b) => { P.yd = [a, b]; return P; };
    P.sx = v => P.m.l + (v - P.xd[0]) / (P.xd[1] - P.xd[0]) * (P.W - P.m.l - P.m.r);
    P.sy = v => P.H - P.m.b - (v - P.yd[0]) / (P.yd[1] - P.yd[0]) * (P.H - P.m.t - P.m.b);
    P.ix = px => P.xd[0] + (px - P.m.l) / (P.W - P.m.l - P.m.r) * (P.xd[1] - P.xd[0]);
    P.iy = py => P.yd[0] + (P.H - P.m.b - py) / (P.H - P.m.t - P.m.b) * (P.yd[1] - P.yd[0]);
    P.inside = (px, py) => px >= P.m.l && px <= P.W - P.m.r && py >= P.m.t && py <= P.H - P.m.b;

    function node(tag, attrs, parent) {
      const e = document.createElementNS(NS, tag);
      for (const k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
      (parent || P.layer).appendChild(e);
      return e;
    }
    P.node = node;

    /* ---------- main draw ---------- */
    P.draw = function (fn) { if (fn) P.drawFn = fn; P.redraw(); };
    P.redraw = function () {
      const w = Math.max(260, Math.round(host.clientWidth || 600));
      P.W = w;
      P.H = opts.height ? (typeof opts.height === 'function' ? opts.height(w) : opts.height) : Math.round(Math.max(240, Math.min(520, w * (opts.aspect || 0.62))));
      if (opts.square) P.H = Math.round((w - P.m.l - P.m.r) * (opts.square === true ? 1 : opts.square) + P.m.t + P.m.b);
      svg.setAttribute('viewBox', '0 0 ' + P.W + ' ' + P.H);
      svg.setAttribute('width', P.W); svg.setAttribute('height', P.H);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      P.tips = [];
      const C = {
        surface: MLS.css('--chart-surface'), grid: MLS.css('--chart-grid'), axis: MLS.css('--chart-axis'),
        text: MLS.css('--chart-text'), muted: MLS.css('--chart-muted'), ink: MLS.css('--chart-ink')
      };
      P.C = C;
      const defs = node('defs', {}, svg);
      const cid = 'clip' + Math.random().toString(36).slice(2, 8);
      const cp = node('clipPath', { id: cid }, defs);
      node('rect', { x: P.m.l, y: P.m.t, width: P.W - P.m.l - P.m.r, height: P.H - P.m.t - P.m.b }, cp);
      node('rect', { x: 0, y: 0, width: P.W, height: P.H, fill: C.surface, 'data-bg': '1' }, svg);
      /* grid + ticks */
      const gAxis = node('g', { 'font-size': 12, fill: C.text, 'font-family': 'IBM Plex Sans, sans-serif' }, svg);
      const xt = opts.x && opts.x.ticks === false ? [] : opts.x && typeof opts.x.ticks === 'function' ? opts.x.ticks() : ticks(P.xd[0], P.xd[1], Math.max(3, Math.round((P.W - P.m.l - P.m.r) / 85)));
      const yt = opts.y && opts.y.ticks === false ? [] : ticks(P.yd[0], P.yd[1], Math.max(3, Math.round((P.H - P.m.t - P.m.b) / 55)));
      const xs = xt.length > 1 ? xt[1] - xt[0] : 1, ys = yt.length > 1 ? yt[1] - yt[0] : 1;
      const pick = (o, st) => v => { const r = o && o.fmt ? o.fmt(v) : undefined; return r === undefined || r === null ? tickFmt(v, st) : r; };
      const xf = pick(opts.x, xs), yf = pick(opts.y, ys);
      xt.forEach(v => {
        const x = P.sx(v);
        node('line', { x1: x, x2: x, y1: P.m.t, y2: P.H - P.m.b, stroke: C.grid, 'stroke-width': 1 }, gAxis);
        node('text', { x, y: P.H - P.m.b + 17, 'text-anchor': 'middle' }, gAxis).textContent = xf(v);
      });
      yt.forEach(v => {
        const y = P.sy(v);
        node('line', { x1: P.m.l, x2: P.W - P.m.r, y1: y, y2: y, stroke: C.grid, 'stroke-width': 1 }, gAxis);
        node('text', { x: P.m.l - 7, y: y + 4, 'text-anchor': 'end' }, gAxis).textContent = yf(v);
      });
      node('line', { x1: P.m.l, x2: P.W - P.m.r, y1: P.H - P.m.b, y2: P.H - P.m.b, stroke: C.axis, 'stroke-width': 1 }, gAxis);
      node('line', { x1: P.m.l, x2: P.m.l, y1: P.m.t, y2: P.H - P.m.b, stroke: C.axis, 'stroke-width': 1 }, gAxis);
      if (opts.x && opts.x.label) node('text', { x: P.m.l + (P.W - P.m.l - P.m.r) / 2, y: P.H - 8, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 600, fill: C.text }, gAxis).textContent = typeof opts.x.label === 'function' ? opts.x.label() : opts.x.label;
      if (opts.y && opts.y.label) { const t = node('text', { x: 0, y: 0, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 600, fill: C.text, transform: 'translate(' + 14 + ',' + (P.m.t + (P.H - P.m.t - P.m.b) / 2) + ') rotate(-90)' }, gAxis); t.textContent = typeof opts.y.label === 'function' ? opts.y.label() : opts.y.label; }
      /* data layer (clipped) + overlay layer (not clipped) */
      P.layer = node('g', { 'clip-path': 'url(#' + cid + ')' }, svg);
      P.dataLayer = P.layer;
      P.overlay = node('g', {}, svg);
      if (P.drawFn) P.drawFn(P);
      P.hoverLayer = node('g', { 'pointer-events': 'none' }, svg);
    };

    /* ---------- primitives (data coordinates) ---------- */
    P.points = function (data, o) {
      o = o || {};
      const g = node('g', {});
      const r = o.r || 5.5;
      const ring = o.ring === false ? null : (o.ringColor || P.C.surface);
      data.forEach((d, i) => {
        const x = o.x ? o.x(d, i) : d.x !== undefined ? d.x : d[0];
        const y = o.y ? o.y(d, i) : d.y !== undefined ? d.y : d[1];
        if (!Number.isFinite(x) || !Number.isFinite(y)) return;
        const cx = P.sx(x), cy = P.sy(y);
        const color = typeof o.color === 'function' ? o.color(d, i) : (o.color || P.C.ink);
        const shape = typeof o.shape === 'function' ? o.shape(d, i) : (o.shape || 'circle');
        const rr = typeof o.r === 'function' ? o.r(d, i) : r;
        const sp = shapePath(shape, cx, cy, rr);
        const attrs = { fill: o.hollow ? 'none' : color, stroke: o.hollow ? color : ring, 'stroke-width': o.hollow ? 2 : (ring ? 1.5 : 0), opacity: o.opacity };
        const e = sp ? node('path', Object.assign({ d: sp }, attrs), g) : node('circle', Object.assign({ cx, cy, r: rr }, attrs), g);
        if (o.tip) { e.setAttribute('data-tip', P.tips.length); P.tips.push(() => o.tip(d, i)); }
        if (o.highlight && o.highlight(d, i)) node('circle', { cx, cy, r: rr + 5, fill: 'none', stroke: P.C.ink, 'stroke-width': 1.6 }, g);
      });
      return g;
    };
    P.line = function (pts, o) {
      o = o || {};
      let d = '', pen = false;
      pts.forEach(p => {
        const x = p.x !== undefined ? p.x : p[0], y = p.y !== undefined ? p.y : p[1];
        if (!Number.isFinite(x) || !Number.isFinite(y)) { pen = false; return; }
        const X = P.sx(x), Y = Math.max(-1e5, Math.min(1e5, P.sy(y)));
        d += (pen ? 'L' : 'M') + X.toFixed(2) + ',' + Y.toFixed(2); pen = true;
      });
      return node('path', { d, fill: 'none', stroke: o.color || P.C.ink, 'stroke-width': o.width || 2, 'stroke-dasharray': o.dash, opacity: o.opacity, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, o.parent);
    };
    P.fn = function (f, o) {
      o = o || {};
      const n = o.samples || 200, pts = [];
      for (let k = 0; k <= n; k++) { const x = P.xd[0] + (P.xd[1] - P.xd[0]) * k / n; pts.push([x, f(x)]); }
      return P.line(pts, o);
    };
    P.segment = (x1, y1, x2, y2, o) => { o = o || {}; return node('line', { x1: P.sx(x1), y1: P.sy(y1), x2: P.sx(x2), y2: P.sy(y2), stroke: o.color || P.C.muted, 'stroke-width': o.width || 1.2, 'stroke-dasharray': o.dash, opacity: o.opacity }, o.parent); };
    P.vline = (x, o) => P.segment(x, P.yd[0], x, P.yd[1], o);
    P.hline = (y, o) => P.segment(P.xd[0], y, P.xd[1], y, o);
    P.rect = (x0, y0, x1, y1, o) => { o = o || {}; const a = P.sx(Math.min(x0, x1)), b = P.sy(Math.max(y0, y1)); return node('rect', { x: a, y: b, width: Math.abs(P.sx(x1) - P.sx(x0)), height: Math.abs(P.sy(y1) - P.sy(y0)), fill: o.fill || 'none', stroke: o.stroke, 'stroke-width': o.width, opacity: o.opacity, rx: o.rx }, o.parent); };
    P.text = (x, y, s, o) => { o = o || {}; const t = node('text', { x: P.sx(x) + (o.dx || 0), y: P.sy(y) + (o.dy || 0), 'text-anchor': o.anchor || 'start', 'font-size': o.size || 12.5, 'font-weight': o.weight || 600, fill: o.color || P.C.text, 'paint-order': 'stroke', stroke: o.halo === false ? null : P.C.surface, 'stroke-width': o.halo === false ? null : 3.5, 'stroke-linejoin': 'round' }, o.parent || P.overlay); t.textContent = s; return t; };
    P.image = (url, x0, x1, y0, y1, o) => { o = o || {}; return node('image', { href: url, x: P.sx(x0), y: P.sy(y1), width: P.sx(x1) - P.sx(x0), height: P.sy(y0) - P.sy(y1), preserveAspectRatio: 'none', opacity: o.opacity, style: 'image-rendering:' + (o.smooth ? 'auto' : 'pixelated') }); };
    P.contour = function (values, xs, ys, levels, o) {
      o = o || {};
      const g = node('g', { fill: 'none', stroke: o.color || P.C.muted, 'stroke-width': o.width || 1, opacity: o.opacity === undefined ? 0.7 : o.opacity });
      levels.forEach(lv => {
        const segs = MLS.contourSegments(values, xs, ys, lv);
        let d = '';
        segs.forEach(s => { d += 'M' + P.sx(s[0]).toFixed(1) + ',' + P.sy(s[1]).toFixed(1) + 'L' + P.sx(s[2]).toFixed(1) + ',' + P.sy(s[3]).toFixed(1); });
        if (d) node('path', { d, 'stroke-dasharray': o.dash }, g);
      });
      return g;
    };
    P.arrow = function (x1, y1, x2, y2, o) {
      o = o || {};
      const X1 = P.sx(x1), Y1 = P.sy(y1), X2 = P.sx(x2), Y2 = P.sy(y2);
      const g = node('g', {}, o.parent);
      node('line', { x1: X1, y1: Y1, x2: X2, y2: Y2, stroke: o.color || P.C.ink, 'stroke-width': o.width || 1.8 }, g);
      const a = Math.atan2(Y2 - Y1, X2 - X1), h = o.head || 7;
      node('path', { d: 'M' + X2 + ',' + Y2 + 'L' + (X2 - h * Math.cos(a - 0.45)) + ',' + (Y2 - h * Math.sin(a - 0.45)) + 'L' + (X2 - h * Math.cos(a + 0.45)) + ',' + (Y2 - h * Math.sin(a + 0.45)) + 'z', fill: o.color || P.C.ink }, g);
      return g;
    };
    P.note = function (s, o) {
      o = o || {};
      const t = node('text', { x: P.m.l + (P.W - P.m.l - P.m.r) / 2, y: P.m.t + (P.H - P.m.t - P.m.b) / 2, 'text-anchor': 'middle', 'font-size': o.size || 14, 'font-weight': 600, fill: P.C.muted }, P.overlay);
      t.textContent = s; return t;
    };

    /* ---------- crosshair for line charts ----------
       series: [{ name, color, pts:[[x,y],...] }] ; fmtX/fmtY optional */
    P.crosshair = function (series, fmtX, fmtY) { P.hover = { series, fmtX: fmtX || (v => MLS.fmt(v)), fmtY: fmtY || (v => MLS.fmt(v)) }; };

    /* ---------- pointer handling ---------- */
    function evPos(e) {
      const r = svg.getBoundingClientRect();
      const px = (e.clientX - r.left) * (P.W / r.width), py = (e.clientY - r.top) * (P.H / r.height);
      return { px, py, x: P.ix(px), y: P.iy(py) };
    }
    P.evPos = evPos;
    let dragging = false;
    svg.addEventListener('pointerdown', e => {
      const p = evPos(e);
      if (!P.inside(p.px, p.py)) return;
      if (opts.onPointerDown && opts.onPointerDown(p.x, p.y, e, p) === true) { dragging = true; svg.setPointerCapture(e.pointerId); e.preventDefault(); }
    });
    svg.addEventListener('pointermove', e => {
      const p = evPos(e);
      if (dragging && opts.onPointerMove) { opts.onPointerMove(p.x, p.y, e, p); return; }
      const t = e.target.getAttribute && e.target.getAttribute('data-tip');
      if (t !== null && t !== undefined && P.tips[t]) { MLS.tip.show(P.tips[t](), e.clientX, e.clientY); return; }
      if (P.hover && P.inside(p.px, p.py)) { showCross(p, e); return; }
      MLS.tip.hide(); clearCross();
    });
    svg.addEventListener('pointerup', e => {
      if (dragging) { dragging = false; try { svg.releasePointerCapture(e.pointerId); } catch (x) { /* ignore */ } if (opts.onPointerUp) { const p = evPos(e); opts.onPointerUp(p.x, p.y, e, p); } return; }
      if (opts.onClick) { const p = evPos(e); if (P.inside(p.px, p.py)) opts.onClick(p.x, p.y, e, p); }
    });
    svg.addEventListener('pointerleave', () => { MLS.tip.hide(); clearCross(); });
    function clearCross() { if (P.hoverLayer) while (P.hoverLayer.firstChild) P.hoverLayer.removeChild(P.hoverLayer.firstChild); }
    function showCross(p, e) {
      clearCross();
      const H = P.hover; let best = null;
      H.series.forEach(s => {
        let bi = -1, bd = Infinity;
        s.pts.forEach((q, i) => { const d = Math.abs(q[0] - p.x); if (d < bd && Number.isFinite(q[1])) { bd = d; bi = i; } });
        if (bi >= 0) { if (!best) best = { x: s.pts[bi][0], rows: [] }; best.rows.push({ s, y: s.pts[bi][1] }); }
      });
      if (!best) return;
      const X = P.sx(best.x);
      node('line', { x1: X, x2: X, y1: P.m.t, y2: P.H - P.m.b, stroke: P.C.axis, 'stroke-width': 1, 'stroke-dasharray': '3 3' }, P.hoverLayer);
      best.rows.forEach(r => { const Y = P.sy(r.y); if (Y >= P.m.t && Y <= P.H - P.m.b) node('circle', { cx: X, cy: Y, r: 4.5, fill: r.s.color, stroke: P.C.surface, 'stroke-width': 2 }, P.hoverLayer); });
      MLS.tip.show('<b>' + H.fmtX(best.x) + '</b><br>' + best.rows.map(r => '<span class="sw" style="background:' + r.s.color + '"></span>' + MLS.esc(r.s.name) + ': <b>' + H.fmtY(r.y) + '</b>').join('<br>'), e.clientX, e.clientY);
    }

    /* ---------- resize ---------- */
    let lastW = 0;
    const ro = new ResizeObserver(MLS.debounce(() => { const w = host.clientWidth; if (w && Math.abs(w - lastW) > 2) { lastW = w; P.redraw(); } }, 60));
    ro.observe(host);
    document.addEventListener('mls:theme', () => { if (host.getClientRects().length) P.redraw(); });

    /* ---------- export ---------- */
    P.svgString = function (title) {
      const c = svg.cloneNode(true);
      c.setAttribute('xmlns', NS);
      c.setAttribute('font-family', 'IBM Plex Sans, Segoe UI, Arial, sans-serif');
      if (title) { const t = document.createElementNS(NS, 'title'); t.textContent = title; c.insertBefore(t, c.firstChild); }
      return '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(c);
    };
    P.pngBlob = function (scale) {
      scale = scale || 2;
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
          const cv = document.createElement('canvas'); cv.width = P.W * scale; cv.height = P.H * scale;
          const ctx = cv.getContext('2d'); ctx.scale(scale, scale); ctx.drawImage(img, 0, 0, P.W, P.H);
          cv.toBlob(b => (b ? resolve(b) : reject(new Error('PNG failed'))), 'image/png');
        };
        img.onerror = () => reject(new Error('Could not render the figure'));
        img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(P.svgString());
      });
    };
    return P;
  };

  /* =================================================================
     legend
     items: [{ label, color, shape } | { label, color, line:true, dash }]
     ================================================================= */
  MLS.legend = function (host, items) {
    host.classList.add('legend');
    host.innerHTML = items.map(it => {
      let sw;
      if (it.line) sw = '<svg width="24" height="12" aria-hidden="true"><line x1="1" y1="6" x2="23" y2="6" stroke="' + it.color + '" stroke-width="' + (it.width || 2.5) + '"' + (it.dash ? ' stroke-dasharray="' + it.dash + '"' : '') + ' stroke-linecap="round"/></svg>';
      else if (it.box) sw = '<svg width="14" height="14" aria-hidden="true"><rect x="1" y="1" width="12" height="12" rx="3" fill="' + it.color + '"' + (it.stroke ? ' stroke="' + it.stroke + '"' : '') + '/></svg>';
      else sw = MLS.shapeSVG(it.shape || 'circle', it.color, 14, it.stroke);
      return '<span class="li">' + sw + '<span>' + it.label + '</span></span>';
    }).join('');
  };

  /* =================================================================
     figure card: title, menu (PNG / SVG / Data / Table), plot, legend
     cfg: { title, sub, file, plot: plotOpts, data: ()=>({ cols:[...], rows:[[...]] }) }
     ================================================================= */
  MLS.figure = function (host, cfg) {
    host.classList.add('figure');
    host.innerHTML =
      '<div class="figure-head"><div><div class="figure-title">' + cfg.title + (cfg.help ? MLS.helpBtn(cfg.help) : '') + '</div>' + (cfg.sub ? '<div class="figure-sub">' + cfg.sub + '</div>' : '') + '</div>' +
      '<div class="figure-menu proj-hide">' +
        '<button type="button" data-x="png" title="Download as PNG image">PNG</button>' +
        '<button type="button" data-x="svg" title="Download as SVG (vector) image">SVG</button>' +
        (cfg.data ? '<button type="button" data-x="xlsx" title="Download the plotted numbers as an Excel file">Data</button><button type="button" data-x="table" title="Show the numbers as a table" aria-pressed="false">Table</button>' : '') +
      '</div></div>' +
      '<div class="fig-plot"></div><div class="fig-legend"></div><div class="fig-caption"></div><div class="fig-table" hidden></div>';
    const plotHost = host.querySelector('.fig-plot'), legendHost = host.querySelector('.fig-legend'), capHost = host.querySelector('.fig-caption');
    const P = MLS.plot(plotHost, cfg.plot || {});
    const tableHost = host.querySelector('.fig-table');
    const fname = cfg.file || MLS.slug(cfg.title);
    function tableHTML() {
      const d = cfg.data(); const rows = d.rows.slice(0, 500);
      return '<table class="tbl"><thead><tr>' + d.cols.map(c => '<th>' + MLS.esc(c) + '</th>').join('') + '</tr></thead><tbody>' +
        rows.map(r => '<tr>' + r.map(v => '<td>' + (typeof v === 'number' ? MLS.fmt(v, Math.abs(v) >= 100 ? 1 : 3) : MLS.esc(v)) + '</td>').join('') + '</tr>').join('') + '</tbody></table>' +
        (d.rows.length > 500 ? '<div class="note" style="padding:6px">Showing the first 500 of ' + d.rows.length + ' rows; use Data for all rows.</div>' : '');
    }
    host.querySelector('.figure-menu').addEventListener('click', async e => {
      const b = e.target.closest('button'); if (!b) return;
      const x = b.dataset.x;
      if (x === 'svg') MLS.download(new Blob([P.svgString(cfg.title)], { type: 'image/svg+xml' }), '4MLSim-' + fname + '.svg');
      else if (x === 'png') { try { MLS.download(await P.pngBlob(2), '4MLSim-' + fname + '.png'); } catch (err) { MLS.toast(err.message); } }
      else if (x === 'xlsx') { const d = cfg.data(); MLS.exportXLSX({ [cfg.title.slice(0, 31)]: [d.cols].concat(d.rows) }, '4MLSim-' + fname + '.xlsx'); }
      else if (x === 'table') { const on = tableHost.hidden; tableHost.hidden = !on; b.setAttribute('aria-pressed', String(on)); if (on) tableHost.innerHTML = tableHTML(); }
    });
    return {
      plot: P,
      legend: items => MLS.legend(legendHost, items),
      caption: html => { capHost.innerHTML = html; },
      refreshTable: () => { if (!tableHost.hidden) tableHost.innerHTML = tableHTML(); }
    };
  };
})();
