/* =====================================================================
   4MLSim SVM modules: fitting on standardized inputs, cross-validation,
   and the chart with the boundary, the margins and the support vectors.
   ===================================================================== */
(function () {
  'use strict';
  const MLS = window.MLS, ML = MLS.ml, CL = MLS.cls, F = MLS.fmt;
  const V = (MLS.svmui = {});

  /* pts: [{x:[..], y:0/1}]; ko: { kernel, gamma, degree }; labels +1 for class 1 */
  V.fit = function (pts, ko, C) {
    const sc = ML.scaler(pts.map(p => p.x)), Z = pts.map(p => sc.t(p.x)), s = pts.map(p => (p.y === 1 ? 1 : -1));
    const ker = ML.kernel(ko, Z[0].length), m = ML.svc(Z, s, C, ker, 1e-5);
    const f = x => m.f(sc.t(x));
    const alpha = new Map(pts.map((p, i) => [p, m.alpha[i]]));
    const sv = new Set(m.sv.map(i => pts[i]));
    /* linear kernel: the line in standardized units (w, rho) and in the original units */
    let w = null;
    if (ko.kernel === 'linear') {
      w = Z[0].map((_, j) => m.sv.reduce((acc, i) => acc + m.alpha[i] * s[i] * Z[i][j], 0));
      const a = w.map((v, j) => v / sc.sd[j]), b = -m.rho - w.reduce((acc, v, j) => acc + v * sc.mu[j] / sc.sd[j], 0);
      w = { w, a, b, norm: Math.sqrt(w.reduce((acc, v) => acc + v * v, 0)) };
    }
    return { sc, m, f, alpha, sv, w, C, prob: x => { const v = ML.sigmoid(2 * f(x)); return [1 - v, v]; } };
  };
  V.acc = (fit, pts) => pts.filter(p => ((fit.f(p.x) >= 0 ? 1 : 0) === p.y)).length / Math.max(1, pts.length);
  /* 4-fold cross-validation on the training set (consecutive blocks) */
  V.cv = function (pts, ko, C) {
    const n = pts.length, K = 4; let ok = 0;
    for (let q = 0; q < K; q++) {
      const lo = Math.round(q * n / K), hi = Math.round((q + 1) * n / K), tr = pts.filter((_, i) => i < lo || i >= hi), va = pts.slice(lo, hi);
      if (new Set(tr.map(p => p.y)).size < 2) return NaN;
      ok += V.acc(V.fit(tr, ko, C), va) * va.length;
    }
    return ok / n;
  };
  /* where a training point stands: y·f(x) >= 1 outside the margin, 0..1 inside, < 0 wrong side */
  V.where = (fit, p) => { const yf = (p.y === 1 ? 1 : -1) * fit.f(p.x), a = fit.alpha.get(p) || 0;
    return yf < -1e-9 ? 'on the wrong side' : a > 1e-8 && a < fit.C - 1e-6 ? 'on the margin' : yf < 1 - 1e-6 ? 'inside the margin' : 'outside the margin'; };

  V.draw = function (P, D, fit, o) {
    o = o || {};
    const ink = MLS.color('chart-ink'), dom = D.dom;
    P.setX(dom.x[0], dom.x[1]); P.setY(dom.y[0], dom.y[1]);
    const nx = 110, ny = 80, xs = [], ys = [];
    for (let i = 0; i < nx; i++) xs.push(dom.x[0] + (i + 0.5) * (dom.x[1] - dom.x[0]) / nx);
    for (let j = 0; j < ny; j++) ys.push(dom.y[0] + (j + 0.5) * (dom.y[1] - dom.y[0]) / ny);
    const fg = ys.map(y => xs.map(x => fit.f([x, y])));
    P.draw(P => {
      CL.drawRegions(P, { xs, ys, P: fg.map(r => r.map(v => { const q = ML.sigmoid(2 * v); return [1 - q, q]; })) }, dom, 2, 0.5);
      P.contour(fg, xs, ys, [-1, 1], { color: ink, width: 1.4, dash: '6 4', opacity: 0.85 });
      const tip = p => '<b>' + (p.added ? 'Your point' : p.te ? 'Test set' : 'Training set') + '</b><br>' + D.names[0] + ': ' + F(p.x[0], 2) + '<br>' + D.names[1] + ': ' + F(p.x[1], 2) + '<br>Actual: <b>' + MLS.esc(D.classes[p.y]) + '</b><br>Decision value f(x) = ' + F(fit.f(p.x), 3) +
        (p.te ? '' : '<br>' + (fit.sv.has(p) ? '<b>Support vector</b>, α = ' + F(fit.alpha.get(p), 3) + ', ' + V.where(fit, p) : V.where(fit, p) + ': not a support vector')) + (o.clickable ? '<br><i>Click to delete</i>' : '');
      const tr = D.pts.filter(p => !p.te), te = D.pts.filter(p => p.te);
      P.points(tr, { x: p => p.x[0], y: p => p.x[1], color: p => CL.color(p.y), r: 4.8, tip, highlight: p => fit.sv.has(p) });
      P.points(te, { x: p => p.x[0], y: p => p.x[1], color: p => CL.color(p.y), r: 5.2, hollow: true, tip });
      if (o.after) o.after(P);
    });
  };
  V.legend = function (D, fit, extra) {
    const mu = MLS.color('chart-muted'), ink = MLS.color('chart-ink');
    return D.classes.map((c, k) => { const a = D.pts.filter(p => p.y === k); return { label: MLS.esc(c) + ' (' + a.filter(p => !p.te).length + ' training, ' + a.filter(p => p.te).length + ' test)', color: CL.color(k) }; })
      .concat([{ label: 'filled = training set, ring = test set', color: 'transparent', stroke: mu }, { label: 'Separator, f(x) = 0', color: ink, line: true }, { label: 'Margins, f(x) = ±1', color: ink, line: true, dash: '6 4' }, { label: 'Support vector (' + fit.sv.size + ')', color: 'transparent', stroke: ink }])
      .concat(extra || []);
  };
})();
