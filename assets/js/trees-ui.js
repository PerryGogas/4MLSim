/* =====================================================================
   4MLSim tree and SVM modules: shared data sets (classification and
   regression), the tree diagram, and the chart of the data with the
   model (coloured regions or a forecast line).
   ===================================================================== */
(function () {
  'use strict';
  const MLS = window.MLS, ML = MLS.ml, CL = MLS.cls, F = MLS.fmt;
  const T = (MLS.trees = {});

  /* data sets: classification (2 inputs) and regression (1 input) */
  T.DATA = {
    fruits: { task: 'class', label: 'Fruits (diameter, height)', gen: ML.DATA.fruits, n: 120, unit: 'fruits', short: ['Diameter', 'Height'] },
    banks: { task: 'class', label: 'Bank failures (two ratios)', gen: ML.DATA.banks, n: 200, unit: 'banks', short: ['D/E', 'Tier 1'] },
    circles: { task: 'class', label: 'Circles (generic)', gen: ML.DATA.circles, n: 160, unit: 'points', short: ['x1', 'x2'] },
    kwh: { task: 'reg', label: 'Temperature and electricity use', gen: ML.DATA.kwh, n: 100, unit: 'days', short: ['Temp.'] },
    power: { task: 'reg', label: 'Gas price and electricity price', gen: ML.DATA.power, n: 100, unit: 'days', short: ['Gas'] }
  };
  T.make = function (key, seed, n) {
    const d = T.DATA[key], g = d.gen.gen(ML.rng(seed), n || d.n), test = ML.split(g.X.length, seed);
    const pts = g.X.map((x, i) => ({ x, y: g.y[i], te: test[i], id: i }));
    const out = { key, task: d.task, pts, names: g.names, short: d.short, classes: g.classes || [], K: (g.classes || []).length, unit: d.unit, label: d.label };
    if (d.task === 'class') out.dom = CL.domain(pts);
    else { const xs = pts.map(p => p.x[0]), ys = pts.map(p => p.y); out.dom = { x: MLS.niceDomain(Math.min(...xs), Math.max(...xs), 0.04), y: MLS.niceDomain(Math.min(...ys), Math.max(...ys), 0.1) }; }
    out.train = pts.filter(p => !p.te); out.test = pts.filter(p => p.te);
    return out;
  };
  /* scores: accuracy (classification) or RMSE (regression) of a prediction function */
  T.score = (D, pts, f) => (D.task === 'class' ? pts.filter(p => argmax(f(p.x)) === p.y).length / pts.length : Math.sqrt(pts.reduce((s, p) => s + (p.y - f(p.x)) ** 2, 0) / pts.length));
  const argmax = p => { let b = 0; for (let k = 1; k < p.length; k++) if (p[k] > p[b] + 1e-12) b = k; return b; };
  T.argmax = argmax;
  T.fmtScore = (D, v) => (D.task === 'class' ? CL.pct(v) : F(v, 2));
  T.scoreName = D => (D.task === 'class' ? 'Accuracy' : 'RMSE');
  T.unitY = D => ((D.names[1] || '').match(/\(([^)]+)\)/) || [])[1] || '';

  /* the data with the model: regions (classification) or the forecast line (regression) */
  T.drawData = function (P, D, f, o) {
    o = o || {};
    const ink = MLS.color('chart-ink'), mu = MLS.color('chart-muted');
    P.setX(D.dom.x[0], D.dom.x[1]); P.setY(D.dom.y[0], D.dom.y[1]);
    P.draw(P => {
      if (D.task === 'class') {
        if (f) CL.drawRegions(P, CL.regionGrid(D.dom, f, o.res || 90, Math.round((o.res || 90) * 0.72)), D.dom, D.K, null);
        if (o.before) o.before(P);
        const tip = p => '<b>' + (p.te ? 'Test set' : 'Training set') + '</b><br>' + D.names[0] + ': ' + F(p.x[0], 2) + '<br>' + D.names[1] + ': ' + F(p.x[1], 2) + '<br>Actual: <b>' + MLS.esc(D.classes[p.y]) + '</b>' + (f ? '<br>Model: ' + MLS.esc(D.classes[argmax(f(p.x))]) : '') + (o.tipExtra ? o.tipExtra(p) : '');
        P.points(D.train, { x: p => p.x[0], y: p => p.x[1], color: p => CL.color(p.y), r: o.r ? q => o.r(q) : 4.4, opacity: o.opacity ? q => o.opacity(q) : 0.92, tip, highlight: o.highlight });
        P.points(D.test, { x: p => p.x[0], y: p => p.x[1], color: p => CL.color(p.y), r: 5, hollow: true, tip, highlight: o.highlight });
      } else {
        if (o.before) o.before(P);
        const tip = p => '<b>' + (p.te ? 'Test set' : 'Training set') + '</b><br>' + D.names[0] + ': ' + F(p.x[0], 2) + '<br>' + D.names[1] + ': <b>' + F(p.y, 2) + '</b>' + (f ? '<br>Forecast: ' + F(f(p.x), 2) : '') + (o.tipExtra ? o.tipExtra(p) : '');
        P.points(D.train, { x: p => p.x[0], y: p => p.y, color: MLS.color('series-1'), r: o.r ? q => o.r(q) : 4.4, opacity: o.opacity ? q => o.opacity(q) : 0.9, tip, highlight: o.highlight });
        P.points(D.test, { x: p => p.x[0], y: p => p.y, color: MLS.color('series-1'), r: 5, hollow: true, tip, highlight: o.highlight });
        if (f) {
          const xs = []; for (let k = 0; k <= 500; k++) xs.push(D.dom.x[0] + (D.dom.x[1] - D.dom.x[0]) * k / 500);
          const pts = xs.map(x => [x, f([x])]);
          if (o.prev) P.line(xs.map(x => [x, o.prev([x])]), { color: mu, width: 1.6, dash: '5 4' });
          P.line(pts, { color: MLS.color('series-2'), width: 2.6 });
          P.crosshair([{ name: o.lineName || 'Forecast', color: MLS.color('series-2'), pts }], v => D.names[0].replace(/ \(.*\)/, '') + ' ' + F(v, 1), v => F(v, 2));
        }
      }
      if (o.after) o.after(P);
    });
  };
  T.legendData = function (D, extra) {
    const mu = MLS.color('chart-muted');
    const items = D.task === 'class'
      ? D.classes.map((c, k) => { const a = D.pts.filter(p => p.y === k); return { label: MLS.esc(c) + ' (' + a.filter(p => !p.te).length + ' training, ' + a.filter(p => p.te).length + ' test)', color: CL.color(k) }; }).concat([{ label: 'filled = training set, ring = test set', color: 'transparent', stroke: mu }])
      : [{ label: 'Training set (' + D.train.length + ')', color: MLS.color('series-1') }, { label: 'Test set (' + D.test.length + ')', color: 'transparent', stroke: MLS.color('series-1') }];
    return items.concat(extra || []);
  };

  /* ---------------------------------------------------------------
     tree diagram. root: nodes with left/right, feat, thr (and split =
     order of the split). o: { upto, maxDepth, lines(nd) -> [text], fill(nd), highlight(nd), op }
     --------------------------------------------------------------- */
  T.treeHeight = depth => w => Math.max(170, 64 + 76 * depth);
  T.shownDepth = function (root, o) { let d = 0; (function w(nd, k) { d = Math.max(d, k); if (nd.left && (o.upto === undefined || nd.split < o.upto) && k < o.maxDepth) { w(nd.left, k + 1); w(nd.right, k + 1); } })(root, 0); return d; };
  T.drawTree = function (P, root, o) {
    const open = (nd, k) => nd.left && (o.upto === undefined || nd.split < o.upto) && k < o.maxDepth;
    const leaves = [], pos = new Map();
    (function walk(nd, k) { if (open(nd, k)) { walk(nd.left, k + 1); walk(nd.right, k + 1); pos.set(nd, { x: (pos.get(nd.left).x + pos.get(nd.right).x) / 2, k }); } else { pos.set(nd, { x: leaves.length, k }); leaves.push(nd); } })(root, 0);
    const depth = Math.max(...[...pos.values()].map(v => v.k)), ink = MLS.color('chart-ink'), mu = MLS.color('chart-muted');
    P.setX(-0.55, Math.max(leaves.length - 0.45, 0.45)); P.setY(-depth - 0.55, 0.55);
    P.draw(P => {
      const pxX = (P.W - P.m.l - P.m.r) / (P.xd[1] - P.xd[0]), pxY = (P.H - P.m.t - P.m.b) / (P.yd[1] - P.yd[0]);
      const bw = Math.min(150, Math.max(60, pxX * 0.94)) / pxX, bh = Math.min(64, pxY * 0.82) / pxY;
      pos.forEach((v, nd) => { if (open(nd, v.k)) [nd.left, nd.right].forEach((c, s) => { const w = pos.get(c); P.segment(v.x, -v.k - bh / 2, w.x, -w.k + bh / 2, { color: mu, width: 1.4 }); P.text((v.x + w.x) / 2, (-v.k - w.k) / 2, s ? 'no' : 'yes', { anchor: 'middle', size: 10.5, weight: 700, color: mu }); }); });
      pos.forEach((v, nd) => {
        const hl = o.highlight && o.highlight(nd), fill = o.fill ? o.fill(nd, open(nd, v.k)) : MLS.color('surface');
        const el = P.rect(v.x - bw / 2, -v.k - bh / 2, v.x + bw / 2, -v.k + bh / 2, { fill, stroke: hl ? MLS.color('series-2') : ink, width: hl ? 3 : 1, rx: 6 });
        const lines = o.lines(nd, open(nd, v.k)), fs = Math.max(9, Math.min(11.5, bw * pxX / 12));
        lines.forEach((t, i) => P.text(v.x, -v.k, t, { anchor: 'middle', dy: (i - (lines.length - 1) / 2) * (fs + 2) + 4, size: fs, weight: i === 0 ? 800 : 600, halo: false, color: ink }));
        if (o.tip) { el.setAttribute('data-tip', P.tips.length); P.tips.push(() => o.tip(nd, open(nd, v.k))); }
      });
    });
    return { leaves: leaves.length, depth };
  };
})();
