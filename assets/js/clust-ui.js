/* =====================================================================
   4MLSim unsupervised modules: data sets, cluster colours, regions.
   ===================================================================== */
(function () {
  'use strict';
  const MLS = window.MLS, ML = MLS.ml, CL = MLS.cls;
  const U = (MLS.clust = {});
  U.DATA = {
    customers: { label: 'Supermarket customers (spending, visits)', gen: ML.DATA.customers, n: 150 },
    payments: { label: 'Card payments (hour, amount)', gen: ML.DATA.payments, n: 220 },
    regimes: { label: 'Market regimes (stock and bond returns)', gen: ML.DATA.regimes, n: 250 },
    blobs3: { label: 'Generic: three round groups', gen: ML.DATA.blobs3, n: 150 },
    elong: { label: 'Generic: stretched groups', gen: ML.DATA.elong, n: 240 },
    moonsU: { label: 'Generic: moons', gen: ML.DATA.moonsU, n: 160 },
    circlesU: { label: 'Generic: circles', gen: ML.DATA.circlesU, n: 200 }
  };
  U.make = function (key, seed, n) {
    const d = U.DATA[key], g = d.gen.gen(ML.rng(seed), n || d.n), sc = ML.scaler(g.X);
    const pts = g.X.map((x, i) => ({ x, id: i, truth: g.y ? g.y[i] : 0 }));
    return { key, pts, X: g.X, Z: g.X.map(sc.t), sc, names: g.names, dom: CL.domain(pts), label: d.label, note: d.gen.note };
  };
  U.color = k => (k < 0 ? MLS.color('chart-muted') : MLS.seriesColor(k));
  /* paint the plane by a hard assignment x -> cluster */
  U.regions = function (P, dom, assign, K, alpha) {
    const g = CL.regionGrid(dom, assign, 110, 80), rgb = Array.from({ length: K }, (_, k) => MLS.hexToRgb(U.color(k))), a = alpha || (MLS.getTheme() === 'dark' ? 0.17 : 0.15);
    P.image(MLS.gridImage(g.P, k => (k < 0 ? [0, 0, 0, 0] : rgb[k].concat([Math.round(255 * a)]))), dom.x[0], dom.x[1], dom.y[0], dom.y[1], { smooth: true });
  };
  U.sizes = (labels, K) => { const c = new Array(K).fill(0); labels.forEach(l => { if (l >= 0) c[l]++; }); return c; };
  U.legend = (K, sizes, extra) => Array.from({ length: K }, (_, k) => ({ label: 'Cluster ' + (k + 1) + (sizes ? ' (' + sizes[k] + ')' : ''), color: U.color(k) })).concat(extra || []);
})();
