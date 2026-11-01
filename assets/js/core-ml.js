/* =====================================================================
   4MLSim Core: small, readable machine learning algorithms, data sets
   and metrics for the Core Playground. No dependencies; also runs in
   Node (module.exports) for checks against scikit-learn.
   ===================================================================== */
(function (root) {
  'use strict';
  const ML = {};

  /* ---------------------------------------------------------------
     random numbers (same generator as MLS.rng)
     --------------------------------------------------------------- */
  ML.rng = function (seed) {
    let a = (Number(seed) >>> 0) || 1;
    function random() {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    let spare = null;
    function normal(mu, sd) {
      mu = mu || 0; sd = sd === undefined ? 1 : sd;
      if (spare !== null) { const s = spare; spare = null; return mu + sd * s; }
      let u = 0; while (u === 0) u = random();
      const v = random(), r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return mu + sd * r * Math.cos(2 * Math.PI * v);
    }
    return {
      random, normal,
      uniform: (lo, hi) => lo + (hi - lo) * random(),
      int: (lo, hi) => lo + Math.floor(random() * (hi - lo + 1)),
      shuffle(arr) { for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; } return arr; }
    };
  };

  /* ---------------------------------------------------------------
     small helpers
     --------------------------------------------------------------- */
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const sigmoid = z => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
  function softmax(z) { const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); }
  ML.sigmoid = sigmoid; ML.softmax = softmax;

  /* standardize with the training set's mean and standard deviation */
  ML.scaler = function (X) {
    const p = X[0].length, n = X.length, mu = new Array(p).fill(0), sd = new Array(p).fill(0);
    X.forEach(r => r.forEach((v, j) => (mu[j] += v / n)));
    X.forEach(r => r.forEach((v, j) => (sd[j] += (v - mu[j]) ** 2 / n)));
    for (let j = 0; j < p; j++) sd[j] = Math.sqrt(sd[j]) || 1;
    return { mu, sd, t: x => x.map((v, j) => (v - mu[j]) / sd[j]) };
  };

  /* solve A x = b (Gaussian elimination with partial pivoting) */
  function solve(A, b) {
    const n = b.length, M = A.map((r, i) => r.slice().concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      const d = M[c][c]; if (Math.abs(d) < 1e-14) return null;
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / d; if (f) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
    return x;
  }
  ML.solve = solve;

  /* split: the last 20% of a random order is the test set */
  ML.split = function (n, seed, share) {
    const idx = ML.rng(seed * 7 + 3).shuffle(Array.from({ length: n }, (_, i) => i));
    const nTe = Math.round((share === undefined ? 0.2 : share) * n), test = new Array(n).fill(false);
    idx.slice(n - nTe).forEach(i => (test[i] = true));
    return test;
  };

  /* =================================================================
     data sets.  Every generator returns
       { X: [[x1, x2]] or [[x]], y: [] (class index or number),
         names: [axis names], classes: [class names] (classification),
         task: 'class' | 'reg' | 'clust' }
     ================================================================= */
  const DATA = {};
  function gauss2(r, n, cls, specs) {   /* specs: [[mx, sx, my, sy], ...] per class; cls(i) picks the class */
    const X = [], y = [];
    for (let i = 0; i < n; i++) { const k = cls(i), s = specs[k]; X.push([r.normal(s[0], s[1]), r.normal(s[2], s[3])]); y.push(k); }
    return { X, y };
  }
  const round = (v, d) => +v.toFixed(d);
  DATA.banks = {
    task: 'class', label: 'Bank failures', n: 150,
    note: 'Banks described by two financial ratios. Failed banks tend to have more debt per unit of equity and less capital. About 30% of the banks failed.',
    gen(r, n, share) {
      const X = [], y = [], sh = share === undefined ? 0.3 : share;
      for (let i = 0; i < n; i++) {
        const f = r.random() < sh ? 1 : 0;
        const de = f ? r.normal(6.2, 1.6) : r.normal(3.4, 1.3), t1 = f ? r.normal(9, 2.2) : r.normal(12.8, 2.3);
        X.push([round(clamp(de, 0.3, 11), 2), round(clamp(t1, 2, 22), 1)]); y.push(f);
      }
      return { X, y, names: ['Debt-to-equity ratio', 'Tier 1 capital ratio (%)'], classes: ['Healthy', 'Failed'], pos: 1 };
    }
  };
  DATA.cycle = {
    task: 'class', label: 'Recession / normal / growth', n: 150,
    note: 'Months of an economy described by the purchasing managers\' index (PMI) and the change in the unemployment rate over the last 12 months. Three classes.',
    gen(r, n) {
      const d = gauss2(r, n, () => r.int(0, 2), [[44.5, 2.8, 1.1, 0.55], [50.5, 2.2, 0.05, 0.4], [55.5, 2.6, -0.75, 0.4]]);
      d.X = d.X.map(x => [round(x[0], 1), round(x[1], 2)]);
      return Object.assign(d, { names: ['PMI (index)', 'Change in unemployment (points)'], classes: ['Recession', 'Normal', 'Growth'] });
    }
  };
  DATA.fruits = {
    task: 'class', label: 'Fruits', n: 120,
    note: 'Fruits measured by diameter and height, as in the decision tree example of the lecture. Apples and pears overlap.',
    gen(r, n) {
      const d = gauss2(r, n, () => r.int(0, 3), [[7.4, 0.7, 7.0, 0.7], [6.4, 0.7, 9.4, 1.0], [3.8, 0.5, 18, 2], [1.8, 0.3, 2.1, 0.35]]);
      d.X = d.X.map(x => [round(Math.max(0.8, x[0]), 1), round(Math.max(0.8, x[1]), 1)]);
      return Object.assign(d, { names: ['Diameter (cm)', 'Height (cm)'], classes: ['Apple', 'Pear', 'Banana', 'Grape'] });
    }
  };
  DATA.firms = {
    task: 'class', label: 'Firms: two related ratios', n: 200,
    note: 'Firms described by the profit margin and the return on assets. The two ratios move together (correlation about 0.85 within each group), which breaks the naive assumption.',
    gen(r, n) {
      const X = [], y = [];
      for (let i = 0; i < n; i++) {
        const k = r.random() < 0.35 ? 1 : 0, m = k ? r.normal(1.5, 3) : r.normal(8, 3), roa = 0.55 * m + (k ? -1.2 : 0.4) + r.normal(0, 0.9);
        X.push([round(m, 2), round(roa, 2)]); y.push(k);
      }
      return { X, y, names: ['Profit margin (%)', 'Return on assets (%)'], classes: ['Sound', 'Distressed'], pos: 1 };
    }
  };
  DATA.sep = {
    task: 'class', label: 'Banks, clearly separated', n: 60,
    note: 'Two groups of banks that a straight line can separate perfectly: the textbook case of the maximum-margin line.',
    gen(r, n) {
      const X = [], y = [];
      for (let i = 0; i < n; i++) { const f = i % 3 === 0 ? 1 : 0; X.push([round(clamp(f ? r.normal(7.2, 0.9) : r.normal(3, 0.9), 0.3, 11), 2), round(clamp(f ? r.normal(7.5, 1.3) : r.normal(13.5, 1.3), 2, 22), 1)]); y.push(f); }
      return { X, y, names: ['Debt-to-equity ratio', 'Tier 1 capital ratio (%)'], classes: ['Healthy', 'Failed'], pos: 1 };
    }
  };
  DATA.de1 = {
    task: 'class', label: 'Banks: debt-to-equity only (lecture example)', n: 80,
    note: 'One input. A bank is "bad" when its debt-to-equity ratio is too high (too risky) or too low (too little use of leverage to earn a return). No single cutoff can separate the classes.',
    gen(r, n) {
      const X = [], y = [];
      for (let i = 0; i < n; i++) { const x = r.uniform(0.2, 11.8), bad = (x < 2.6 || x > 8.4) ? (r.random() < 0.93 ? 1 : 0) : (r.random() < 0.07 ? 1 : 0); X.push([round(x, 2)]); y.push(bad); }
      return { X, y, names: ['Debt-to-equity ratio'], classes: ['Good', 'Bad'], pos: 1 };
    }
  };
  DATA.blobs2 = {
    task: 'class', label: 'Generic: two groups', n: 120, generic: true,
    note: 'Two groups of points that overlap a little.',
    gen(r, n) { const d = gauss2(r, n, i => i % 2, [[-1, 0.9, -0.6, 0.9], [1.1, 0.9, 0.7, 0.9]]); return Object.assign(d, { names: ['x1', 'x2'], classes: ['Class 0', 'Class 1'], pos: 1 }); }
  };
  DATA.circles = {
    task: 'class', label: 'Generic: circles', n: 160, generic: true,
    note: 'One class inside a ring of the other. No straight line can separate them.',
    gen(r, n) {
      const X = [], y = [];
      for (let i = 0; i < n; i++) { const k = i % 2, a = r.uniform(0, 2 * Math.PI), rad = (k ? 1 : 2.2) + r.normal(0, 0.25); X.push([rad * Math.cos(a), rad * Math.sin(a)]); y.push(k); }
      return { X, y, names: ['x1', 'x2'], classes: ['Outer ring', 'Inner circle'], pos: 1 };
    }
  };
  DATA.moons = {
    task: 'class', label: 'Generic: moons', n: 160, generic: true,
    note: 'Two interleaving half circles.',
    gen(r, n) {
      const X = [], y = [];
      for (let i = 0; i < n; i++) { const k = i % 2, t = r.uniform(0, Math.PI); X.push(k ? [1 - Math.cos(t) + r.normal(0, 0.18), 0.5 - Math.sin(t) + r.normal(0, 0.18)] : [Math.cos(t) + r.normal(0, 0.18), Math.sin(t) + r.normal(0, 0.18)]); y.push(k); }
      return { X, y, names: ['x1', 'x2'], classes: ['Moon A', 'Moon B'], pos: 1 };
    }
  };
  DATA.xor = {
    task: 'class', label: 'Generic: XOR', n: 160, generic: true,
    note: 'Class 1 in two opposite corners, class 0 in the other two.',
    gen(r, n) {
      const X = [], y = [];
      for (let i = 0; i < n; i++) { const a = r.uniform(-1, 1), b = r.uniform(-1, 1); X.push([a + r.normal(0, 0.08), b + r.normal(0, 0.08)]); y.push(a * b > 0 ? 1 : 0); }
      return { X, y, names: ['x1', 'x2'], classes: ['Class 0', 'Class 1'], pos: 1 };
    }
  };
  DATA.kwh = {
    task: 'reg', label: 'Temperature and electricity use', n: 80,
    note: 'Daily electricity use of a household against the outside temperature: heating on cold days, air-conditioning on hot days.',
    gen(r, n) {
      const X = [], y = [];
      for (let i = 0; i < n; i++) { const t = r.uniform(-5, 38); X.push([round(t, 1)]); y.push(round(9 + 0.55 * Math.max(0, 16 - t) + 0.9 * Math.max(0, t - 22) + r.normal(0, 1.2), 2)); }
      return { X, y, names: ['Temperature (°C)', 'Electricity use (kWh per day)'] };
    }
  };
  DATA.power = {
    task: 'reg', label: 'Gas price and electricity price', n: 80,
    note: 'Wholesale electricity price against the natural gas price: a roughly straight-line relation.',
    gen(r, n) {
      const X = [], y = [];
      for (let i = 0; i < n; i++) { const g = r.uniform(20, 120); X.push([round(g, 1)]); y.push(round(25 + 1.9 * g + r.normal(0, 14), 1)); }
      return { X, y, names: ['Natural gas price (€/MWh)', 'Electricity price (€/MWh)'] };
    }
  };
  DATA.line = {
    task: 'reg', label: 'Generic: straight line', n: 60, generic: true, note: 'A straight line plus noise.',
    gen(r, n) { const X = [], y = []; for (let i = 0; i < n; i++) { const x = r.uniform(0, 10); X.push([x]); y.push(5 + 1.2 * x + r.normal(0, 1.5)); } return { X, y, names: ['x', 'y'] }; }
  };
  DATA.wave = {
    task: 'reg', label: 'Generic: wave', n: 80, generic: true, note: 'A wave plus noise.',
    gen(r, n) { const X = [], y = []; for (let i = 0; i < n; i++) { const x = r.uniform(0, 10); X.push([x]); y.push(10 + 3 * Math.sin(0.8 * x) + r.normal(0, 0.6)); } return { X, y, names: ['x', 'y'] }; }
  };
  DATA.customers = {
    task: 'clust', label: 'Supermarket customers', n: 150,
    note: 'Customers described by yearly spending and visits per month. The groups are not labelled: the algorithm must find them.',
    gen(r, n) {
      const d = gauss2(r, n, () => r.int(0, 2), [[1.3, 0.45, 2.2, 0.8], [3.6, 0.6, 7, 1.2], [7.5, 0.9, 4, 1]]);
      d.X = d.X.map(x => [round(Math.max(0.2, x[0]), 2), round(Math.max(0.5, x[1]), 1)]);
      return Object.assign(d, { names: ['Yearly spending (thousand €)', 'Visits per month'] });
    }
  };
  DATA.blobs3 = {
    task: 'clust', label: 'Generic: three groups', n: 150, generic: true, note: 'Three round groups.',
    gen(r, n) { const d = gauss2(r, n, i => i % 3, [[-2, 0.6, 0, 0.6], [1.5, 0.6, 1.5, 0.6], [1, 0.6, -1.8, 0.6]]); return Object.assign(d, { names: ['x1', 'x2'] }); }
  };
  DATA.moonsU = {
    task: 'clust', label: 'Generic: moons', n: 160, generic: true, note: 'Two interleaving half circles, without labels.',
    gen(r, n) {
      const X = [], y = [];
      for (let i = 0; i < n; i++) { const k = i % 2, t = r.uniform(0, Math.PI); X.push(k ? [1 - Math.cos(t) + r.normal(0, 0.09), 0.5 - Math.sin(t) + r.normal(0, 0.09)] : [Math.cos(t) + r.normal(0, 0.09), Math.sin(t) + r.normal(0, 0.09)]); y.push(k); }
      return { X, y, names: ['x1', 'x2'] };
    }
  };

  /* monthly series for the LSTM module */
  DATA.unemp = {
    task: 'ts', label: 'Unemployment rate (%)', n: 240,
    note: 'Monthly unemployment rate. In recessions it jumps up quickly; in expansions it drifts down slowly: an asymmetric, nonlinear pattern.',
    gen(r, n) {
      const y = []; let u = 5, rec = 0;
      for (let t = 0; t < n; t++) {
        if (!rec && r.random() < 0.025) rec = 6 + r.int(0, 8);
        u += rec ? 0.28 + r.normal(0, 0.06) : -0.045 * (u - 3.8) / 2 + r.normal(0, 0.06); if (rec) rec--;
        u = Math.max(2.5, u); y.push(round(u, 2));
      }
      return { y, name: 'Unemployment rate (%)' };
    }
  };
  DATA.inflation = {
    task: 'ts', label: 'Inflation (%, year on year)', n: 240,
    note: 'Monthly year-on-year inflation: persistent, mean-reverting towards about 2%, hit by shocks.',
    gen(r, n) { const y = []; let p = 2.5, e1 = 0; for (let t = 0; t < n; t++) { const e = r.normal(0, 0.22); p = 2 + 0.95 * (p - 2) + e + 0.4 * e1; e1 = e; y.push(round(p, 2)); } return { y, name: 'Inflation (%)' }; }
  };
  DATA.ipgrowth = {
    task: 'ts', label: 'Industrial production growth (%, month on month)', n: 240,
    note: 'Monthly growth of industrial production with a seasonal pattern (12 months) and noise.',
    gen(r, n) { const y = []; let a = 0, b = 0; for (let t = 0; t < n; t++) { const v = 0.15 + 0.35 * a - 0.15 * b + 0.9 * Math.sin(2 * Math.PI * t / 12) + 0.5 * Math.cos(2 * Math.PI * t / 6) + r.normal(0, 0.35); b = a; a = v - 0.9 * Math.sin(2 * Math.PI * t / 12) - 0.5 * Math.cos(2 * Math.PI * t / 6); y.push(round(v, 3)); } return { y, name: 'IP growth (%)' }; }
  };
  ML.DATA = DATA;

  /* =================================================================
     classification models: fit(X, y, K, opts) -> { proba(x) -> [K] }
     X is already standardized by the caller (except trees, which do
     not need it; standardizing does not change their splits' meaning).
     ================================================================= */
  const CLS = {};

  /* features for the logit: inputs, plus squares and product when degree = 2 */
  function polyFeat(x, deg) {
    if (deg < 2) return x.slice();
    const f = x.slice();
    for (let i = 0; i < x.length; i++) for (let j = i; j < x.length; j++) f.push(x[i] * x[j]);
    return f;
  }
  ML.polyFeat = polyFeat;

  /* logit (binary) and multinomial logit; ridge penalty lambda/2 * |w|^2 on the slopes (as scikit-learn with C = 1/lambda) */
  CLS.logit = function (X, y, K, o) {
    const deg = o.degree || 1, lam = o.lambda === undefined ? 0.01 : o.lambda;
    const F = X.map(x => [1].concat(polyFeat(x, deg))), p = F[0].length, n = F.length;
    if (K === 2) {
      let b = new Array(p).fill(0);
      for (let it = 0; it < 100; it++) {
        const g = new Array(p).fill(0), H = Array.from({ length: p }, () => new Array(p).fill(0));
        F.forEach((f, i) => { const pr = sigmoid(f.reduce((s, v, j) => s + v * b[j], 0)), w = pr * (1 - pr); for (let j = 0; j < p; j++) { g[j] += (pr - y[i]) * f[j]; for (let k = 0; k < p; k++) H[j][k] += w * f[j] * f[k]; } });
        for (let j = 1; j < p; j++) { g[j] += lam * b[j]; H[j][j] += lam; }
        const step = solve(H, g); if (!step) break;
        b = b.map((v, j) => v - step[j]);
        if (Math.max(...step.map(Math.abs)) < 1e-10) break;
      }
      return { coef: b, proba: x => { const pr = sigmoid([1].concat(polyFeat(x, deg)).reduce((s, v, j) => s + v * b[j], 0)); return [1 - pr, pr]; } };
    }
    /* multinomial: Newton on all K x p coefficients */
    const P = K * p; let w = new Array(P).fill(0);
    const prob = f => softmax(Array.from({ length: K }, (_, k) => f.reduce((s, v, j) => s + v * w[k * p + j], 0)));
    for (let it = 0; it < 100; it++) {
      const g = new Array(P).fill(0), H = Array.from({ length: P }, () => new Array(P).fill(0));
      F.forEach((f, i) => {
        const pr = prob(f);
        for (let k = 0; k < K; k++) {
          const e = pr[k] - (y[i] === k ? 1 : 0);
          for (let j = 0; j < p; j++) g[k * p + j] += e * f[j];
          for (let l = 0; l < K; l++) { const c = pr[k] * ((k === l ? 1 : 0) - pr[l]); if (!c) continue; for (let j = 0; j < p; j++) for (let m = 0; m < p; m++) H[k * p + j][l * p + m] += c * f[j] * f[m]; }
        }
      });
      for (let k = 0; k < K; k++) for (let j = 1; j < p; j++) { g[k * p + j] += lam * w[k * p + j]; H[k * p + j][k * p + j] += lam; }
      for (let k = 0; k < K; k++) H[k * p][k * p] += 1e-9;      /* intercepts: only identified up to a constant */
      const step = solve(H, g); if (!step) break;
      w = w.map((v, j) => v - step[j]);
      if (Math.max(...step.map(Math.abs)) < 1e-10) break;
    }
    return { coef: w, proba: x => prob([1].concat(polyFeat(x, deg))) };
  };

  /* k nearest neighbours: share of each class among the k closest training points */
  function nearest(X, x, k) {
    const d = X.map((r, i) => { let s = 0; for (let j = 0; j < r.length; j++) s += (r[j] - x[j]) ** 2; return [s, i]; });
    d.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    return d.slice(0, Math.min(k, d.length)).map(v => v[1]);
  }
  ML.nearest = nearest;
  CLS.knn = function (X, y, K, o) {
    const k = o.k || 5;
    return { proba: x => { const c = new Array(K).fill(0), nb = nearest(X, x, k); nb.forEach(i => (c[y[i]] += 1 / nb.length)); return c; }, neighbours: x => nearest(X, x, k) };
  };

  /* ---------------------------------------------------------------
     decision tree (CART). Classification: Gini; regression: squared error.
     rows: indices into X; returns a node tree. Splits at midpoints.
     --------------------------------------------------------------- */
  const gini = c => { const n = c.reduce((a, b) => a + b, 0); return n ? 1 - c.reduce((s, v) => s + (v / n) ** 2, 0) : 0; };
  ML.gini = gini;
  function buildTree(X, y, rows, o, depth, rng) {
    const K = o.K, reg = !K, n = rows.length;
    const node = { n, depth };
    if (reg) { node.value = rows.reduce((s, i) => s + y[i], 0) / n; node.imp = rows.reduce((s, i) => s + (y[i] - node.value) ** 2, 0) / n; }
    else { node.counts = new Array(K).fill(0); rows.forEach(i => node.counts[y[i]]++); node.imp = gini(node.counts); node.value = node.counts.map(c => c / n); }
    if (depth >= o.maxDepth || n < 2 * (o.minLeaf || 1) || node.imp < 1e-12) return node;
    const p = X[0].length;
    let feats = Array.from({ length: p }, (_, j) => j);
    if (o.maxFeatures && o.maxFeatures < p) feats = rng.shuffle(feats.slice()).slice(0, o.maxFeatures);
    let best = null;
    for (const j of feats) {
      const ord = rows.slice().sort((a, b) => X[a][j] - X[b][j] || a - b);
      if (reg) {
        let sl = 0, sl2 = 0; const st = ord.reduce((s, i) => s + y[i], 0), st2 = ord.reduce((s, i) => s + y[i] * y[i], 0);
        for (let t = 0; t < n - 1; t++) {
          const v = y[ord[t]]; sl += v; sl2 += v * v;
          const a = X[ord[t]][j], b = X[ord[t + 1]][j]; if (a === b) continue;
          const nl = t + 1, nr = n - nl; if (nl < (o.minLeaf || 1) || nr < (o.minLeaf || 1)) continue;
          const sse = (sl2 - sl * sl / nl) + ((st2 - sl2) - (st - sl) ** 2 / nr);
          if (!best || sse < best.score - 1e-12) best = { j, thr: (a + b) / 2, score: sse, t };
        }
      } else {
        const L = new Array(K).fill(0), R = node.counts.slice();
        for (let t = 0; t < n - 1; t++) {
          const c = y[ord[t]]; L[c]++; R[c]--;
          const a = X[ord[t]][j], b = X[ord[t + 1]][j]; if (a === b) continue;
          const nl = t + 1, nr = n - nl; if (nl < (o.minLeaf || 1) || nr < (o.minLeaf || 1)) continue;
          const score = nl * gini(L) + nr * gini(R);
          if (!best || score < best.score - 1e-12) best = { j, thr: (a + b) / 2, score };
        }
      }
    }
    if (!best) return node;      /* as scikit-learn: a split is made even when it does not lower the impurity (needed e.g. for XOR) */
    node.feat = best.j; node.thr = best.thr;
    node.left = buildTree(X, y, rows.filter(i => X[i][best.j] <= best.thr), o, depth + 1, rng);
    node.right = buildTree(X, y, rows.filter(i => X[i][best.j] > best.thr), o, depth + 1, rng);
    return node;
  }
  function treeLeaf(node, x) { while (node.left) node = x[node.feat] <= node.thr ? node.left : node.right; return node; }
  ML.buildTree = buildTree; ML.treeLeaf = treeLeaf;
  CLS.tree = function (X, y, K, o) {
    const root = buildTree(X, y, X.map((_, i) => i), { K, maxDepth: o.depth || 3 }, 0, ML.rng(o.seed || 1));
    return { root, proba: x => treeLeaf(root, x).value };
  };

  /* random forest: bootstrap samples, one variable tried at each split (of 2), trees vote */
  CLS.forest = function (X, y, K, o) {
    const r = ML.rng((o.seed || 1) * 31 + 7), n = X.length, p = X[0].length, trees = [];
    for (let t = 0; t < (o.trees || 50); t++) {
      const rows = Array.from({ length: n }, () => Math.floor(r.random() * n));
      trees.push(buildTree(X, y, rows, { K, maxDepth: o.depth || 6, maxFeatures: Math.max(1, Math.floor(Math.sqrt(p))) }, 0, r));
    }
    return {
      trees,
      proba: x => { const v = new Array(K).fill(0); trees.forEach(tr => { const pr = treeLeaf(tr, x).value; let b = 0; for (let k = 1; k < K; k++) if (pr[k] > pr[b]) b = k; v[b] += 1 / trees.length; }); return v; }
    };
  };

  /* ---------------------------------------------------------------
     gradient-boosted trees, XGBoost style: each new tree is fitted to
     the gradients g and curvatures h of the loss; leaf weight -G/(H+lambda),
     split gain G_L^2/(H_L+l) + G_R^2/(H_R+l) - G^2/(H+l).
     --------------------------------------------------------------- */
  function xgbTree(X, g, h, rows, o, depth) {
    const lam = o.lambda === undefined ? 1 : o.lambda, mcw = o.minChild === undefined ? 1 : o.minChild;
    let G = 0, H = 0; rows.forEach(i => { G += g[i]; H += h[i]; });
    const node = { w: -G / (H + lam), n: rows.length };
    if (depth >= o.maxDepth || rows.length < 2) return node;
    let best = null; const base = G * G / (H + lam);
    for (let j = 0; j < X[0].length; j++) {
      const ord = rows.slice().sort((a, b) => X[a][j] - X[b][j] || a - b);
      let gl = 0, hl = 0;
      for (let t = 0; t < ord.length - 1; t++) {
        gl += g[ord[t]]; hl += h[ord[t]];
        const a = X[ord[t]][j], b = X[ord[t + 1]][j]; if (a === b) continue;
        const gr = G - gl, hr = H - hl; if (hl < mcw || hr < mcw) continue;
        const gain = 0.5 * (gl * gl / (hl + lam) + gr * gr / (hr + lam) - base);
        if (gain > 1e-12 && (!best || gain > best.gain)) best = { j, thr: (a + b) / 2, gain };
      }
    }
    if (!best) return node;
    node.feat = best.j; node.thr = best.thr;
    node.left = xgbTree(X, g, h, rows.filter(i => X[i][best.j] <= best.thr), o, depth + 1);
    node.right = xgbTree(X, g, h, rows.filter(i => X[i][best.j] > best.thr), o, depth + 1);
    return node;
  }
  const xgbOut = (node, x) => { while (node.left) node = x[node.feat] <= node.thr ? node.left : node.right; return node.w; };
  ML.xgbTree = xgbTree; ML.xgbOut = xgbOut;
  CLS.xgb = function (X, y, K, o) {
    const n = X.length, eta = o.eta || 0.3, rounds = o.rounds || 50, opt = { maxDepth: o.depth || 3 }, rows = X.map((_, i) => i);
    if (K === 2) {
      const F = new Array(n).fill(0), trees = [];          /* start: probability 0.5 for everyone (margin 0) */
      for (let t = 0; t < rounds; t++) {
        const pr = F.map(sigmoid), g = pr.map((p, i) => p - y[i]), h = pr.map(p => Math.max(p * (1 - p), 1e-16));
        const tr = xgbTree(X, g, h, rows, opt, 0); trees.push(tr);
        for (let i = 0; i < n; i++) F[i] += eta * xgbOut(tr, X[i]);
      }
      return { trees, proba: x => { const f = trees.reduce((s, tr) => s + eta * xgbOut(tr, x), 0), p = sigmoid(f); return [1 - p, p]; } };
    }
    const F = Array.from({ length: n }, () => new Array(K).fill(0)), trees = [];
    for (let t = 0; t < rounds; t++) {
      const pr = F.map(softmax), round = [];
      for (let k = 0; k < K; k++) {
        const g = pr.map((p, i) => p[k] - (y[i] === k ? 1 : 0)), h = pr.map(p => Math.max(2 * p[k] * (1 - p[k]), 1e-16));
        round.push(xgbTree(X, g, h, rows, opt, 0));
      }
      trees.push(round);
      for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) F[i][k] += eta * xgbOut(round[k], X[i]);
    }
    return { trees, proba: x => softmax(Array.from({ length: K }, (_, k) => trees.reduce((s, r) => s + eta * xgbOut(r[k], x), 0))) };
  };

  /* ---------------------------------------------------------------
     support vector machine. Dual coordinate descent; the bias is
     handled by adding 1 to the kernel (as in LIBLINEAR).
     More than 2 classes: one class against the rest.
     --------------------------------------------------------------- */
  ML.kernel = function (o, p) {
    const gam = o.gamma || 1 / p, deg = o.degree || 3;
    if (o.kernel === 'rbf') return (a, b) => { let s = 0; for (let j = 0; j < a.length; j++) s += (a[j] - b[j]) ** 2; return Math.exp(-gam * s); };
    if (o.kernel === 'poly') return (a, b) => { let s = 0; for (let j = 0; j < a.length; j++) s += a[j] * b[j]; return Math.pow(gam * s + 1, deg); };
    return (a, b) => { let s = 0; for (let j = 0; j < a.length; j++) s += a[j] * b[j]; return s; };
  };
  function svmBinary(Kx, s, C, maxPass) {          /* s: labels +1/-1 ; Kx: augmented kernel matrix */
    const n = s.length, a = new Array(n).fill(0), f = new Array(n).fill(0);   /* f_i = sum_j a_j s_j K_ij */
    const order = Array.from({ length: n }, (_, i) => i), r = ML.rng(12345);
    for (let pass = 0; pass < (maxPass || 2000); pass++) {
      r.shuffle(order); let viol = 0;
      for (const i of order) {
        const G = s[i] * f[i] - 1, Q = Kx[i][i];
        let pg = G; if (a[i] <= 0) pg = Math.min(G, 0); else if (a[i] >= C) pg = Math.max(G, 0);
        viol = Math.max(viol, Math.abs(pg));
        if (Math.abs(pg) < 1e-12) continue;
        const na = clamp(a[i] - G / Q, 0, C), d = (na - a[i]) * s[i];
        if (d) { for (let j = 0; j < n; j++) f[j] += d * Kx[i][j]; a[i] = na; }
      }
      if (viol < 1e-4) break;
    }
    return a;
  }
  /* Platt scaling: probability = 1 / (1 + exp(A f + B)), fitted on the training decision values */
  function platt(fv, t) {
    let A = 0, B = 0;
    const np = t.filter(v => v).length, nn = t.length - np, hi = (np + 1) / (np + 2), lo = 1 / (nn + 2);
    const tt = t.map(v => (v ? hi : lo));
    for (let it = 0; it < 100; it++) {
      let g1 = 0, g2 = 0, h11 = 1e-12, h22 = 1e-12, h21 = 0;
      fv.forEach((f, i) => { const p = sigmoid(-(A * f + B)), d = tt[i] - p, w = p * (1 - p); g1 += f * d; g2 += d; h11 += f * f * w; h22 += w; h21 += f * w; });
      const det = h11 * h22 - h21 * h21; if (Math.abs(det) < 1e-18) break;
      const dA = -(h22 * g1 - h21 * g2) / det, dB = -(-h21 * g1 + h11 * g2) / det;
      A += dA; B += dB; if (Math.abs(dA) + Math.abs(dB) < 1e-10) break;
    }
    return f => sigmoid(-(A * f + B));
  }
  CLS.svm = function (X, y, K, o) {
    const n = X.length, ker = ML.kernel(o, X[0].length), C = o.C || 1;
    const Kx = X.map(a => X.map(b => ker(a, b) + 1));
    const one = cls => { const s = y.map(v => (v === cls ? 1 : -1)), a = svmBinary(Kx, s, C); const sv = []; a.forEach((v, i) => { if (v > 1e-8) sv.push(i); });
      return { a, s, sv, f: x => sv.reduce((acc, i) => acc + a[i] * s[i] * (ker(X[i], x) + 1), 0) }; };
    if (K === 2) {
      const m = one(1), fv = X.map((x, i) => m.sv.reduce((acc, j) => acc + m.a[j] * m.s[j] * Kx[j][i], 0)), pr = platt(fv, y.map(v => v === 1));
      return { sv: m.sv, decision: x => m.f(x), proba: x => { const p = pr(m.f(x)); return [1 - p, p]; } };
    }
    const ms = Array.from({ length: K }, (_, k) => one(k)), sv = [...new Set(ms.flatMap(m => m.sv))];
    return { sv, decision: x => ms.map(m => m.f(x)), proba: x => { const d = ms.map(m => m.f(x)); let b = 0; for (let k = 1; k < K; k++) if (d[k] > d[b]) b = k; return d.map((_, k) => (k === b ? 1 : 0)); } };
  };

  /* ---------------------------------------------------------------
     neural network: hidden layers of neurons, softmax (classes) or
     linear (regression) output; trained by backpropagation with Adam.
     --------------------------------------------------------------- */
  const ACT = {
    relu: { f: z => (z > 0 ? z : 0), d: (z, a) => (z > 0 ? 1 : 0) },
    tanh: { f: z => Math.tanh(z), d: (z, a) => 1 - a * a },
    sigmoid: { f: z => sigmoid(z), d: (z, a) => a * (1 - a) }
  };
  ML.trainNet = function (X, Y, sizes, o) {      /* Y: rows of targets (one-hot or [y]); sizes: [in, h1, ..., out] */
    const r = ML.rng((o.seed || 1) * 17 + 5), act = ACT[o.act || 'relu'], out = o.out || 'softmax', L = sizes.length - 1;
    const W = [], b = [];
    for (let l = 0; l < L; l++) {
      const fan = sizes[l], sc = (o.act === 'relu' || !o.act) && l < L - 1 ? Math.sqrt(2 / fan) : Math.sqrt(1 / fan);
      W.push(Array.from({ length: sizes[l + 1] }, () => Array.from({ length: fan }, () => r.normal(0, sc))));
      b.push(new Array(sizes[l + 1]).fill(0));
    }
    const forward = x => {
      const A = [x], Z = [];
      for (let l = 0; l < L; l++) {
        const z = W[l].map((row, k) => row.reduce((s, w, j) => s + w * A[l][j], b[l][k]));
        Z.push(z);
        A.push(l < L - 1 ? z.map(act.f) : out === 'softmax' ? softmax(z) : z);
      }
      return { A, Z };
    };
    const mW = W.map(m => m.map(row => row.map(() => 0))), vW = W.map(m => m.map(row => row.map(() => 0)));
    const mb = b.map(v => v.map(() => 0)), vb = b.map(v => v.map(() => 0));
    const lr = o.lr || 0.02, b1 = 0.9, b2 = 0.999, l2 = o.l2 === undefined ? 1e-4 : o.l2, n = X.length, epochs = o.epochs || 800, hist = [];
    for (let ep = 1; ep <= epochs; ep++) {
      const gW = W.map(m => m.map(row => row.map(() => 0))), gb = b.map(v => v.map(() => 0));
      let loss = 0;
      for (let i = 0; i < n; i++) {
        const { A, Z } = forward(X[i]); const yhat = A[L];
        let d = yhat.map((v, k) => (v - Y[i][k]) / n);
        loss += out === 'softmax' ? -Math.log(Math.max(1e-12, yhat[Y[i].indexOf(1)])) / n : 0.5 * yhat.reduce((s, v, k) => s + (v - Y[i][k]) ** 2, 0) / n;
        for (let l = L - 1; l >= 0; l--) {
          for (let k = 0; k < d.length; k++) { gb[l][k] += d[k]; const row = gW[l][k]; for (let j = 0; j < row.length; j++) row[j] += d[k] * A[l][j]; }
          if (l > 0) { const nd = new Array(sizes[l]).fill(0); for (let k = 0; k < d.length; k++) for (let j = 0; j < nd.length; j++) nd[j] += W[l][k][j] * d[k]; d = nd.map((v, j) => v * act.d(Z[l - 1][j], A[l][j])); }
        }
      }
      const c1 = 1 - Math.pow(b1, ep), c2 = 1 - Math.pow(b2, ep);
      for (let l = 0; l < L; l++) {
        for (let k = 0; k < W[l].length; k++) {
          for (let j = 0; j < W[l][k].length; j++) {
            const g = gW[l][k][j] + l2 * W[l][k][j];
            mW[l][k][j] = b1 * mW[l][k][j] + (1 - b1) * g; vW[l][k][j] = b2 * vW[l][k][j] + (1 - b2) * g * g;
            W[l][k][j] -= lr * (mW[l][k][j] / c1) / (Math.sqrt(vW[l][k][j] / c2) + 1e-8);
          }
          const g = gb[l][k]; mb[l][k] = b1 * mb[l][k] + (1 - b1) * g; vb[l][k] = b2 * vb[l][k] + (1 - b2) * g * g;
          b[l][k] -= lr * (mb[l][k] / c1) / (Math.sqrt(vb[l][k] / c2) + 1e-8);
        }
      }
      if (ep === 1 || ep % 10 === 0) hist.push([ep, loss]);
    }
    return { W, b, hist, run: x => forward(x).A[L] };
  };
  CLS.ann = function (X, y, K, o) {
    const sizes = [X[0].length].concat(new Array(o.layers || 1).fill(o.neurons || 8), [K]);
    const net = ML.trainNet(X, y.map(v => Array.from({ length: K }, (_, k) => (k === v ? 1 : 0))), sizes, { act: o.act, seed: o.seed, epochs: o.epochs || 600, lr: 0.02 });
    return { hist: net.hist, proba: x => net.run(x) };
  };
  ML.CLS = CLS;

  /* =================================================================
     regression models (one input): fit(X, y, opts) -> { predict(x) }
     X standardized by the caller; y in its own units.
     ================================================================= */
  const REG = {};
  REG.linear = function (X, y) {
    const n = X.length, mx = X.reduce((s, x) => s + x[0], 0) / n, my = y.reduce((s, v) => s + v, 0) / n;
    let sxy = 0, sxx = 0; X.forEach((x, i) => { sxy += (x[0] - mx) * (y[i] - my); sxx += (x[0] - mx) ** 2; });
    const b = sxx ? sxy / sxx : 0, a = my - b * mx;
    return { a, b, predict: x => a + b * x[0] };
  };
  REG.knn = function (X, y, o) { const k = o.k || 5; return { predict: x => { const nb = nearest(X, x, k); return nb.reduce((s, i) => s + y[i], 0) / nb.length; } }; };
  REG.tree = function (X, y, o) { const root = buildTree(X, y, X.map((_, i) => i), { maxDepth: o.depth || 3 }, 0, ML.rng(1)); return { root, predict: x => treeLeaf(root, x).value }; };
  REG.forest = function (X, y, o) {
    const r = ML.rng((o.seed || 1) * 31 + 7), n = X.length, trees = [];
    for (let t = 0; t < (o.trees || 50); t++) { const rows = Array.from({ length: n }, () => Math.floor(r.random() * n)); trees.push(buildTree(X, y, rows, { maxDepth: o.depth || 6 }, 0, r)); }
    return { trees, predict: x => trees.reduce((s, tr) => s + treeLeaf(tr, x).value, 0) / trees.length };
  };
  REG.xgb = function (X, y, o) {
    const n = X.length, eta = o.eta || 0.3, base = y.reduce((s, v) => s + v, 0) / n, F = new Array(n).fill(base), trees = [], rows = X.map((_, i) => i);
    for (let t = 0; t < (o.rounds || 50); t++) {
      const g = F.map((f, i) => f - y[i]), h = new Array(n).fill(1), tr = xgbTree(X, g, h, rows, { maxDepth: o.depth || 3 }, 0);
      trees.push(tr); for (let i = 0; i < n; i++) F[i] += eta * xgbOut(tr, X[i]);
    }
    return { base, trees, predict: x => base + trees.reduce((s, tr) => s + eta * xgbOut(tr, x), 0) };
  };
  /* support vector regression: errors inside the band of +/- epsilon cost nothing.
     min 1/2 b'Kb + eps |b|_1 - y'b, -C <= b_i <= C  (dual, bias in the kernel), on standardized y */
  REG.svr = function (X, y, o) {
    const n = X.length, my = y.reduce((s, v) => s + v, 0) / n, sy = Math.sqrt(y.reduce((s, v) => s + (v - my) ** 2, 0) / n) || 1;
    const t = y.map(v => (v - my) / sy), eps = (o.eps === undefined ? 0.5 : o.eps) / sy, C = o.C || 1, ker = ML.kernel(o, 1);
    const Kx = X.map(a => X.map(b => ker(a, b) + 1)), beta = new Array(n).fill(0), f = new Array(n).fill(0), r = ML.rng(4321), order = X.map((_, i) => i);
    for (let pass = 0; pass < 3000; pass++) {
      r.shuffle(order); let ch = 0;
      for (const i of order) {
        const Q = Kx[i][i], gi = f[i] - Q * beta[i] - t[i], z = -gi;
        const nb = clamp(Math.sign(z) * Math.max(0, Math.abs(z) - eps) / Q, -C, C), d = nb - beta[i];
        if (d) { for (let j = 0; j < n; j++) f[j] += d * Kx[i][j]; beta[i] = nb; ch = Math.max(ch, Math.abs(d)); }
      }
      if (ch < 1e-7) break;
    }
    const sv = []; beta.forEach((v, i) => { if (Math.abs(v) > 1e-8) sv.push(i); });
    return { sv, eps: eps * sy, predict: x => my + sy * sv.reduce((s, i) => s + beta[i] * (ker(X[i], x) + 1), 0) };
  };
  REG.ann = function (X, y, o) {
    const n = y.length, my = y.reduce((s, v) => s + v, 0) / n, sy = Math.sqrt(y.reduce((s, v) => s + (v - my) ** 2, 0) / n) || 1;
    const sizes = [1].concat(new Array(o.layers || 1).fill(o.neurons || 8), [1]);
    const net = ML.trainNet(X, y.map(v => [(v - my) / sy]), sizes, { act: o.act, seed: o.seed, out: 'linear', epochs: o.epochs || 1500, lr: 0.02 });
    return { hist: net.hist, predict: x => my + sy * net.run(x)[0] };
  };
  ML.REG = REG;

  /* =================================================================
     clustering
     ================================================================= */
  /* k-means, one step at a time: state { centers, labels, iter, moved } */
  ML.kmeansInit = function (X, k, seed) {
    const idx = ML.rng(seed * 11 + 1).shuffle(X.map((_, i) => i)).slice(0, k);
    return { centers: idx.map(i => X[i].slice()), labels: null, iter: 0, done: false };
  };
  ML.kmeansAssign = (X, C) => X.map(x => { let b = 0, bd = Infinity; C.forEach((c, k) => { const d = (x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2; if (d < bd - 1e-15) { bd = d; b = k; } }); return b; });
  ML.kmeansStep = function (X, st) {
    const labels = ML.kmeansAssign(X, st.centers), k = st.centers.length;
    const sum = Array.from({ length: k }, () => [0, 0, 0]);
    X.forEach((x, i) => { sum[labels[i]][0] += x[0]; sum[labels[i]][1] += x[1]; sum[labels[i]][2]++; });
    const centers = st.centers.map((c, j) => (sum[j][2] ? [sum[j][0] / sum[j][2], sum[j][1] / sum[j][2]] : c.slice()));
    const same = st.labels && labels.every((v, i) => v === st.labels[i]);
    return { centers, labels, iter: st.iter + 1, done: !!same, prev: st.centers };
  };
  ML.wcss = (X, labels, centers) => X.reduce((s, x, i) => s + (x[0] - centers[labels[i]][0]) ** 2 + (x[1] - centers[labels[i]][1]) ** 2, 0);

  /* agglomerative (hierarchical) clustering, Lance-Williams updates.
     Returns merges [a, b, height, size] with scipy's numbering (new cluster = n + step). */
  ML.hclust = function (X, linkage) {
    const n = X.length, D = X.map(a => X.map(b => { let s = 0; for (let j = 0; j < a.length; j++) s += (a[j] - b[j]) ** 2; return Math.sqrt(s); }));
    const ward = linkage === 'ward', active = new Map(), size = {}, merges = [];
    for (let i = 0; i < n; i++) { active.set(i, i); size[i] = 1; }
    const dist = {}; const key = (a, b) => (a < b ? a + ',' + b : b + ',' + a);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) dist[key(i, j)] = ward ? D[i][j] : D[i][j];
    let ids = Array.from({ length: n }, (_, i) => i);
    for (let step = 0; step < n - 1; step++) {
      let best = null;
      for (let x = 0; x < ids.length; x++) for (let y = x + 1; y < ids.length; y++) { const d = dist[key(ids[x], ids[y])]; if (!best || d < best[2] - 1e-12) best = [ids[x], ids[y], d]; }
      const [a, b, h] = best, c = n + step; size[c] = size[a] + size[b];
      merges.push([Math.min(a, b), Math.max(a, b), h, size[c]]);
      ids = ids.filter(v => v !== a && v !== b);
      for (const q of ids) {
        const da = dist[key(a, q)], db = dist[key(b, q)], na = size[a], nb = size[b], nq = size[q];
        let d;
        if (linkage === 'single') d = Math.min(da, db);
        else if (linkage === 'complete') d = Math.max(da, db);
        else if (ward) d = Math.sqrt(((na + nq) * da * da + (nb + nq) * db * db - nq * h * h) / (na + nb + nq));
        else d = (na * da + nb * db) / (na + nb);
        dist[key(c, q)] = d;
      }
      ids.push(c);
    }
    return merges;
  };
  /* cut the tree into k clusters: undo the last k-1 merges */
  ML.hcut = function (merges, n, k) {
    const parent = {}; const find = v => (parent[v] === undefined ? v : (parent[v] = find(parent[v])));
    merges.slice(0, n - k).forEach((m, s) => { parent[m[0]] = n + s; parent[m[1]] = n + s; });
    const roots = {}, labels = [];
    for (let i = 0; i < n; i++) { const r = find(i); if (roots[r] === undefined) roots[r] = Object.keys(roots).length; labels.push(roots[r]); }
    return labels;
  };

  /* =================================================================
     metrics
     ================================================================= */
  ML.confusion = function (yt, yp, K) { const M = Array.from({ length: K }, () => new Array(K).fill(0)); yt.forEach((v, i) => M[v][yp[i]]++); return M; };
  /* binary: positive class pos (default 1). Rows = actual, columns = predicted. */
  ML.binaryMetrics = function (M, pos) {
    pos = pos === undefined ? 1 : pos; const neg = 1 - pos;
    const TP = M[pos][pos], FN = M[pos][neg], FP = M[neg][pos], TN = M[neg][neg], n = TP + FN + FP + TN;
    const div = (a, b) => (b ? a / b : NaN);
    const sens = div(TP, TP + FN), spec = div(TN, TN + FP), prec = div(TP, TP + FP);
    return { TP, FN, FP, TN, n, acc: div(TP + TN, n), mis: div(FP + FN, n), sens, spec, fa: div(FP, FP + TN), prec, f1: div(2 * prec * sens, prec + sens) };
  };
  ML.multiMetrics = function (M) {
    const K = M.length, n = M.flat().reduce((a, b) => a + b, 0); let diag = 0; for (let k = 0; k < K; k++) diag += M[k][k];
    const per = M.map((row, k) => { const act = row.reduce((a, b) => a + b, 0), pred = M.reduce((s, r) => s + r[k], 0), sens = act ? M[k][k] / act : NaN, prec = pred ? M[k][k] / pred : NaN; return { sens, prec, f1: 2 * prec * sens / (prec + sens), n: act }; });
    return { n, acc: n ? diag / n : NaN, mis: n ? 1 - diag / n : NaN, per };
  };
  /* ROC: points (false alarm rate, sensitivity) for every threshold, and the area under it */
  ML.roc = function (score, isPos) {
    const idx = score.map((s, i) => i).sort((a, b) => score[b] - score[a]);
    const P = isPos.filter(v => v).length, N = isPos.length - P, pts = [[0, 0, Infinity]];
    let tp = 0, fp = 0, auc = 0;
    for (let t = 0; t < idx.length;) {
      const s = score[idx[t]]; let u = t;
      while (u < idx.length && score[idx[u]] === s) { if (isPos[idx[u]]) tp++; else fp++; u++; }
      const prev = pts[pts.length - 1], x = N ? fp / N : 0, y = P ? tp / P : 0;
      auc += (x - prev[0]) * (y + prev[1]) / 2; pts.push([x, y, s]); t = u;
    }
    return { pts, auc: P && N ? auc : NaN };
  };
  ML.regMetrics = function (y, f) {
    const n = y.length; if (!n) return null;
    let se = 0, ae = 0, pe = 0, ape = 0, okp = 0; const my = y.reduce((s, v) => s + v, 0) / n; let st = 0;
    y.forEach((v, i) => { const e = v - f[i]; se += e * e; ae += Math.abs(e); st += (v - my) ** 2; if (v !== 0) { pe += e / v; ape += Math.abs(e / v); okp++; } });
    return { n, mse: se / n, rmse: Math.sqrt(se / n), mae: ae / n, mpe: okp ? 100 * pe / okp : NaN, mape: okp ? 100 * ape / okp : NaN, r2: st > 0 ? 1 - se / st : NaN };
  };


  /* =================================================================
     classification batch (4MLSim build 11)
     ================================================================= */
  /* one-input data sets for the logistic regression module: x -> 0/1 */
  ML.DATA1 = {
    banks: { label: 'Bank failures', xname: 'Debt-to-equity ratio', yname: 'Failed', classes: ['Healthy', 'Failed'], unit: 'banks', n: 200, share: 0.3,
      note: 'US-style bank data: the more debt a bank carries per unit of equity, the more likely it fails.',
      gen(r, n, sh) { const X = [], y = []; for (let i = 0; i < n; i++) { const f = r.random() < sh ? 1 : 0; X.push(round(clamp(f ? r.normal(6.2, 1.7) : r.normal(3.6, 1.4), 0.2, 11), 2)); y.push(f); } return { x: X, y }; } },
    recession: { label: 'Recessions and the yield curve', xname: 'Term spread (10-year minus 3-month yield, points)', yname: 'Recession', classes: ['No recession', 'Recession'], unit: 'months', n: 240, share: 0.15,
      note: 'Monthly data: when long-term yields fall below short-term yields (a negative spread, an "inverted" yield curve), a recession within the next 12 months becomes more likely.',
      gen(r, n, sh) { const X = [], y = []; for (let i = 0; i < n; i++) { const f = r.random() < sh ? 1 : 0; X.push(round(f ? r.normal(-0.1, 0.9) : r.normal(1.8, 1.1), 2)); y.push(f); } return { x: X, y }; } },
    loans: { label: 'Loan defaults', xname: 'Debt-to-income ratio (%)', yname: 'Default', classes: ['Repaid', 'Default'], unit: 'loans', n: 300, share: 0.2,
      note: 'Past loans of a bank: borrowers who spend a larger share of their income on debt payments default more often.',
      gen(r, n, sh) { const X = [], y = []; for (let i = 0; i < n; i++) { const f = r.random() < sh ? 1 : 0; X.push(round(clamp(f ? r.normal(45, 11) : r.normal(31, 10), 2, 90), 1)); y.push(f); } return { x: X, y }; } }
  };

  /* logistic regression by Newton's method (maximum likelihood, no penalty), on raw inputs.
     X: rows without the constant. Returns every iteration, standard errors and log-likelihoods
     (as statsmodels Logit). */
  ML.logitNewton = function (X, y, maxIt) {
    const n = X.length, F = X.map(x => [1].concat(x)), p = F[0].length;
    const ll = b => F.reduce((s, f, i) => { const z = f.reduce((a, v, j) => a + v * b[j], 0); return s + (y[i] ? -Math.log1p(Math.exp(-z)) : -Math.log1p(Math.exp(z))); }, 0);
    let b = new Array(p).fill(0); const path = [{ b: b.slice(), ll: ll(b) }]; let H = null, sep = false;
    for (let it = 0; it < (maxIt || 30); it++) {
      const g = new Array(p).fill(0); H = Array.from({ length: p }, () => new Array(p).fill(0));
      F.forEach((f, i) => { const pr = sigmoid(f.reduce((s, v, j) => s + v * b[j], 0)), w = pr * (1 - pr); for (let j = 0; j < p; j++) { g[j] += (y[i] - pr) * f[j]; for (let k = 0; k < p; k++) H[j][k] += w * f[j] * f[k]; } });
      const step = solve(H, g); if (!step) { sep = true; break; }
      b = b.map((v, j) => v + step[j]);
      path.push({ b: b.slice(), ll: ll(b) });
      if (Math.max(...step.map(Math.abs)) < 1e-10) break;
      if (Math.max(...b.map(Math.abs)) > 1e4) { sep = true; break; }
    }
    /* covariance = inverse information at the estimate */
    const I = Array.from({ length: p }, () => new Array(p).fill(0));
    F.forEach(f => { const pr = sigmoid(f.reduce((s, v, j) => s + v * b[j], 0)), w = pr * (1 - pr); for (let j = 0; j < p; j++) for (let k = 0; k < p; k++) I[j][k] += w * f[j] * f[k]; });
    const cov = Array.from({ length: p }, (_, j) => solve(I, Array.from({ length: p }, (_, k) => (k === j ? 1 : 0))) || new Array(p).fill(NaN));
    const se = cov.map((c, j) => Math.sqrt(c[j]));
    const ybar = y.reduce((a, v) => a + v, 0) / n, ll0 = n * (ybar > 0 && ybar < 1 ? ybar * Math.log(ybar) + (1 - ybar) * Math.log(1 - ybar) : 0);
    return { b, se, cov, path, ll: path[path.length - 1].ll, ll0, separated: sep, prob: x => sigmoid([1].concat(x).reduce((s, v, j) => s + v * b[j], 0)) };
  };

  /* Gaussian naive Bayes (as scikit-learn GaussianNB: variances + 1e-9 x the largest input variance) */
  ML.gnb = function (X, y, K, o) {
    o = o || {}; const n = X.length, p = X[0].length;
    const cnt = new Array(K).fill(0); y.forEach(v => cnt[v]++);
    const prior = o.equalPriors ? new Array(K).fill(1 / K) : cnt.map(c => c / n);
    const mu = Array.from({ length: K }, () => new Array(p).fill(0)), va = Array.from({ length: K }, () => new Array(p).fill(0));
    X.forEach((x, i) => x.forEach((v, j) => (mu[y[i]][j] += v / cnt[y[i]])));
    X.forEach((x, i) => x.forEach((v, j) => (va[y[i]][j] += (v - mu[y[i]][j]) ** 2 / cnt[y[i]])));
    const all = Array.from({ length: p }, (_, j) => { const m = X.reduce((s, x) => s + x[j], 0) / n; return X.reduce((s, x) => s + (x[j] - m) ** 2, 0) / n; });
    const eps = 1e-9 * Math.max(...all);
    for (let k = 0; k < K; k++) for (let j = 0; j < p; j++) va[k][j] += eps;
    const dens = (k, j, v) => Math.exp(-((v - mu[k][j]) ** 2) / (2 * va[k][j])) / Math.sqrt(2 * Math.PI * va[k][j]);
    const parts = x => { const like = Array.from({ length: K }, (_, k) => x.map((v, j) => dens(k, j, v))), joint = like.map((l, k) => prior[k] * l.reduce((a, b) => a * b, 1)); const tot = joint.reduce((a, b) => a + b, 0);
      /* log space for the posterior, so far-away points do not give 0/0 */
      const lj = Array.from({ length: K }, (_, k) => Math.log(prior[k]) + x.reduce((s, v, j) => s - 0.5 * Math.log(2 * Math.PI * va[k][j]) - (v - mu[k][j]) ** 2 / (2 * va[k][j]), 0));
      return { like, joint, post: softmax(lj), tot }; };
    return { prior, mu, va, dens, parts, proba: x => parts(x).post };
  };

  /* precision-recall curve and average precision (as scikit-learn average_precision_score) */
  ML.pr = function (score, isPos) {
    const idx = score.map((s, i) => i).sort((a, b) => score[b] - score[a]);
    const P = isPos.filter(v => v).length, pts = []; let tp = 0, fp = 0, ap = 0, prevR = 0;
    for (let t = 0; t < idx.length;) {
      const s = score[idx[t]]; let u = t;
      while (u < idx.length && score[idx[u]] === s) { if (isPos[idx[u]]) tp++; else fp++; u++; }
      const rec = P ? tp / P : 0, prec = tp / (tp + fp);
      ap += (rec - prevR) * prec; prevR = rec; pts.push([rec, prec, s]); t = u;
    }
    return { pts, ap: P ? ap : NaN, base: P / isPos.length };
  };
  /* calibration: equal-width bins of the predicted probability (as scikit-learn calibration_curve, empty bins dropped) */
  ML.calibration = function (prob, y, bins) {
    const B = bins || 10, out = [], step = 1 / B, bin = q => { let c = 0; for (let e = 1; e < B; e++) if (e * step < q) c++; return c; };   /* bins (lo, hi], the first [0, hi] */
    const which = prob.map(bin);
    for (let b = 0; b < B; b++) {
      const lo = b / B, hi = (b + 1) / B, idx = prob.map((p, i) => i).filter(i => which[i] === b);
      if (idx.length) out.push({ lo, hi, n: idx.length, meanP: idx.reduce((s, i) => s + prob[i], 0) / idx.length, frac: idx.reduce((s, i) => s + y[i], 0) / idx.length });
    }
    return out;
  };
  ML.brier = (prob, y) => prob.reduce((s, p, i) => s + (p - y[i]) ** 2, 0) / prob.length;
  /* accuracy of KNN in 4-fold cross-validation on the training set (folds = consecutive blocks) */
  ML.knnCV = function (X, y, K, k, folds, near) {
    near = near || nearest;
    const n = X.length, f = folds || 4, size = n / f; let ok = 0;
    for (let q = 0; q < f; q++) {
      const lo = Math.round(q * size), hi = Math.round((q + 1) * size), tr = [], te = [];
      for (let i = 0; i < n; i++) (i >= lo && i < hi ? te : tr).push(i);
      const Xt = tr.map(i => X[i]);
      te.forEach(i => { const nb = near(Xt, X[i], Math.min(k, Xt.length)), c = new Array(K).fill(0); nb.forEach(j => c[y[tr[j]]]++); let b = 0; for (let m = 1; m < K; m++) if (c[m] > c[b]) b = m; if (b === y[i]) ok++; });
    }
    return ok / n;
  };
  ML.manhattanNearest = function (X, x, k) {
    const d = X.map((r, i) => [r.reduce((s, v, j) => s + Math.abs(v - x[j]), 0), i]); d.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    return d.slice(0, Math.min(k, d.length)).map(v => v[1]);
  };


  /* =================================================================
     SVM & trees batch (4MLSim build 12)
     ================================================================= */
  const entropy = c => { const n = c.reduce((a, b) => a + b, 0); return n ? -c.reduce((s, v) => (v ? s + (v / n) * Math.log2(v / n) : s), 0) : 0; };
  ML.entropy = entropy;
  /* CART tree (as scikit-learn): criterion 'gini' | 'entropy' (classification, K classes) or 'mse' (regression).
     o: { K, crit, maxDepth, minLeaf, maxFeatures, rng }. Every node keeps its rows, impurity and a breadth-first id. */
  ML.cart = function (X, y, rows, o) {
    const K = o.K, reg = !K, imp = c => (o.crit === 'entropy' ? entropy(c) : gini(c)), minLeaf = o.minLeaf || 1, p = X[0].length;
    function node(rows, depth) {
      const n = rows.length, nd = { n, depth, rows };
      if (reg) { nd.value = rows.reduce((s, i) => s + y[i], 0) / n; nd.imp = rows.reduce((s, i) => s + (y[i] - nd.value) ** 2, 0) / n; }
      else { nd.counts = new Array(K).fill(0); rows.forEach(i => nd.counts[y[i]]++); nd.imp = imp(nd.counts); nd.value = nd.counts.map(c => c / n); }
      return nd;
    }
    function best(nd) {
      const rows = nd.rows, n = rows.length; let b = null;
      let feats = Array.from({ length: p }, (_, j) => j);
      if (o.maxFeatures && o.maxFeatures < p) feats = o.rng.shuffle(feats.slice()).slice(0, o.maxFeatures);
      for (const j of feats) {
        const ord = rows.slice().sort((a, c) => X[a][j] - X[c][j] || a - c);
        if (reg) {
          let sl = 0, sl2 = 0; const st = ord.reduce((s, i) => s + y[i], 0), st2 = ord.reduce((s, i) => s + y[i] * y[i], 0);
          for (let t = 0; t < n - 1; t++) { const v = y[ord[t]]; sl += v; sl2 += v * v; const a = X[ord[t]][j], c = X[ord[t + 1]][j]; if (a === c) continue; const nl = t + 1, nr = n - nl; if (nl < minLeaf || nr < minLeaf) continue;
            const sc = (sl2 - sl * sl / nl) + ((st2 - sl2) - (st - sl) ** 2 / nr); if (!b || sc < b.score - 1e-12) b = { j, thr: (a + c) / 2, score: sc }; }
        } else {
          const L = new Array(K).fill(0), R = nd.counts.slice();
          for (let t = 0; t < n - 1; t++) { const c0 = y[ord[t]]; L[c0]++; R[c0]--; const a = X[ord[t]][j], c = X[ord[t + 1]][j]; if (a === c) continue; const nl = t + 1, nr = n - nl; if (nl < minLeaf || nr < minLeaf) continue;
            const sc = nl * imp(L) + nr * imp(R); if (!b || sc < b.score - 1e-12) b = { j, thr: (a + c) / 2, score: sc }; }
        }
      }
      return b;
    }
    const root = node(rows, 0), queue = [root], order = [];
    while (queue.length) {                       /* breadth first, so the splits can be shown one at a time */
      const nd = queue.shift();
      if (nd.depth >= o.maxDepth || nd.n < 2 * minLeaf || nd.imp < 1e-12) continue;
      const b = best(nd); if (!b) continue;
      nd.feat = b.j; nd.thr = b.thr; nd.split = order.length; order.push(nd);
      nd.left = node(nd.rows.filter(i => X[i][b.j] <= b.thr), nd.depth + 1); nd.right = node(nd.rows.filter(i => X[i][b.j] > b.thr), nd.depth + 1);
      nd.gain = nd.n * nd.imp - nd.left.n * nd.left.imp - nd.right.n * nd.right.imp;    /* weighted impurity decrease */
      queue.push(nd.left, nd.right);
    }
    return { root, order };
  };
  /* leaf reached by x when only the first `upto` splits are made */
  ML.cartLeaf = (nd, x, upto) => { while (nd.left && (upto === undefined || nd.split < upto)) nd = x[nd.feat] <= nd.thr ? nd.left : nd.right; return nd; };
  /* impurity after splitting a node at every threshold of input j (for "choosing the question") */
  ML.splitCurve = function (X, y, rows, j, K, crit) {
    const n = rows.length, ord = rows.slice().sort((a, c) => X[a][j] - X[c][j] || a - c), out = [];
    if (!K) { let sl = 0, sl2 = 0; const st = ord.reduce((s, i) => s + y[i], 0), st2 = ord.reduce((s, i) => s + y[i] * y[i], 0);
      for (let t = 0; t < n - 1; t++) { const v = y[ord[t]]; sl += v; sl2 += v * v; const a = X[ord[t]][j], c = X[ord[t + 1]][j]; if (a === c) continue; const nl = t + 1, nr = n - nl; out.push([(a + c) / 2, ((sl2 - sl * sl / nl) + ((st2 - sl2) - (st - sl) ** 2 / nr)) / n]); } return out; }
    const imp = c => (crit === 'entropy' ? entropy(c) : gini(c)), L = new Array(K).fill(0), R = new Array(K).fill(0); rows.forEach(i => R[y[i]]++);
    for (let t = 0; t < n - 1; t++) { const c0 = y[ord[t]]; L[c0]++; R[c0]--; const a = X[ord[t]][j], c = X[ord[t + 1]][j]; if (a === c) continue; const nl = t + 1, nr = n - nl; out.push([(a + c) / 2, (nl * imp(L) + nr * imp(R)) / n]); }
    return out;
  };

  /* random forest (as scikit-learn): bootstrap rows, maxFeatures inputs tried per split, out-of-bag predictions,
     impurity-based importances; classification probability = average of the trees' leaf shares, vote = majority */
  ML.forest = function (X, y, o) {
    const r = ML.rng((o.seed || 1) * 31 + 7), n = X.length, p = X[0].length, trees = [], K = o.K;
    for (let t = 0; t < (o.trees || 100); t++) {
      const rows = Array.from({ length: n }, () => Math.floor(r.random() * n)), inBag = new Set(rows);
      const tr = ML.cart(X, y, rows, { K, crit: o.crit || (K ? 'gini' : 'mse'), maxDepth: o.maxDepth || 20, maxFeatures: o.maxFeatures || p, rng: r });
      tr.rows = rows; tr.oob = X.map((_, i) => i).filter(i => !inBag.has(i));
      trees.push(tr);
    }
    const leaf = (tr, x) => ML.cartLeaf(tr.root, x).value;
    const imp = new Array(p).fill(0);
    trees.forEach(tr => { const a = new Array(p).fill(0); tr.order.forEach(nd => (a[nd.feat] += nd.gain)); const s = a.reduce((u, v) => u + v, 0); if (s > 0) a.forEach((v, j) => (imp[j] += v / s / trees.length)); });
    const predM = (x, m) => { const ts = trees.slice(0, m || trees.length); if (!K) return ts.reduce((s, tr) => s + leaf(tr, x), 0) / ts.length; const v = new Array(K).fill(0); ts.forEach(tr => leaf(tr, x).forEach((q, k) => (v[k] += q / ts.length))); return v; };
    return { trees, importance: imp, leaf, predict: predM };
  };

  /* support vector classifier by SMO (as LIBSVM / scikit-learn SVC): labels +1/-1, free bias.
     Returns alpha, rho (decision = sum alpha_i y_i K(x_i, x) - rho) and the support vectors. */
  ML.svc = function (X, s, C, ker, eps) {
    const n = X.length, TAU = 1e-12, K = X.map(a => X.map(b => ker(a, b))), QD = K.map((r, i) => r[i]);
    const a = new Array(n).fill(0), G = new Array(n).fill(-1), ub = i => a[i] >= C, lb = i => a[i] <= 0; eps = eps || 1e-5;
    for (let it = 0; it < 100000; it++) {
      let Gmax = -Infinity, i = -1;
      for (let t = 0; t < n; t++) { if (s[t] === 1) { if (!ub(t) && -G[t] >= Gmax) { Gmax = -G[t]; i = t; } } else if (!lb(t) && G[t] >= Gmax) { Gmax = G[t]; i = t; } }
      let Gmax2 = -Infinity, j = -1, objMin = Infinity;
      for (let t = 0; t < n; t++) {
        const Qit = i >= 0 ? s[i] * s[t] * K[i][t] : 0;
        if (s[t] === 1) { if (!lb(t)) { const gd = Gmax + G[t]; if (G[t] >= Gmax2) Gmax2 = G[t]; if (gd > 0) { const q = QD[i] + QD[t] - 2 * s[i] * Qit, ob = -(gd * gd) / (q > 0 ? q : TAU); if (ob <= objMin) { j = t; objMin = ob; } } } }
        else if (!ub(t)) { const gd = Gmax - G[t]; if (-G[t] >= Gmax2) Gmax2 = -G[t]; if (gd > 0) { const q = QD[i] + QD[t] + 2 * s[i] * Qit, ob = -(gd * gd) / (q > 0 ? q : TAU); if (ob <= objMin) { j = t; objMin = ob; } } }
      }
      if (Gmax + Gmax2 < eps || j < 0) break;
      const ai = a[i], aj = a[j], Qij = s[i] * s[j] * K[i][j];
      if (s[i] !== s[j]) {
        let q = QD[i] + QD[j] + 2 * Qij; if (q <= 0) q = TAU;
        const d = (-G[i] - G[j]) / q, diff = a[i] - a[j]; a[i] += d; a[j] += d;
        if (diff > 0) { if (a[j] < 0) { a[j] = 0; a[i] = diff; } } else if (a[i] < 0) { a[i] = 0; a[j] = -diff; }
        if (diff > 0) { if (a[i] > C) { a[i] = C; a[j] = C - diff; } } else if (a[j] > C) { a[j] = C; a[i] = C + diff; }
      } else {
        let q = QD[i] + QD[j] - 2 * Qij; if (q <= 0) q = TAU;
        const d = (G[i] - G[j]) / q, sum = a[i] + a[j]; a[i] -= d; a[j] += d;
        if (sum > C) { if (a[i] > C) { a[i] = C; a[j] = sum - C; } } else if (a[j] < 0) { a[j] = 0; a[i] = sum; }
        if (sum > C) { if (a[j] > C) { a[j] = C; a[i] = sum - C; } } else if (a[i] < 0) { a[i] = 0; a[j] = sum; }
      }
      const di = a[i] - ai, dj = a[j] - aj;
      for (let t = 0; t < n; t++) G[t] += s[i] * s[t] * K[i][t] * di + s[j] * s[t] * K[j][t] * dj;
    }
    let up = Infinity, lo = -Infinity, nf = 0, sf = 0;
    for (let t = 0; t < n; t++) { const yG = s[t] * G[t]; if (ub(t)) { if (s[t] === -1) up = Math.min(up, yG); else lo = Math.max(lo, yG); } else if (lb(t)) { if (s[t] === 1) up = Math.min(up, yG); else lo = Math.max(lo, yG); } else { nf++; sf += yG; } }
    const rho = nf ? sf / nf : (up + lo) / 2, sv = [];
    a.forEach((v, t) => { if (v > 1e-10) sv.push(t); });
    return { alpha: a, rho, sv, C, f: x => sv.reduce((acc, t) => acc + a[t] * s[t] * ker(X[t], x), 0) - rho };
  };

  /* gradient boosting (as scikit-learn GradientBoostingRegressor / binary Classifier with criterion squared_error):
     each round fits a regression tree to the residuals; F_m = F_(m-1) + eta x tree. Keeps every round. */
  ML.gbm = function (X, y, o) {
    const n = X.length, eta = o.eta || 0.1, cls = !!o.classify, rows = X.map((_, i) => i);
    const pbar = cls ? y.reduce((s, v) => s + v, 0) / n : 0;
    const F0 = cls ? Math.log(pbar / (1 - pbar)) : y.reduce((s, v) => s + v, 0) / n, F = new Array(n).fill(F0), trees = [];
    for (let m = 0; m < (o.rounds || 100); m++) {
      const pr = cls ? F.map(sigmoid) : null, r = y.map((v, i) => (cls ? v - pr[i] : v - F[i]));
      const t = ML.cart(X, r, rows, { maxDepth: o.depth || 3, minLeaf: 1 });
      /* leaf values: the mean residual (regression) or one Newton step on the log-loss (classification) */
      const leaves = []; (function walk(nd) { if (nd.left) { walk(nd.left); walk(nd.right); } else leaves.push(nd); })(t.root);
      leaves.forEach(lf => { if (cls) { let num = 0, den = 0; lf.rows.forEach(i => { num += r[i]; den += pr[i] * (1 - pr[i]); }); lf.w = Math.abs(den) < 1e-150 ? 0 : num / den; } else lf.w = lf.value; });
      t.resid = r.slice();
      trees.push(t);
      for (let i = 0; i < n; i++) F[i] += eta * ML.cartLeaf(t.root, X[i]).w;
    }
    const score = (x, m) => trees.slice(0, m === undefined ? trees.length : m).reduce((s, t) => s + eta * ML.cartLeaf(t.root, x).w, F0);
    return { F0, eta, trees, score, predict: (x, m) => (cls ? sigmoid(score(x, m)) : score(x, m)) };
  };

  /* XGBoost (as xgboost, exact greedy): gradients g and curvatures h, leaf weight -G/(H + lambda),
     gain = G_L^2/(H_L+l) + G_R^2/(H_R+l) - G^2/(H+l); splits with gain below gamma are pruned bottom-up. */
  ML.xgboost = function (X, y, o) {
    const n = X.length, eta = o.eta === undefined ? 0.3 : o.eta, lam = o.lambda === undefined ? 1 : o.lambda, gam = o.gamma || 0, mcw = o.minChild === undefined ? 1 : o.minChild, cls = !!o.classify;
    const base = o.base !== undefined ? o.base : cls ? 0.5 : y.reduce((s, v) => s + v, 0) / n;
    const F0 = cls ? Math.log(base / (1 - base)) : base, F = new Array(n).fill(F0), trees = [], rows = X.map((_, i) => i);
    function grow(g, h, rows, depth) {
      let G = 0, H = 0; rows.forEach(i => { G += g[i]; H += h[i]; });
      const nd = { G, H, n: rows.length, w: -G / (H + lam), depth, rows };
      if (depth >= (o.depth || 3)) return nd;
      let b = null; const root = G * G / (H + lam);
      for (let j = 0; j < X[0].length; j++) {
        const ord = rows.slice().sort((a, c) => X[a][j] - X[c][j] || a - c); let gl = 0, hl = 0;
        for (let t = 0; t < ord.length - 1; t++) { gl += g[ord[t]]; hl += h[ord[t]]; const a = X[ord[t]][j], c = X[ord[t + 1]][j]; if (a === c) continue; const gr = G - gl, hr = H - hl; if (hl < mcw || hr < mcw) continue;
          const gain = gl * gl / (hl + lam) + gr * gr / (hr + lam) - root; if (gain > 1e-12 && (!b || gain > b.gain)) b = { j, thr: (a + c) / 2, gain }; }
      }
      if (!b) return nd;
      nd.feat = b.j; nd.thr = b.thr; nd.gain = b.gain;
      nd.left = grow(g, h, rows.filter(i => X[i][b.j] < b.thr), depth + 1); nd.right = grow(g, h, rows.filter(i => X[i][b.j] >= b.thr), depth + 1);
      return nd;
    }
    function prune(nd) { if (!nd.left) return nd; prune(nd.left); prune(nd.right); if (!nd.left.left && !nd.right.left && nd.gain < gam) { nd.pruned = { feat: nd.feat, thr: nd.thr, gain: nd.gain }; delete nd.left; delete nd.right; delete nd.feat; } return nd; }
    const out = (nd, x) => { while (nd.left) nd = x[nd.feat] < nd.thr ? nd.left : nd.right; return nd.w; };
    for (let m = 0; m < (o.rounds || 50); m++) {
      const pr = cls ? F.map(sigmoid) : null, g = y.map((v, i) => (cls ? pr[i] - v : F[i] - v)), h = cls ? pr.map(p => Math.max(p * (1 - p), 1e-16)) : new Array(n).fill(1);
      const t = prune(grow(g, h, rows, 0)); t.g = g; t.h = h;
      trees.push(t);
      for (let i = 0; i < n; i++) F[i] += eta * out(t, X[i]);
    }
    const score = (x, m) => trees.slice(0, m === undefined ? trees.length : m).reduce((s, t) => s + eta * out(t, x), F0);
    return { F0, eta, trees, out, score, predict: (x, m) => (cls ? sigmoid(score(x, m)) : score(x, m)) };
  };


  /* =================================================================
     unsupervised batch (4MLSim build 13)
     ================================================================= */
  const d2 = (a, b) => { let s = 0; for (let j = 0; j < a.length; j++) s += (a[j] - b[j]) ** 2; return s; };
  ML.dist2 = d2;
  /* k-means++ (one candidate per step): the next center is drawn with probability proportional to D(x)² */
  ML.kmeansPP = function (X, k, seed) {
    const r = ML.rng(seed * 7 + 3), C = [X[Math.floor(r.random() * X.length)].slice()];
    while (C.length < k) {
      const D = X.map(x => Math.min(...C.map(c => d2(x, c)))), tot = D.reduce((a, b) => a + b, 0);
      let u = r.random() * tot, i = 0; while (i < X.length - 1 && u > D[i]) { u -= D[i]; i++; }
      C.push(X[i].slice());
    }
    return C;
  };
  /* Lloyd's k-means to convergence from given centers (any dimension) */
  ML.kmeansRun = function (X, C, maxIter) {
    let centers = C.map(c => c.slice()), labels = null, it = 0;
    for (; it < (maxIter || 300); it++) {
      const L = X.map(x => { let b = 0, bd = Infinity; centers.forEach((c, k) => { const d = d2(x, c); if (d < bd - 1e-15) { bd = d; b = k; } }); return b; });
      const p = X[0].length, sum = centers.map(() => new Array(p + 1).fill(0));
      X.forEach((x, i) => { for (let j = 0; j < p; j++) sum[L[i]][j] += x[j]; sum[L[i]][p]++; });
      const nc = centers.map((c, k) => (sum[k][p] ? sum[k].slice(0, p).map(v => v / sum[k][p]) : c.slice()));
      const same = labels && L.every((v, i) => v === labels[i]); labels = L; centers = nc; if (same) break;
    }
    const wcss = X.reduce((s, x, i) => s + d2(x, centers[labels[i]]), 0);
    return { centers, labels, wcss, iter: it + 1 };
  };
  /* best of several k-means++ starts (lowest WCSS) */
  ML.kmeansBest = function (X, k, seed, starts) {
    let best = null; for (let s = 0; s < (starts || 5); s++) { const r = ML.kmeansRun(X, ML.kmeansPP(X, k, seed * 31 + s)); if (!best || r.wcss < best.wcss - 1e-9) best = r; } return best;
  };
  /* silhouette of every point and the mean (as sklearn silhouette_samples / silhouette_score) */
  ML.silhouette = function (X, labels) {
    const n = X.length, K = Math.max(...labels) + 1, D = X.map(a => X.map(b => Math.sqrt(d2(a, b)))), cnt = new Array(K).fill(0);
    labels.forEach(l => cnt[l]++);
    const s = X.map((_, i) => {
      if (cnt[labels[i]] <= 1) return 0;
      const sum = new Array(K).fill(0); for (let j = 0; j < n; j++) if (j !== i) sum[labels[j]] += D[i][j];
      const a = sum[labels[i]] / (cnt[labels[i]] - 1); let b = Infinity;
      for (let k = 0; k < K; k++) if (k !== labels[i] && cnt[k]) b = Math.min(b, sum[k] / cnt[k]);
      return b === Infinity ? 0 : (b - a) / Math.max(a, b);
    });
    return { s, mean: s.reduce((a, b) => a + b, 0) / n };
  };
  /* DBSCAN (as sklearn: neighbours within distance <= eps, the point itself counts toward minPts).
     labels: cluster number or -1 for noise; kind: 'core' | 'border' | 'noise' */
  ML.dbscan = function (X, eps, minPts) {
    const n = X.length, e2 = eps * eps, nb = X.map(a => { const r = []; for (let j = 0; j < n; j++) if (d2(a, X[j]) <= e2 + 1e-12) r.push(j); return r; });
    const core = nb.map(r => r.length >= minPts), labels = new Array(n).fill(-1); let c = 0;
    const order = [];   /* the order in which points join a cluster, for the step-by-step view */
    for (let i = 0; i < n; i++) {
      if (labels[i] !== -1 || !core[i]) continue;
      const stack = [i]; labels[i] = c; order.push([i, c]);
      while (stack.length) { const q = stack.pop(); if (!core[q]) continue; for (const j of nb[q]) if (labels[j] === -1) { labels[j] = c; order.push([j, c]); if (core[j]) stack.push(j); } }
      c++;
    }
    return { labels, core, nb, k: c, order, kind: labels.map((l, i) => (core[i] ? 'core' : l >= 0 ? 'border' : 'noise')) };
  };
  /* distance to the k-th nearest neighbour (counting the point itself), sorted: the "k-distance plot" */
  ML.kDistance = (X, k) => X.map(a => X.map(b => Math.sqrt(d2(a, b))).sort((u, v) => u - v)[Math.min(k - 1, X.length - 1)]).sort((u, v) => u - v);

  /* Gaussian mixture, full covariances, 2 inputs. state { w, mu, S } ; reg as sklearn reg_covar */
  const REGC = 1e-6;
  function gpdf2(x, m, S) {
    const a = S[0][0], b = S[0][1], d = S[1][1], det = a * d - b * b, u = x[0] - m[0], v = x[1] - m[1];
    const q = (d * u * u - 2 * b * u * v + a * v * v) / det;
    return { logp: -Math.log(2 * Math.PI) - 0.5 * Math.log(det) - 0.5 * q, q };
  }
  ML.gmmE = function (X, st) {
    let ll = 0; const R = X.map(x => {
      const lp = st.w.map((w, k) => Math.log(w) + gpdf2(x, st.mu[k], st.S[k]).logp), m = Math.max(...lp), s = lp.reduce((t, v) => t + Math.exp(v - m), 0);
      ll += m + Math.log(s); return lp.map(v => Math.exp(v - m) / s);
    });
    return { R, ll: ll / X.length };
  };
  ML.gmmM = function (X, R) {
    const n = X.length, K = R[0].length, w = [], mu = [], S = [];
    for (let k = 0; k < K; k++) {
      const nk = R.reduce((t, r) => t + r[k], 0) + 10 * Number.EPSILON, m = [0, 0];
      X.forEach((x, i) => { m[0] += R[i][k] * x[0] / nk; m[1] += R[i][k] * x[1] / nk; });
      const C = [[REGC, 0], [0, REGC]];
      X.forEach((x, i) => { const u = x[0] - m[0], v = x[1] - m[1], r = R[i][k] / nk; C[0][0] += r * u * u; C[0][1] += r * u * v; C[1][1] += r * v * v; });
      C[1][0] = C[0][1]; w.push(nk / n); mu.push(m); S.push(C);
    }
    return { w, mu, S };
  };
  ML.gmmPdf = (x, st) => st.w.reduce((t, w, k) => t + w * Math.exp(gpdf2(x, st.mu[k], st.S[k]).logp), 0);
  ML.gmmMahal = (x, m, S) => Math.sqrt(gpdf2(x, m, S).q);
  /* ellipse of a 2x2 covariance at c standard deviations: points */
  ML.ellipse = function (m, S, c, n) {
    const a = S[0][0], b = S[0][1], d = S[1][1], tr = (a + d) / 2, q = Math.sqrt(((a - d) / 2) ** 2 + b * b), l1 = tr + q, l2 = Math.max(tr - q, 1e-12);
    const th = Math.abs(b) < 1e-15 ? (a >= d ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b), out = [];
    for (let i = 0; i <= (n || 72); i++) { const t = 2 * Math.PI * i / (n || 72), u = c * Math.sqrt(l1) * Math.cos(t), v = c * Math.sqrt(l2) * Math.sin(t); out.push([m[0] + u * Math.cos(th) - v * Math.sin(th), m[1] + u * Math.sin(th) + v * Math.cos(th)]); }
    return out;
  };
  /* BIC as sklearn: -2 log L + p log n, p = K-1 + 2K + 3K for full 2x2 covariances */
  ML.gmmBIC = (llMean, n, K) => -2 * llMean * n + (K * 6 - 1) * Math.log(n);

  /* symmetric eigen decomposition (Jacobi); values descending, vectors as columns -> returns vecs[k] = k-th vector */
  ML.eigSym = function (A) {
    const n = A.length, a = A.map(r => r.slice()), V = a.map((_, i) => a.map((_, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] ** 2;
      if (off < 1e-22) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const th = (a[q][q] - a[p][p]) / (2 * a[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
        for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
        for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
      }
    }
    const idx = a.map((_, i) => i).sort((i, j) => a[j][j] - a[i][i]);
    return { values: idx.map(i => a[i][i]), vectors: idx.map(i => { const v = V.map(r => r[i]); let m = 0; v.forEach((x, k) => { if (Math.abs(x) > Math.abs(v[m])) m = k; }); return v[m] < 0 ? v.map(x => -x) : v; }) };
  };
  /* PCA (as sklearn PCA: centered, covariance with n - 1; sign: largest loading positive). standardize: correlation PCA */
  ML.pca = function (X, standardize) {
    const n = X.length, p = X[0].length, mu = new Array(p).fill(0), sd = new Array(p).fill(1);
    X.forEach(r => r.forEach((v, j) => (mu[j] += v / n)));
    if (standardize) { sd.fill(0); X.forEach(r => r.forEach((v, j) => (sd[j] += (v - mu[j]) ** 2 / (n - 1)))); for (let j = 0; j < p; j++) sd[j] = Math.sqrt(sd[j]) || 1; }
    const Z = X.map(r => r.map((v, j) => (v - mu[j]) / sd[j])), C = Array.from({ length: p }, () => new Array(p).fill(0));
    Z.forEach(r => { for (let i = 0; i < p; i++) for (let j = i; j < p; j++) C[i][j] += r[i] * r[j] / (n - 1); });
    for (let i = 0; i < p; i++) for (let j = 0; j < i; j++) C[i][j] = C[j][i];
    const e = ML.eigSym(C), tot = e.values.reduce((a, b) => a + b, 0);
    const scores = Z.map(r => e.vectors.map(v => v.reduce((s, w, j) => s + w * r[j], 0)));
    return { mu, sd, cov: C, values: e.values, vectors: e.vectors, ratio: e.values.map(v => v / tot), scores,
      back: (sc, m) => mu.map((u, j) => u + sd[j] * e.vectors.slice(0, m).reduce((s, v, k) => s + v[j] * sc[k], 0)) };
  };

  /* data for the unsupervised modules */
  DATA.payments = {
    task: 'clust', label: 'Card payments: amount and hour', n: 220,
    note: 'Card payments of a bank\'s customers: amount and hour of the day. Most payments form dense groups (morning coffee, lunch, evening shopping); a few scattered ones are unusual and worth a check.',
    gen(r, n) {
      const X = [], y = [];
      for (let i = 0; i < n; i++) {
        const u = r.random();
        if (u < 0.045) { X.push([round(r.uniform(0.5, 23.5), 2), round(r.uniform(5, 400), 1)]); y.push(3); continue; }
        const k = u < 0.35 ? 0 : u < 0.68 ? 1 : 2, sp = [[8.3, 0.6, 8, 3], [13.2, 0.7, 35, 9], [19.2, 0.9, 110, 22]][k];
        X.push([round(clamp(r.normal(sp[0], sp[1]), 0, 24), 2), round(Math.max(1, r.normal(sp[2], sp[3])), 1)]); y.push(k);
      }
      return { X, y, names: ['Hour of the day', 'Amount (€)'] };
    }
  };
  DATA.circlesU = {
    task: 'clust', label: 'Generic: circles', n: 200, generic: true, note: 'A ring around a circle, without labels.',
    gen(r, n) {
      const X = [], y = [];
      for (let i = 0; i < n; i++) { const k = i % 3 === 0 ? 1 : 0, a = r.uniform(0, 2 * Math.PI), rad = (k ? 0.9 : 2.2) + r.normal(0, 0.12); X.push([rad * Math.cos(a), rad * Math.sin(a)]); y.push(k); }
      return { X, y, names: ['x1', 'x2'] };
    }
  };
  DATA.regimes = {
    task: 'clust', label: 'Market regimes: stock and bond returns', n: 250,
    note: 'Daily returns (%) of a stock index and a government bond index. On calm days both move little; on turbulent days stocks swing widely and bonds rise when stocks fall (flight to safety). The regime of each day is not labelled.',
    gen(r, n) {
      const X = [], y = [];
      for (let i = 0; i < n; i++) {
        const k = r.random() < 0.3 ? 1 : 0;
        if (k) { const st = r.normal(-0.15, 2.0); X.push([round(st, 3), round(0.05 - 0.22 * (st + 0.15) + r.normal(0, 0.3), 3)]); }
        else { const st = r.normal(0.06, 0.55); X.push([round(st, 3), round(0.01 + 0.05 * st + r.normal(0, 0.16), 3)]); }
        y.push(k);
      }
      return { X, y, names: ['Stock index return (%)', 'Bond index return (%)'] };
    }
  };
  DATA.elong = {
    task: 'clust', label: 'Generic: stretched groups', n: 240, generic: true, note: 'Two long, tilted groups side by side and a small round one.',
    gen(r, n) {
      const X = [], y = [];
      for (let i = 0; i < n; i++) { const k = i % 5 === 4 ? 2 : i % 2; const t = r.normal(0, 1.6), e = r.normal(0, 0.28);
        if (k === 2) X.push([r.normal(3.2, 0.35), r.normal(-2.2, 0.35)]); else X.push([t * 0.8 - e * 0.6 + (k ? 1.2 : -1.2), t * 0.6 + e * 0.8 + (k ? -0.9 : 0.9)]); y.push(k); }
      return { X, y, names: ['x1', 'x2'] };
    }
  };
  /* yield curve: rates at 8 maturities from level, slope and curvature factors (Nelson-Siegel loadings) */
  DATA.yields = {
    task: 'pca', label: 'Yield curve: interest rates at 8 maturities', n: 120,
    mats: [0.25, 0.5, 1, 2, 3, 5, 10, 30],
    note: 'Monthly government bond yields (%) at 8 maturities, from 3 months to 30 years. The 8 rates move together: a few hidden factors drive them.',
    gen(r, n) {
      const M = DATA.yields.mats, lam = 0.6, X = []; let L = 3.5, Sl = -1.5, Cu = 0;
      for (let t = 0; t < n; t++) {
        L = 3.5 + 0.97 * (L - 3.5) + r.normal(0, 0.22); Sl = -1.5 + 0.93 * (Sl + 1.5) + r.normal(0, 0.3); Cu = 0.9 * Cu + r.normal(0, 0.35);
        X.push(M.map(m => { const a = (1 - Math.exp(-lam * m)) / (lam * m); return round(L + Sl * a + Cu * (a - Math.exp(-lam * m)) + r.normal(0, 0.04), 3); }));
      }
      return { X, names: M.map(m => (m < 1 ? Math.round(m * 12) + 'm' : m + 'y')) };
    }
  };
  DATA.gdpjobs = {
    task: 'pca', label: 'Two indicators: GDP growth and job growth', n: 80,
    note: 'Yearly GDP growth and employment growth (%) of 80 regions: the two move together (Okun\'s law).',
    gen(r, n) { const X = []; for (let i = 0; i < n; i++) { const g = r.normal(2, 1.6); X.push([round(g, 2), round(0.2 + 0.55 * (g - 2) + 0.8 + r.normal(0, 0.45), 2)]); } return { X, names: ['GDP growth (%)', 'Employment growth (%)'] }; }
  };


  /* =================================================================
     neural and evaluation batch (4MLSim build 14)
     ================================================================= */
  const ACTS = {
    relu: { f: z => (z > 0 ? z : 0), d: (z, a) => (z > 0 ? 1 : 0) },
    tanh: { f: z => Math.tanh(z), d: (z, a) => 1 - a * a },
    sigmoid: { f: z => sigmoid(z), d: (z, a) => a * (1 - a) }
  };
  /* small fully connected network for binary classification (sigmoid output, log loss),
     trained by full-batch gradient descent; every quantity is exposed for teaching */
  ML.Net = function (sizes, act, seed) {
    const r = ML.rng(seed * 101 + 7), L = sizes.length - 1, A = ACTS[act];
    const W = [], b = [];
    for (let l = 0; l < L; l++) {
      const sc = act === 'relu' && l < L - 1 ? Math.sqrt(2 / sizes[l]) : Math.sqrt(1 / sizes[l]);
      W.push(Array.from({ length: sizes[l + 1] }, () => Array.from({ length: sizes[l] }, () => r.normal(0, sc)))); b.push(new Array(sizes[l + 1]).fill(0));
    }
    const net = { sizes, act, W, b, L, epoch: 0, vel: null };
    net.forward = x => {
      const a = [x], z = [];
      for (let l = 0; l < L; l++) { const zz = W[l].map((row, k) => row.reduce((s, w, j) => s + w * a[l][j], b[l][k])); z.push(zz); a.push(l < L - 1 ? zz.map(A.f) : zz.map(sigmoid)); }
      return { a, z, p: a[L][0] };
    };
    net.p = x => net.forward(x).p;
    /* gradients of the mean log loss over (X, y) */
    net.grad = (X, y, l2) => {
      const gW = W.map(m => m.map(row => row.map(() => 0))), gb = b.map(v => v.map(() => 0)), n = X.length; let loss = 0;
      X.forEach((x, i) => {
        const f = net.forward(x), p = Math.min(1 - 1e-12, Math.max(1e-12, f.p));
        loss -= (y[i] ? Math.log(p) : Math.log(1 - p)) / n;
        let d = [(f.p - y[i]) / n];
        for (let l = L - 1; l >= 0; l--) {
          for (let k = 0; k < d.length; k++) { gb[l][k] += d[k]; for (let j = 0; j < f.a[l].length; j++) gW[l][k][j] += d[k] * f.a[l][j]; }
          if (l > 0) { const nd = new Array(sizes[l]).fill(0); for (let k = 0; k < d.length; k++) for (let j = 0; j < nd.length; j++) nd[j] += W[l][k][j] * d[k]; d = nd.map((v, j) => v * A.d(f.z[l - 1][j], f.a[l][j])); }
        }
      });
      if (l2) for (let l = 0; l < L; l++) W[l].forEach((row, k) => row.forEach((w, j) => { gW[l][k][j] += l2 * w; loss += 0.5 * l2 * w * w; }));
      return { gW, gb, loss };
    };
    net.loss = (X, y) => -X.reduce((s, x, i) => { const p = Math.min(1 - 1e-12, Math.max(1e-12, net.p(x))); return s + (y[i] ? Math.log(p) : Math.log(1 - p)); }, 0) / X.length;
    /* one epoch of full-batch gradient descent with momentum mu */
    net.step = (X, y, lr, mu, l2) => {
      const g = net.grad(X, y, l2);
      if (!net.vel) net.vel = { W: W.map(m => m.map(row => row.map(() => 0))), b: b.map(v => v.map(() => 0)) };
      for (let l = 0; l < L; l++) for (let k = 0; k < W[l].length; k++) {
        for (let j = 0; j < W[l][k].length; j++) { net.vel.W[l][k][j] = mu * net.vel.W[l][k][j] - lr * g.gW[l][k][j]; W[l][k][j] += net.vel.W[l][k][j]; }
        net.vel.b[l][k] = mu * net.vel.b[l][k] - lr * g.gb[l][k]; b[l][k] += net.vel.b[l][k];
      }
      net.epoch++; return g;
    };
    return net;
  };

  /* LSTM, one layer with H units and a linear output: forecast y from a window of scalars.
     Gates q = 0 forget, 1 input, 2 output, 3 candidate. Flat parameters w:
     per gate [Wx (H), Wh (H x H, row k = unit k), b (H)], then v (H), then c0. */
  ML.LSTM = function (H, seed) {
    const r = ML.rng(seed * 211 + 3), per = H * (H + 2), NP = 4 * per + H + 1, w = new Float64Array(NP), V = 4 * per;
    for (let q = 0; q < 4; q++) { const o = q * per; for (let k = 0; k < H; k++) { w[o + k] = r.normal(0, 0.5); w[o + H + H * H + k] = q === 0 ? 1 : 0; for (let j = 0; j < H; j++) w[o + H + k * H + j] = r.normal(0, 0.5 / Math.sqrt(H)); } }
    for (let k = 0; k < H; k++) w[V + k] = r.normal(0, 1 / Math.sqrt(H));
    const net = { H, w, NP, epoch: 0 };
    /* forward: gate values per step in flat arrays (step t, unit k at t * H + k) */
    function fwd(seq) {
      const T = seq.length, A = { T, f: new Float64Array(T * H), i: new Float64Array(T * H), o: new Float64Array(T * H), g: new Float64Array(T * H), c: new Float64Array(T * H), h: new Float64Array(T * H) };
      for (let t = 0; t < T; t++) {
        const x = seq[t], hp = t ? (t - 1) * H : -1;
        for (let k = 0; k < H; k++) {
          const pre = [0, 0, 0, 0];
          for (let q = 0; q < 4; q++) { const o = q * per; let z = w[o + H + H * H + k] + w[o + k] * x; if (hp >= 0) for (let j = 0; j < H; j++) z += w[o + H + k * H + j] * A.h[hp + j]; pre[q] = z; }
          const f = sigmoid(pre[0]), ii = sigmoid(pre[1]), oo = sigmoid(pre[2]), g = Math.tanh(pre[3]), cp = hp >= 0 ? A.c[hp + k] : 0, c = f * cp + ii * g;
          A.f[t * H + k] = f; A.i[t * H + k] = ii; A.o[t * H + k] = oo; A.g[t * H + k] = g; A.c[t * H + k] = c; A.h[t * H + k] = oo * Math.tanh(c);
        }
      }
      let y = w[NP - 1]; for (let k = 0; k < H; k++) y += w[V + k] * A.h[(T - 1) * H + k];
      A.y = y; return A;
    }
    net.predict = seq => fwd(seq).y;
    net.trace = seq => { const A = fwd(seq), out = []; for (let t = 0; t < A.T; t++) { const sl = a => Array.from(a.slice(t * H, t * H + H)); out.push({ x: seq[t], f: sl(A.f), i: sl(A.i), o: sl(A.o), cc: sl(A.g), c: sl(A.c), h: sl(A.h) }); } return { steps: out, yhat: A.y }; };
    /* mean squared error and its gradient over windows (backpropagation through time) */
    net.grad = (S, Y) => {
      const g = new Float64Array(NP), n = S.length; let loss = 0;
      const dh = new Float64Array(H), dc = new Float64Array(H), dhp = new Float64Array(H), dp = new Float64Array(4 * H);
      for (let s = 0; s < n; s++) {
        const seq = S[s], A = fwd(seq), T = A.T, e = A.y - Y[s], dy = 2 * e / n; loss += e * e / n;
        g[NP - 1] += dy; for (let k = 0; k < H; k++) { g[V + k] += dy * A.h[(T - 1) * H + k]; dh[k] = dy * w[V + k]; dc[k] = 0; }
        for (let t = T - 1; t >= 0; t--) {
          const b = t * H, hp = t ? (t - 1) * H : -1;
          for (let k = 0; k < H; k++) {
            const tc = Math.tanh(A.c[b + k]), o = A.o[b + k], f = A.f[b + k], ii = A.i[b + k], gg = A.g[b + k], cp = hp >= 0 ? A.c[hp + k] : 0;
            const dC = dc[k] + dh[k] * o * (1 - tc * tc);
            dp[2 * H + k] = dh[k] * tc * o * (1 - o); dp[k] = dC * cp * f * (1 - f); dp[H + k] = dC * gg * ii * (1 - ii); dp[3 * H + k] = dC * ii * (1 - gg * gg);
            dc[k] = dC * f;
          }
          dhp.fill(0);
          for (let q = 0; q < 4; q++) { const o = q * per; for (let k = 0; k < H; k++) { const d = dp[q * H + k]; if (!d) continue; g[o + H + H * H + k] += d; g[o + k] += d * seq[t]; if (hp >= 0) for (let j = 0; j < H; j++) { g[o + H + k * H + j] += d * A.h[hp + j]; dhp[j] += d * w[o + H + k * H + j]; } } }
          for (let k = 0; k < H; k++) dh[k] = dhp[k];
        }
      }
      return { g, loss };
    };
    net.loss = (S, Y) => S.reduce((t, seq, s) => t + (fwd(seq).y - Y[s]) ** 2, 0) / S.length;
    const m = new Float64Array(NP), v2 = new Float64Array(NP); let tt = 0;
    net.step = (S, Y, lr) => {
      const { g, loss } = net.grad(S, Y); tt++;
      const c1 = 1 - Math.pow(0.9, tt), c2 = 1 - Math.pow(0.999, tt);
      for (let k = 0; k < NP; k++) { m[k] = 0.9 * m[k] + 0.1 * g[k]; v2[k] = 0.999 * v2[k] + 0.001 * g[k] * g[k]; w[k] -= lr * (m[k] / c1) / (Math.sqrt(v2[k] / c2) + 1e-8); }
      net.epoch++; return loss;
    };
    return net;
  };


  /* ---------- data snooping: trading rules, bootstrap tests ---------- */
  /* daily market returns (%): r_t = phi r_{t-1} + e_t */
  ML.marketReturns = function (r, T, phi, sd) { const y = []; let p = 0; for (let t = 0; t < T; t++) { p = phi * p + r.normal(0, sd); y.push(p); } return y; };
  /* moving-average rules: long (+1) when MA(s) > MA(l) of the price up to yesterday, else short (-1) */
  ML.maRulePool = function (seed) { const pool = []; for (let s = 1; s <= 40; s++) for (let l = s + 5; l <= 200; l += 5) pool.push([s, l]); return ML.rng(seed * 13 + 1).shuffle(pool); };
  ML.maRuleReturns = function (ret, rules, start) {
    const T = ret.length, P = [0]; for (let t = 0; t < T; t++) P.push(P[t] + ret[t]);           /* log-price index, P[t] = before day t */
    const cs = [0]; for (let t = 0; t <= T; t++) cs.push(cs[t] + P[t]);                          /* cumulative sums of the price */
    const ma = (t, w) => (cs[t + 1] - cs[t + 1 - w]) / w;                                        /* mean of P[t-w+1..t] */
    return rules.map(([s, l]) => { const d = new Float64Array(T - start); for (let t = start; t < T; t++) d[t - start] = (ma(t, s) > ma(t, l) ? 1 : -1) * ret[t]; return d; });
  };
  /* stationary bootstrap (Politis and Romano) index sequences */
  ML.stationaryIdx = function (r, T, block) { const idx = new Int32Array(T), p = 1 / block; let k = Math.floor(r.random() * T); for (let t = 0; t < T; t++) { if (t > 0) k = r.random() < p ? Math.floor(r.random() * T) : (k + 1) % T; idx[t] = k; } return idx; };
  /* bootstrap means of every series: out[k * B + b] */
  ML.bootMeans = function (D, B, block, seed) {
    const M = D.length, T = D[0].length, r = ML.rng(seed * 7 + 5), out = new Float64Array(M * B);
    for (let b = 0; b < B; b++) { const idx = ML.stationaryIdx(r, T, block); for (let k = 0; k < M; k++) { const d = D[k]; let s = 0; for (let t = 0; t < T; t++) s += d[idx[t]]; out[k * B + b] = s / T; } }
    return out;
  };
  /* White's Reality Check and Hansen's SPA (consistent) for H0: no rule beats the benchmark (mean <= 0) */
  ML.rcSpa = function (D, BM, B) {
    const M = D.length, T = D[0].length, mean = D.map(d => d.reduce((a, b) => a + b, 0) / T), sq = Math.sqrt(T);
    const om = mean.map((m, k) => { let s = 0; for (let b = 0; b < B; b++) s += (BM[k * B + b] - m) ** 2; return Math.sqrt(T * s / B) || 1e-12; });
    const V = Math.max(...mean) * sq, Tspa = Math.max(0, Math.max(...mean.map((m, k) => sq * m / om[k])));
    const thr = -Math.sqrt(2 * Math.log(Math.log(T))), gc = mean.map((m, k) => (sq * m / om[k] >= thr ? m : 0));
    let rc = 0, spa = 0;
    for (let b = 0; b < B; b++) {
      let v = -Infinity, w = 0;
      for (let k = 0; k < M; k++) { const x = BM[k * B + b]; v = Math.max(v, sq * (x - mean[k])); w = Math.max(w, sq * (x - gc[k]) / om[k]); }
      if (v >= V) rc++; if (w >= Tspa) spa++;
    }
    return { rc: rc / B, spa: spa / B, mean, omega: om };
  };
  /* Model Confidence Set (Hansen, Lunde and Nason 2011) with the T_max statistic on losses L = -d */
  ML.mcs = function (D, BM, B, alpha) {
    const M = D.length, T = D[0].length, mean = D.map(d => -d.reduce((a, b) => a + b, 0) / T);
    let set = D.map((_, k) => k); const pval = new Array(M).fill(1), order = []; let pmax = 0;
    while (set.length > 1) {
      const m = set.length, avg = set.reduce((s, k) => s + mean[k], 0) / m, bavg = new Float64Array(B);
      for (let b = 0; b < B; b++) { let s = 0; set.forEach(k => (s -= BM[k * B + b])); bavg[b] = s / m; }
      const dk = set.map(k => mean[k] - avg), vk = set.map((k, i) => { let s = 0; for (let b = 0; b < B; b++) s += (-BM[k * B + b] - bavg[b] - dk[i]) ** 2; return s / B || 1e-18; });
      const tk = dk.map((d, i) => d / Math.sqrt(vk[i])), Tmax = Math.max(...tk);
      let cnt = 0; for (let b = 0; b < B; b++) { let mx = -Infinity; set.forEach((k, i) => { mx = Math.max(mx, (-BM[k * B + b] - bavg[b] - dk[i]) / Math.sqrt(vk[i])); }); if (mx >= Tmax) cnt++; }
      const p = cnt / B; pmax = Math.max(pmax, p);
      const worst = set[tk.indexOf(Tmax)]; pval[worst] = pmax; order.push(worst); set = set.filter(k => k !== worst);
    }
    pval[set[0]] = 1;
    return { pval, included: pval.map(p => p >= alpha) };
  };


  /* ---------- explainability ---------- */
  DATA.houses = {
    task: 'reg', label: 'House prices', n: 300,
    note: 'Sale prices of houses (thousand €) with five inputs: size, rooms (closely tied to size), distance to the city centre, age, and the last digit of the street number, which has nothing to do with the price.',
    gen(r, n) {
      const X = [], y = [];
      for (let i = 0; i < n; i++) {
        const size = clamp(r.normal(110, 35), 35, 260), rooms = clamp(Math.round(size / 28 + r.normal(0, 0.6)), 1, 8), dist = clamp(-6 * Math.log(1 - r.random() * 0.98), 0.3, 25), age = r.uniform(0, 100), digit = r.int(0, 9);
        X.push([round(size, 0), rooms, round(dist, 1), round(age, 0), digit]);
        y.push(round(30 + size * (2.6 - 0.07 * Math.min(dist, 20)) - 1.6 * age + 0.02 * age * age + 4 * rooms + r.normal(0, 18), 1));
      }
      return { X, y, names: ['Size (m²)', 'Rooms', 'Distance to centre (km)', 'Age (years)', 'Street number, last digit'], target: 'Price (thousand €)' };
    }
  };
  ML.ols = function (X, y) { const p = X[0].length + 1, A = Array.from({ length: p }, () => new Array(p).fill(0)), b = new Array(p).fill(0); X.forEach((x, i) => { const z = [1].concat(x); for (let a = 0; a < p; a++) { b[a] += z[a] * y[i]; for (let c = 0; c < p; c++) A[a][c] += z[a] * z[c]; } }); const beta = solve(A, b); return { beta, predict: x => beta[0] + x.reduce((s, v, j) => s + v * beta[j + 1], 0) }; };
  /* permutation importance: increase in RMSE when one input is shuffled (average of reps shuffles) */
  ML.permImportance = function (f, X, y, reps, seed) {
    const r = ML.rng(seed * 5 + 1), rm = P => Math.sqrt(P.reduce((s, x, i) => s + (f(x) - y[i]) ** 2, 0) / y.length), base = rm(X);
    return { base, imp: X[0].map((_, j) => { let s = 0; for (let q = 0; q < reps; q++) { const col = r.shuffle(X.map(x => x[j])); s += rm(X.map((x, i) => { const z = x.slice(); z[j] = col[i]; return z; })) - base; } return s / reps; }) };
  };
  /* exact interventional Shapley values with a background sample: v(S) = mean over b of f(x_S, b_rest) */
  ML.shapExact = function (f, x, bg) {
    const p = x.length, N = 1 << p, v = new Float64Array(N);
    for (let m = 0; m < N; m++) { let s = 0; bg.forEach(b => { const z = b.slice(); for (let j = 0; j < p; j++) if (m & (1 << j)) z[j] = x[j]; s += f(z); }); v[m] = s / bg.length; }
    const fact = [1]; for (let k = 1; k <= p; k++) fact[k] = fact[k - 1] * k;
    const phi = new Array(p).fill(0);
    for (let m = 0; m < N; m++) { let k = 0; for (let j = 0; j < p; j++) if (m & (1 << j)) k++; for (let j = 0; j < p; j++) if (!(m & (1 << j))) phi[j] += fact[k] * fact[p - k - 1] / fact[p] * (v[m | (1 << j)] - v[m]); }
    return { phi, base: v[0], pred: v[N - 1] };
  };


  /* =================================================================
     reinforcement learning and language (4MLSim build 15)
     ================================================================= */
  /* gridworld: rows of characters. S start, G goal (+10), g small goal (+3), T trap (-10), # wall, . empty.
     every move costs 0.1; with probability slip the move goes to a random side instead */
  ML.Grid = function (rows, slip) {
    const H = rows.length, W = rows[0].length, cell = (r, c) => (r < 0 || c < 0 || r >= H || c >= W ? '#' : rows[r][c]);
    let start = 0; rows.forEach((row, r) => [...row].forEach((ch, c) => { if (ch === 'S') start = r * W + c; }));
    const DR = [-1, 0, 1, 0], DC = [0, 1, 0, -1], REW = { G: 10, g: 3, T: -10 };
    const env = { H, W, rows, start, slip: slip || 0, nS: H * W, step: -0.1 };
    env.kind = s => cell(Math.floor(s / W), s % W);
    env.terminal = s => 'GgT'.includes(env.kind(s));
    env.move = (s, a) => { const r = Math.floor(s / W) + DR[a], c = (s % W) + DC[a]; return cell(r, c) === '#' ? s : r * W + c; };
    env.reward = s2 => env.step + (REW[env.kind(s2)] || 0);
    /* the outcomes of action a in s: [probability, next state] */
    env.outcomes = (s, a) => (env.slip ? [[1 - env.slip, env.move(s, a)], [env.slip / 2, env.move(s, (a + 1) % 4)], [env.slip / 2, env.move(s, (a + 3) % 4)]] : [[1, env.move(s, a)]]);
    env.sample = (s, a, r) => { const u = r.random(); if (!env.slip || u >= env.slip) return env.move(s, a); return env.move(s, u < env.slip / 2 ? (a + 1) % 4 : (a + 3) % 4); };
    return env;
  };
  /* value iteration: the optimal Q* (the Bellman optimality equation solved exactly) */
  ML.valueIteration = function (env, gamma) {
    const Q = Array.from({ length: env.nS }, () => [0, 0, 0, 0]);
    for (let it = 0; it < 2000; it++) {
      let diff = 0;
      for (let s = 0; s < env.nS; s++) { if (env.terminal(s) || env.kind(s) === '#') continue; for (let a = 0; a < 4; a++) { const q = env.outcomes(s, a).reduce((t, [p, s2]) => t + p * (env.reward(s2) + (env.terminal(s2) ? 0 : gamma * Math.max(...Q[s2]))), 0); diff = Math.max(diff, Math.abs(q - Q[s][a])); Q[s][a] = q; } }
      if (diff < 1e-10) break;
    }
    return Q;
  };
  /* one Q-learning episode with an epsilon-greedy policy; returns the path and the total reward */
  ML.qEpisode = function (env, Q, o, r) {
    let s = env.start, total = 0; const path = [s];
    for (let t = 0; t < (o.maxSteps || 200); t++) {
      const best = Math.max(...Q[s]), greedy = [0, 1, 2, 3].filter(a => Q[s][a] === best);
      const a = r.random() < o.eps ? Math.floor(r.random() * 4) : greedy[Math.floor(r.random() * greedy.length)];
      const s2 = env.sample(s, a, r), rew = env.reward(s2), target = rew + (env.terminal(s2) ? 0 : o.gamma * Math.max(...Q[s2]));
      Q[s][a] += o.alpha * (target - Q[s][a]); total += rew; s = s2; path.push(s);
      if (env.terminal(s)) break;
    }
    return { path, total };
  };
  /* follow the greedy policy from the start (no slipping) */
  ML.greedyPath = function (env, Q, maxSteps) {
    let s = env.start; const path = [s], seen = new Set([s]);
    for (let t = 0; t < (maxSteps || 60); t++) { const best = Math.max(...Q[s]), a = Q[s].indexOf(best); s = env.move(s, a); path.push(s); if (env.terminal(s) || seen.has(s)) break; seen.add(s); }
    return { path, end: env.kind(s), done: env.terminal(s) };
  };


  /* ---------- a tiny language model: word n-grams, embeddings from co-occurrence, attention ---------- */
  ML.CORPUS = `the central bank raises interest rates when inflation is high .
the central bank cuts interest rates when the economy is weak .
higher interest rates make loans more expensive for firms and households .
lower interest rates make loans cheaper and support investment .
when inflation rises the real value of money falls .
when prices rise quickly households lose purchasing power .
inflation is the rate at which prices rise over time .
the central bank targets an inflation rate of two percent .
the bank gives loans to firms and households .
the bank takes deposits from households and firms .
a bank fails when its losses are larger than its capital .
banks with more capital can absorb larger losses .
the bank pays interest on deposits and charges interest on loans .
firms borrow money to invest in new machines .
firms invest more when demand is strong and rates are low .
households spend more when their income rises .
households save more when they are uncertain about the future .
unemployment rises when the economy enters a recession .
unemployment falls when the economy grows quickly .
in a recession output falls and unemployment rises .
in an expansion output grows and unemployment falls .
the government spends more in a recession to support demand .
the government collects taxes from households and firms .
higher taxes reduce the income of households .
the stock market rises when firms expect higher profits .
the stock market falls when investors expect a recession .
investors buy bonds when they expect interest rates to fall .
bond prices fall when interest rates rise .
bond prices rise when interest rates fall .
the yield curve shows interest rates for different maturities .
an inverted yield curve often comes before a recession .
the exchange rate is the price of one currency in another currency .
a weaker currency makes exports cheaper for foreign buyers .
a stronger currency makes imports cheaper for households .
exports and imports depend on the exchange rate and on demand .
oil prices affect inflation through energy and transport costs .
higher oil prices raise the costs of firms .
firms raise prices when their costs rise .
wages rise when unemployment is low and firms need workers .
higher wages raise the income of households and the costs of firms .
productivity growth allows wages to rise without higher inflation .
economic growth depends on investment productivity and workers .
the economy grows when investment and productivity rise .
credit grows quickly when interest rates are low .
too much credit can lead to a financial crisis .
in a financial crisis banks stop lending and asset prices fall .
the central bank lends to banks in a financial crisis .
house prices rise when interest rates are low and credit grows .
house prices fall when interest rates rise quickly .
the price of a house depends on its size and its location .
machine learning models forecast inflation and unemployment .
machine learning models learn patterns from past data .
a model that fits the training data too closely may forecast badly .
we test a forecasting model on new data .
a good forecast beats a simple benchmark on new data .
the model predicts the next word from the previous words .
a language model predicts the next word in a sentence .
a language model learns which words follow other words .
the temperature controls how random the next word is .
words that appear in similar sentences have similar meanings .
inflation and prices appear in similar sentences .
loans and deposits appear in similar sentences .
the central bank and the government both affect demand .
the economy and output grow together in an expansion .`;
  ML.lm = function (text) {
    const sents = text.trim().split('\n').map(l => ['<s>'].concat(l.trim().split(/\s+/))), toks = [].concat(...sents);
    const uni = {}, bi = {}, tri = {}, inc = (o, k) => (o[k] = (o[k] || 0) + 1);
    sents.forEach(w => w.forEach((t, i) => { if (t !== '<s>') inc(uni, t); if (i > 0) { inc(bi, w[i - 1] + ' ' + t); } if (i > 1) inc(tri, w[i - 2] + ' ' + w[i - 1] + ' ' + t); }));
    const vocab = Object.keys(uni).sort((a, b) => uni[b] - uni[a] || (a < b ? -1 : 1)), N = vocab.reduce((s, v) => s + uni[v], 0);
    /* next-word distribution: trigram counts if the last two words were seen together, else bigram, else unigram */
    const next = ctx => {
      const a = ctx[ctx.length - 2], b = ctx[ctx.length - 1];
      for (const [level, pre, tab] of [['trigram', a + ' ' + b + ' ', tri], ['bigram', b + ' ', bi]]) {
        if (!b || (level === 'trigram' && !a)) continue;
        const c = {}; let tot = 0; Object.keys(tab).forEach(k => { if (k.startsWith(pre)) { const w = k.slice(pre.length); if (!w.includes(' ')) { c[w] = tab[k]; tot += tab[k]; } } });
        if (tot) return { level, context: level === 'trigram' ? a + ' ' + b : b, total: tot, probs: Object.keys(c).map(w => [w, c[w] / tot, c[w]]).sort((x, y) => y[1] - x[1]) };
      }
      return { level: 'unigram', context: '', total: N, probs: vocab.map(w => [w, uni[w] / N, uni[w]]) };
    };
    return { sents, toks, uni, bi, tri, vocab, N, next };
  };
  /* temperature: p_i^(1/T), renormalized; top-k keeps the k most likely */
  ML.temper = (probs, T, k) => { const kept = probs.slice(0, k || probs.length), w = kept.map(p => Math.pow(p[1], 1 / Math.max(T, 1e-3))), s = w.reduce((a, b) => a + b, 0); return kept.map((p, i) => [p[0], w[i] / s, p[2]]); };
  /* word vectors: positive pointwise mutual information of co-occurrence within +-2 words, reduced by PCA */
  ML.embed = function (lm, minCount, dims) {
    const words = lm.vocab.filter(w => lm.uni[w] >= minCount && w !== '.'), idx = {}; words.forEach((w, i) => (idx[w] = i));
    const V = words.length, C = Array.from({ length: V }, () => new Float64Array(V));
    lm.sents.forEach(s => s.forEach((t, i) => { if (idx[t] === undefined) return; for (let j = Math.max(0, i - 2); j <= Math.min(s.length - 1, i + 2); j++) if (j !== i && idx[s[j]] !== undefined) C[idx[t]][idx[s[j]]]++; }));
    const row = C.map(r => r.reduce((a, b) => a + b, 0)), tot = row.reduce((a, b) => a + b, 0) || 1;
    const P = C.map((r, i) => Array.from(r, (c, j) => (c && row[i] && row[j] ? Math.max(0, Math.log(c * tot / (row[i] * row[j]))) : 0)));
    const pc = ML.pca(P, false), d = Math.min(dims || 16, V);
    const vec = pc.scores.map(s => { const v = s.slice(0, d), n = Math.hypot(...v) || 1; return v.map(x => x / n); });
    return { words, idx, vec, xy: pc.scores.map(s => [s[0], s[1]]), ratio: pc.ratio };
  };
  ML.cosine = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0) / ((Math.hypot(...a) * Math.hypot(...b)) || 1);

  if (typeof module !== 'undefined' && module.exports) module.exports = ML;
  else { root.MLS = root.MLS || {}; root.MLS.ml = ML; }
})(typeof window !== 'undefined' ? window : this);
