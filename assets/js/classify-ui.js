/* =====================================================================
   4MLSim classification modules: shared pieces (confusion matrix,
   metric table with the numbers filled in, help texts, class colours,
   decision-region image). Uses MLS (core.js, plot.js) and MLS.ml.
   ===================================================================== */
(function () {
  'use strict';
  const MLS = window.MLS, ML = MLS.ml, F = MLS.fmt;
  const C = (MLS.cls = {});
  C.color = k => MLS.seriesColor(k);              /* class 0 blue, class 1 orange, class 2 green */
  C.pct = v => (Number.isFinite(v) ? (100 * v).toFixed(1) + '%' : '–');

  /* the two-ratio bank data, split 80/20 (training / test) */
  C.banks = function (seed, n, share) {
    const g = ML.DATA.banks.gen(ML.rng(seed), n, share), test = ML.split(n, seed);
    const pts = g.X.map((x, i) => ({ x, y: g.y[i], te: test[i], id: i }));
    return { pts, names: g.names, classes: g.classes, K: 2 };
  };

  /* confusion matrix: rows = actual, columns = predicted; with 2 classes the positive class (1) first */
  C.cmTable = function (M, classes, o) {
    o = o || {};
    const K = classes.length, order = K === 2 ? [1, 0] : classes.map((_, k) => k), rowSum = a => M[a].reduce((s, v) => s + v, 0);
    const lab = (a, b) => (K !== 2 ? '' : a === 1 ? (b === 1 ? 'TP' : 'FN, type II error') : (b === 1 ? 'FP, type I error' : 'TN'));
    const n = M.flat().reduce((a, b) => a + b, 0);
    const cell = (a, b) => { const v = M[a][b], r = rowSum(a), txt = o.pct ? (r ? (100 * v / r).toFixed(0) + '%' : '–') : String(v);
      return '<td class="' + (a === b ? 'ok' : v ? 'bad' : '') + '">' + txt + (lab(a, b) ? '<small>' + lab(a, b) + '</small>' : '') + '</td>'; };
    return '<table class="cm"><caption>' + (o.title || '') + ' (' + n + ')</caption><thead><tr><th>Actual &darr; / Predicted &rarr;</th>' + order.map(b => '<th>' + MLS.esc(classes[b]) + '</th>').join('') + '</tr></thead><tbody>' +
      order.map(a => '<tr><th>' + MLS.esc(classes[a]) + ' (' + rowSum(a) + ')</th>' + order.map(b => cell(a, b)).join('') + '</tr>').join('') + '</tbody></table>';
  };
  C.cmNote = (classes, pct) => '<p class="note">Green: correct classifications. Red: misclassifications. Rows: the actual class; columns: the predicted class.' + (classes.length === 2 ? ' ' + MLS.esc(classes[1]) + ' is the 1 (positive).' : '') + (pct ? ' Percentages are shares of each actual class (each row adds up to 100%).' : '') + '</p>';

  /* metrics of a 2-class confusion matrix, with the formula and the numbers used */
  C.ROWS = [
    ['acc', 'Accuracy', '(TP + TN) / all', m => '(' + m.TP + ' + ' + m.TN + ') / ' + m.n, 'm_acc'],
    ['mis', 'Misclassification rate', '(FP + FN) / all', m => '(' + m.FP + ' + ' + m.FN + ') / ' + m.n, 'm_mis'],
    ['sens', 'Sensitivity (recall)', 'TP / (TP + FN)', m => m.TP + ' / (' + m.TP + ' + ' + m.FN + ')', 'm_sens'],
    ['spec', 'Specificity', 'TN / (TN + FP)', m => m.TN + ' / (' + m.TN + ' + ' + m.FP + ')', 'm_spec'],
    ['fa', 'False alarms (1 − specificity)', 'FP / (TN + FP)', m => m.FP + ' / (' + m.TN + ' + ' + m.FP + ')', 'm_fa'],
    ['prec', 'Precision', 'TP / (TP + FP)', m => m.TP + ' / (' + m.TP + ' + ' + m.FP + ')', 'm_prec'],
    ['f1', 'F1', '2 × prec. × sens. / (prec. + sens.)', m => '2 × ' + F(m.prec, 3) + ' × ' + F(m.sens, 3) + ' / (' + F(m.prec, 3) + ' + ' + F(m.sens, 3) + ')', 'm_f1']
  ];
  C.metricsTable = function (mTr, mTe, o) {
    o = o || {};
    const rows = C.ROWS.filter(r => !o.only || o.only.includes(r[0]));
    return '<div style="overflow-x:auto"><table class="tbl met"><thead><tr><th>Metric</th>' + (mTr ? '<th>Training set (' + mTr.n + ')</th>' : '') + '<th>Test set (' + mTe.n + ')</th>' + (o.numbers ? '<th>Test set: the numbers</th>' : '') + '</tr></thead><tbody>' +
      rows.map(r => '<tr><td><span style="display:inline-flex;align-items:center">' + r[1] + MLS.helpBtn(r[4]) + '</span><div class="what">' + r[2] + '</div></td>' +
        (mTr ? '<td>' + (r[0] === 'f1' ? F(mTr.f1, 3) : C.pct(mTr[r[0]])) + '</td>' : '') + '<td>' + (r[0] === 'f1' ? F(mTe.f1, 3) : C.pct(mTe[r[0]])) + '</td>' +
        (o.numbers ? '<td class="num-f">' + r[3](mTe) + '</td>' : '') + '</tr>').join('') +
      (o.extra || '') + '</tbody></table></div>';
  };
  C.evalBinary = (pts, prob, T) => {
    const M = ML.confusion(pts.map(p => p.y), pts.map(p => (prob(p) >= T ? 1 : 0)), 2);
    return Object.assign({ M }, ML.binaryMetrics(M, 1));
  };

  /* coloured background: the class predicted at every point of the chart (paler = less sure) */
  C.regionGrid = function (dom, prob, nx, ny) {
    const xs = [], ys = [];
    for (let i = 0; i < nx; i++) xs.push(dom.x[0] + (i + 0.5) * (dom.x[1] - dom.x[0]) / nx);
    for (let j = 0; j < ny; j++) ys.push(dom.y[0] + (j + 0.5) * (dom.y[1] - dom.y[0]) / ny);
    return { xs, ys, P: ys.map(y => xs.map(x => prob([x, y]))) };
  };
  C.drawRegions = function (P, g, dom, K, T) {   /* T = null: the class with the largest share (votes, posterior) */
    const rgb = Array.from({ length: K }, (_, k) => MLS.hexToRgb(C.color(k))), dark = MLS.getTheme() === 'dark';
    const url = MLS.gridImage(g.P, p => {
      let k, conf;
      if (K === 2 && T !== null) { k = p[1] >= T ? 1 : 0; conf = Math.min(1, Math.abs(p[1] - T) / Math.max(T, 1 - T)); }
      else { k = 0; for (let q = 1; q < K; q++) if (p[q] > p[k]) k = q; conf = (p[k] - 1 / K) / (1 - 1 / K); }
      const c = rgb[k]; return [c[0], c[1], c[2], Math.round(255 * (dark ? 0.08 + 0.2 * conf : 0.12 + 0.26 * conf))];
    });
    P.image(url, dom.x[0], dom.x[1], dom.y[0], dom.y[1], { smooth: true });
    if (K === 2) P.contour(g.P.map(r => r.map(p => p[1])), g.xs, g.ys, [T === null ? 0.5 : T], { color: MLS.color('chart-ink'), width: 2.2, opacity: 0.9 });
  };
  C.domain = pts => ({ x: MLS.niceDomain(Math.min(...pts.map(p => p.x[0])), Math.max(...pts.map(p => p.x[0])), 0.08), y: MLS.niceDomain(Math.min(...pts.map(p => p.x[1])), Math.max(...pts.map(p => p.x[1])), 0.1) });

  /* help texts shared by all classification modules (wording of the lectures) */
  MLS.registerHelp({
    cm: { title: 'Confusion matrix', html: 'Every prediction compared with the actual class. Rows: the actual class; columns: the predicted class. Green cells (the diagonal) are correct classifications; red cells are misclassifications. With 2 classes: <b>TP</b> (true positive: a 1 predicted as 1), <b>FN</b> (false negative, a missed case, type II error), <b>FP</b> (false positive, a false alarm, type I error) and <b>TN</b> (true negative). "% of actual" shows each row as shares of the actual class.' },
    metrics: { title: 'Metrics', html: 'How good the predictions are, on the training set (data the model learned from) and on the test set (new data, used once). A much better training score means overfitting.' },
    m_acc: { title: 'Accuracy', html: 'How good is the model overall? Correct predictions / all predictions = (TP + TN) / all. Careful with unequal classes: with 99 healthy banks and 1 failed bank, calling every bank healthy gives 99% accuracy and finds no failed bank.' },
    m_mis: { title: 'Misclassification rate', html: 'How bad is the model overall? Wrong predictions / all = (FP + FN) / all = 1 − accuracy.' },
    m_sens: { title: 'Sensitivity (recall, true positive rate)', html: 'What share of the actual 1s is found? TP / (TP + FN). How good we are at detecting all class 1 cases.' },
    m_spec: { title: 'Specificity (true negative rate)', html: 'What share of the actual 0s is recognised as 0? TN / (TN + FP). How good we are at detecting all class 0 cases.' },
    m_fa: { title: 'False alarms', html: '0s classified as 1s, as a share of all 0s: FP / (TN + FP) = 1 − specificity. The horizontal axis of the ROC curve.' },
    m_prec: { title: 'Precision', html: 'Among the observations predicted as 1, how many are correct? TP / (TP + FP). Also called positive predictive value.' },
    m_f1: { title: 'F1', html: 'Balances sensitivity and precision: 2 × precision × sensitivity / (precision + sensitivity). 1 = both perfect. F0.5 gives more weight to precision, F2 to sensitivity.' },
    m_auc: { title: 'AUC', html: 'The area under the ROC curve: 1 = perfect, 0.5 = a coin toss. If we pick at random one observation of class 1 and one of class 0, AUC is the probability that the model gives the class-1 observation the higher probability. It does not depend on the threshold.' },
    split: { title: 'Training set and test set', html: '80% of the observations, chosen at random, form the <b>training set</b> (filled points): the model learns from them. The other 20% form the <b>test set</b> (rings): new data, used once to check the model.' },
    seed: { title: 'Random seed', html: 'Fixes the simulated data and the split into training and test set. The dice gives new data from the same source.' }
  });
})();
