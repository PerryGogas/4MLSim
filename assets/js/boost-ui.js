/* =====================================================================
   4MLSim boosting modules (Gradient Boosting, XGBoost): one driver for
   the data, the round-by-round model, the charts and the controls.
   cfg: { id, params:[{key,label,help,min,max,step,list}], defaults,
          fit(X, y, cls, S), leafValue(nd), treeLines(nd, open), treeTip(nd, open),
          numbers(R) -> html or '', help:{}, challenges(api) -> [] }
   ===================================================================== */
(function () {
  'use strict';
  const MLS = window.MLS, ML = MLS.ml, CL = MLS.cls, TR = MLS.trees, F = MLS.fmt;
  MLS.boost = {};
  MLS.boost.page = function (cfg) {
    const $ = id => document.getElementById(id);
    const META = MLS.moduleMeta(cfg.id);
    const LISTS = { class: ['banks', 'circles'], reg: ['kwh', 'power'] };
    const S = Object.assign({ task: 'reg', data: { class: 'banks', reg: 'kwh' }, n: {}, seed: 1, m: -1 }, cfg.defaults);
    let D = null, R = null;
    const dkey = () => S.data[S.task], nObs = () => S.n[dkey()] || TR.DATA[dkey()].n, isC = () => D.task === 'class';

    function compute() {
      D = TR.make(dkey(), S.seed, nObs());
      const X = D.train.map(p => p.x), y = D.train.map(p => p.y);
      const model = cfg.fit(X, y, isC(), S);
      R = { model, X, y, M: model.trees.length };
      /* scores after every round (0 = the starting value only) */
      const Ftr = X.map(() => model.F0), Fte = D.test.map(() => model.F0), out = (t, x) => cfg.leafValue(t, x);
      const sc = (Fs, pts) => (isC() ? pts.filter((p, i) => (ML.sigmoid(Fs[i]) >= 0.5 ? 1 : 0) === p.y).length / pts.length : Math.sqrt(pts.reduce((s, p, i) => s + (p.y - Fs[i]) ** 2, 0) / pts.length));
      R.curve = [{ m: 0, tr: sc(Ftr, D.train), te: sc(Fte, D.test) }];
      R.Fhist = [Ftr.slice()];
      model.trees.forEach((t, k) => {
        X.forEach((x, i) => (Ftr[i] += model.eta * out(t, x))); D.test.forEach((p, i) => (Fte[i] += model.eta * out(t, p.x)));
        R.curve.push({ m: k + 1, tr: sc(Ftr, D.train), te: sc(Fte, D.test) }); R.Fhist.push(Ftr.slice());
      });
      let best = R.curve[0]; R.curve.forEach(c => { if (isC() ? c.te > best.te + 1e-12 : c.te < best.te - 1e-12) best = c; }); R.bestTe = best;
      if (S.m > R.M) S.m = -1;
    }
    const cur = () => (S.m < 0 ? R.M : S.m);
    const fAt = m => x => { const s = R.model.score(x, m); return isC() ? [1 - ML.sigmoid(s), ML.sigmoid(s)] : s; };

    function updateText() {
      const m = cur(), c = R.curve[m], sn = TR.scoreName(D), unit = isC() ? '' : TR.unitY(D) + ', ', card = MLS.statCard;
      $('story').innerHTML = '<h3>Data: ' + MLS.esc(D.label) + '</h3><p>' + ML.DATA[D.key].note + ' <b>' + D.train.length + ' ' + D.unit + ' in the training set, ' + D.test.length + ' in the test set.</b> ' + (isC() ? 'The model starts from a probability of ' + F(isC() ? ML.sigmoid(R.model.F0) : 0, 2) + ' for every ' + D.unit.replace(/s$/, '') + '.' : 'The model starts from ' + F(R.model.F0, 2) + ', the average of the training set, for every ' + D.unit.replace(/s$/, '') + '.') + ' Each round adds one small tree.</p>';
      $('stats').innerHTML =
        card(sn + ', training set', TR.fmtScore(D, c.tr), unit + 'after ' + m + ' round' + (m === 1 ? '' : 's'), '', 'st_score') +
        card(sn + ', test set', TR.fmtScore(D, c.te), unit + 'after ' + m + ' round' + (m === 1 ? '' : 's'), '', 'st_score') +
        card('Round', m + ' of ' + R.M, 'trees added so far', '', 'st_round') +
        card('Best round, test set', String(R.bestTe.m), sn + ' ' + TR.fmtScore(D, R.bestTe.te) + ' (for teaching)', '', 'st_best');
      const last = R.curve[R.M], worse = isC() ? R.bestTe.te - last.te : (last.te - R.bestTe.te) / R.bestTe.te;
      let msg, cls = 'status-line';
      if (m === 0) msg = 'Round 0: every ' + D.unit.replace(/s$/, '') + ' gets the starting value. Press Step to add the first tree, fitted to the errors.';
      else if (m === R.M && worse > (isC() ? 0.03 : 0.08)) { msg = 'After ' + R.M + ' rounds the training ' + (isC() ? 'accuracy keeps rising' : 'error keeps falling') + ', but the test result was best at round ' + R.bestTe.m + ': later trees fit noise (overfitting). In practice the number of rounds is chosen on validation data (early stopping).'; cls += ' bad'; }
      else { msg = 'Round ' + m + ': each new tree corrects part of the remaining error, scaled by the learning rate ' + F(R.model.eta, 2) + '. ' + sn + ' training ' + TR.fmtScore(D, c.tr) + ', test ' + TR.fmtScore(D, c.te) + '.'; cls += ' ok'; }
      $('statusLine').textContent = msg; $('statusLine').className = cls;
      $('stepNote').innerHTML = 'Round <b>' + m + '</b> of ' + R.M + '.';
    }

    const mapPlot = { aspect: 0.55, margin: { l: 66, b: 50 }, x: { label: () => D.names[0] }, y: { label: () => D.names[1] } };
    const figMain = MLS.figure($('figMain'), { title: 'The data and the model after this round', help: 'b_main', file: cfg.id + '-model', plot: mapPlot,
      data: () => ({ cols: isC() ? [D.names[0], D.names[1], 'Actual', 'Probability of ' + D.classes[1], 'Set'] : [D.names[0], D.names[1], 'Forecast', 'Set'], rows: D.pts.map(p => (isC() ? [p.x[0], p.x[1], D.classes[p.y], fAt(cur())(p.x)[1], p.te ? 'test' : 'training'] : [p.x[0], p.y, fAt(cur())(p.x), p.te ? 'test' : 'training'])) }) });
    function drawMain() {
      const m = cur();
      TR.drawData(figMain.plot, D, fAt(m), { res: 70, lineName: 'After round ' + m, prev: !isC() && m > 0 ? fAt(m - 1) : null });
      figMain.legend(TR.legendData(D, isC() ? [{ label: 'Boundary (probability 0.5)', color: MLS.color('chart-ink'), line: true }] : [{ label: 'Forecast after round ' + m, color: MLS.color('series-2'), line: true }].concat(m > 0 ? [{ label: 'After round ' + (m - 1), color: MLS.color('chart-muted'), line: true, dash: '5 4' }] : [])));
      figMain.caption(isC() ? 'Background: the class predicted after ' + m + ' rounds (paler = a probability closer to 0.5).' : 'Solid: the forecast after ' + m + ' rounds; dashed: one round earlier. The gap between them is the latest tree, scaled by the learning rate.');
    }
    const figRes = MLS.figure($('figRes'), { title: 'The errors that the next tree learns', help: 'b_res', file: cfg.id + '-residuals',
      plot: { aspect: 0.8, margin: { l: 62, b: 50 }, x: { label: () => D.names[0] }, y: { label: 'Error left: actual − forecast' } } });
    function drawRes() {
      const host = $('figRes'); host.hidden = isC(); if (isC()) return;
      const m = Math.max(1, cur()), Fp = R.Fhist[m - 1], t = R.model.trees[m - 1], P = figRes.plot, c1 = MLS.color('series-1'), c2 = MLS.color('series-2'), mu = MLS.color('chart-muted');
      $('figRes').querySelector('.figure-title').firstChild.textContent = 'Round ' + m + ': the errors left after round ' + (m - 1) + ', and tree ' + m;
      const res = D.train.map((p, i) => ({ x: p.x[0], y: p.y - Fp[i] })), ys = res.map(r => r.y), lim = Math.max(...ys.map(Math.abs)) * 1.1 || 1;
      P.setX(D.dom.x[0], D.dom.x[1]); P.setY(-lim, lim);
      P.draw(P => {
        P.hline(0, { color: mu, width: 1 });
        P.points(res, { color: c1, r: 3.6, opacity: 0.85, tip: d => D.names[0] + ' ' + F(d.x, 1) + '<br>Error left ' + F(d.y, 3) });
        const xs = []; for (let k = 0; k <= 400; k++) xs.push(D.dom.x[0] + (D.dom.x[1] - D.dom.x[0]) * k / 400);
        P.line(xs.map(x => [x, cfg.leafValue(t, [x])]), { color: c2, width: 2.4 });
        P.line(xs.map(x => [x, R.model.eta * cfg.leafValue(t, [x])]), { color: c2, width: 1.6, dash: '5 4' });
      });
      figRes.legend([{ label: 'Error left after round ' + (m - 1) + ' (training set)', color: c1 }, { label: 'Tree ' + m + ': fitted to these errors', color: c2, line: true }, { label: 'Learning rate × tree ' + m + ': the step taken', color: c2, line: true, dash: '5 4' }]);
      figRes.caption('Each round fits a small tree to what the model still gets wrong, then adds only a fraction (the learning rate) of it. As rounds go by, the errors shrink towards 0.');
    }
    const figErr = MLS.figure($('figErr'), { title: 'Training and test results by round', help: 'b_err', file: cfg.id + '-by-round',
      plot: { aspect: 0.8, margin: { l: 62, b: 50 }, x: { label: 'Round (trees added)' }, y: { label: () => TR.scoreName(D), fmt: v => (isC() ? (100 * v).toFixed(0) + '%' : F(v, 1)) } },
      data: () => ({ cols: ['Round', 'Training set', 'Test set'], rows: R.curve.map(c => [c.m, c.tr, c.te]) }) });
    function drawErr() {
      const P = figErr.plot, c1 = MLS.color('series-1'), c2 = MLS.color('series-2'), mu = MLS.color('chart-muted'), v = R.curve.flatMap(c => [c.tr, c.te]);
      P.setX(0, Math.max(1, R.M)); P.setY(isC() ? Math.max(0, Math.min(...v) - 0.05) : 0, (isC() ? 1.01 : Math.max(...v) * 1.05));
      P.draw(P => {
        P.vline(cur(), { color: mu, width: 1.5, dash: '4 3' });
        P.line(R.curve.map(c => [c.m, c.tr]), { color: c1, width: 2 }); P.line(R.curve.map(c => [c.m, c.te]), { color: c2, width: 2.4 });
        P.points([{ x: R.bestTe.m, y: R.bestTe.te }], { color: c2, shape: 'diamond', r: 7, ringColor: MLS.color('chart-ink'), tip: () => 'Best test result at round ' + R.bestTe.m + ': ' + TR.fmtScore(D, R.bestTe.te) });
        P.crosshair([{ name: 'Training set', color: c1, pts: R.curve.map(c => [c.m, c.tr]) }, { name: 'Test set', color: c2, pts: R.curve.map(c => [c.m, c.te]) }], x => 'Round ' + Math.round(x), y => TR.fmtScore(D, y));
      });
      figErr.legend([{ label: 'Training set', color: c1, line: true }, { label: 'Test set', color: c2, line: true }, { label: 'Best test round (' + R.bestTe.m + ')', color: c2, shape: 'diamond' }, { label: 'Current round', color: mu, line: true, dash: '4 3' }]);
      figErr.caption('The training result keeps improving with every tree; the test result improves, levels off and may get worse. The diamond is shown for teaching only: in practice the number of rounds is chosen on validation data, never on the test set.');
    }
    const figTree = MLS.figure($('figTree'), { title: 'The tree added in this round', help: 'b_tree', file: cfg.id + '-tree',
      plot: { height: w => (R ? TR.treeHeight(Math.min(4, depthOf(R.model.trees[Math.max(1, cur()) - 1])))(w) : 200), margin: { l: 8, r: 8, t: 10, b: 10 }, x: { ticks: false }, y: { ticks: false } } });
    const depthOf = t => { let d = 0; (function w(nd, k) { d = Math.max(d, k); if (nd.left) { w(nd.left, k + 1); w(nd.right, k + 1); } })(t.root || t, 0); return d; };
    function drawTree() {
      const m = Math.max(1, cur()), t = R.model.trees[m - 1], root = t.root || t;
      $('figTree').querySelector('.figure-title').firstChild.textContent = 'Tree ' + m + ', fitted in round ' + m;
      TR.drawTree(figTree.plot, root, { maxDepth: 4, lines: (nd, open) => cfg.treeLines(nd, open, D), tip: (nd, open) => cfg.treeTip(nd, open, D),
        fill: (nd, open) => (open ? MLS.color('surface') : MLS.rgba(MLS.color('series-2'), 0.14)) });
      figTree.caption(cfg.treeCaption(D, R, m));
    }
    function drawNumbers() { const h = cfg.numbers ? cfg.numbers(R, D, Math.max(1, cur())) : ''; $('numCard').hidden = !h; $('numCard').innerHTML = h; }

    /* controls */
    MLS.stepper($('stepper'), { speed: 4, runMax: 400,
      step: () => { const m = cur(); if (m >= R.M) return false; S.m = m + 1; return true; },
      reset: () => { S.m = 0; }, after: () => refresh(false) });
    $('params').innerHTML = cfg.params.map(d => '<div class="field-row"><label for="p_' + d.key + '">' + d.label + '<button type="button" class="help" data-help="' + d.help + '">?</button></label><output id="p_' + d.key + 'Val"></output></div>' +
      (d.list ? '<input type="range" id="p_' + d.key + '" min="0" max="' + (d.list.length - 1) + '" step="1">' : '<input type="range" id="p_' + d.key + '" min="' + d.min + '" max="' + d.max + '" step="' + (d.step || 1) + '">')).join('');
    cfg.params.forEach(d => {
      const el = $('p_' + d.key), val = () => (d.list ? d.list[+el.value] : +el.value);
      el.addEventListener('input', () => ($('p_' + d.key + 'Val').textContent = String(val())));
      el.onchange = () => { S[d.key] = val(); S.m = -1; refresh(); };
    });
    function syncUI() {
      MLS.$$('[data-task]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.task === S.task)));
      $('ds').innerHTML = LISTS[S.task].map(k => '<option value="' + k + '"' + (k === dkey() ? ' selected' : '') + '>' + TR.DATA[k].label + '</option>').join('');
      $('n').value = nObs(); $('nVal').textContent = nObs(); $('seed').value = S.seed;
      cfg.params.forEach(d => { const el = $('p_' + d.key); el.value = d.list ? Math.max(0, d.list.indexOf(S[d.key])) : S[d.key]; $('p_' + d.key + 'Val').textContent = String(S[d.key]); });
    }
    let ch = null;
    function refresh(re) { if (re !== false) compute(); syncUI(); updateText(); drawMain(); drawRes(); drawErr(); drawTree(); drawNumbers(); figMain.refreshTable(); if (ch) ch.check(); MLS.state.save(); }
    const full = () => { S.m = -1; refresh(); };
    MLS.$$('[data-task]').forEach(b => (b.onclick = () => { S.task = b.dataset.task; full(); }));
    $('ds').onchange = e => { S.data[S.task] = e.target.value; full(); };
    $('n').addEventListener('input', e => ($('nVal').textContent = e.target.value));
    $('n').onchange = e => { S.n[dkey()] = +e.target.value; full(); };
    $('seed').onchange = e => { S.seed = Math.max(1, Math.round(+e.target.value) || 1); full(); };
    $('newSeed').onclick = () => { S.seed = MLS.newSeed(); full(); };

    MLS.registerHelp(Object.assign({
      task: { title: 'Task and data', html: '<b>Regression</b>: electricity use from the temperature, or the electricity price from the gas price (1 input). <b>Classification</b>: bank failures or circles (2 inputs, 2 classes). Simulated data; 80% training set, 20% test set.' },
      n: { title: 'Observations', html: '80% form the training set on which the trees are fitted.' },
      steps: { title: 'Round by round', html: '<b>Reset</b>: round 0, the starting value only. <b>Step</b>: add the next tree. <b>Play</b>: add trees one after another. <b>End</b>: all rounds.' },
      st_score: { title: 'Accuracy or RMSE', html: 'After the current round. Classification: share classified correctly at probability 0.5. Regression: RMSE, in the units of the target.' },
      st_round: { title: 'Round', html: 'How many trees have been added so far.' },
      st_best: { title: 'Best round on the test set', html: 'The round with the best test result. Shown for teaching only: choosing it with the test set would be leakage; in practice use validation data (early stopping).' },
      b_main: { title: 'The model after this round', html: 'Regression: the forecast line after the current round (solid) and one round earlier (dashed). Classification: the predicted class (background) and the 0.5 boundary.' },
      b_res: { title: 'The errors the next tree learns', html: 'For regression: the errors left after the previous round (actual − forecast) on the training set, the tree fitted to them (solid) and the fraction actually added (dashed: learning rate × tree).' },
      b_err: { title: 'Results by round', html: 'Training and test results after each round. The training result keeps improving; the test result levels off or gets worse when later trees fit noise.' },
      b_tree: { title: 'The tree of this round', html: 'The small tree added in the current round. Its leaves hold the corrections added to the forecast (multiplied by the learning rate).' }
    }, cfg.help));

    function setup(o) { const m = o.m; delete o.m; Object.assign(S, o); MLS.pickTab('sim'); full(); if (m !== undefined) { S.m = m; refresh(false); } window.scrollTo({ top: 0, behavior: 'smooth' }); }
    const api = { S, setup, get R() { return R; }, get D() { return D; }, cur, isC: () => isC() };
    ch = MLS.challenges($('challenges'), META.id, cfg.challenges(api));
    MLS.moduleBar({ id: META.id, title: META.title, section: META.sectionTitle, level: META.level },
      { tabs: [{ id: 'sim', label: 'Simulation' }, { id: 'lesson', label: 'Lesson' }, { id: 'challenges', label: 'Challenges' }, { id: 'real', label: 'On real data' }] });
    const keys = cfg.params.map(d => d.key);
    MLS.state.init({
      id: META.id,
      get: () => Object.assign({ task: S.task, data: S.data, n: S.n, seed: S.seed }, Object.fromEntries(keys.map(k => [k, S[k]]))),
      set: o => {
        if (LISTS[o.task]) S.task = o.task;
        if (o.data) Object.keys(LISTS).forEach(t => { if (LISTS[t].includes(o.data[t])) S.data[t] = o.data[t]; });
        if (o.n) Object.keys(o.n).forEach(k => { if (TR.DATA[k] && Number.isFinite(o.n[k])) S.n[k] = Math.max(40, Math.min(400, o.n[k])); });
        if (Number.isFinite(o.seed)) S.seed = o.seed;
        cfg.params.forEach(d => { const v = o[d.key]; if (Number.isFinite(v) && (d.list ? d.list.includes(v) : v >= d.min && v <= d.max)) S[d.key] = v; });
      }
    });
    $('fourCast').href = MLS.APP.fourCastUrl;
    window.__mls = api;
    document.addEventListener('mls:theme', () => refresh(false));
    full();
  };
})();
