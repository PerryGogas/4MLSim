/* =====================================================================
   4MLSim statistics library
   Distributions (t, F, chi-square, normal), matrix algebra, OLS with
   classical and HC1 standard errors, fit statistics and diagnostics.
   Results are checked against statsmodels (Python) in the test suite.
   (c) 2026 Periklis Gogas, EMEF Lab. MIT License.
   ===================================================================== */
(function () {
  'use strict';
  const MLS = window.MLS;
  const D = (MLS.dist = {});

  /* ---------- special functions ---------- */
  function lgamma(x) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (let j = 0; j < 6; j++) ser += c[j] / ++y;
    return -tmp + Math.log(2.5066282746310005 * ser / x);
  }
  D.lgamma = lgamma;
  function betacf(a, b, x) {
    const MAXIT = 300, EPS = 3e-16, FPMIN = 1e-300;
    let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d; let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; const del = d * c; h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }
  /* regularized incomplete beta I_x(a, b) */
  function ibeta(x, a, b) {
    if (x <= 0) return 0; if (x >= 1) return 1;
    const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  }
  D.ibeta = ibeta;
  /* regularized lower incomplete gamma P(a, x) */
  function gammaP(a, x) {
    if (x <= 0) return 0;
    if (x < a + 1) {
      let ap = a, sum = 1 / a, del = sum;
      for (let n = 0; n < 500; n++) { ap++; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-16) break; }
      return sum * Math.exp(-x + a * Math.log(x) - lgamma(a));
    }
    let b = x + 1 - a, c = 1 / 1e-300, d = 1 / b, h = d;
    for (let i = 1; i < 500; i++) {
      const an = -i * (i - a); b += 2;
      d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
      c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
      d = 1 / d; const del = d * c; h *= del;
      if (Math.abs(del - 1) < 1e-16) break;
    }
    return 1 - Math.exp(-x + a * Math.log(x) - lgamma(a)) * h;
  }
  D.gammaP = gammaP;

  /* ---------- distributions ---------- */
  D.tCdf = (t, df) => { const x = df / (df + t * t); const p = 0.5 * ibeta(x, df / 2, 0.5); return t > 0 ? 1 - p : p; };
  D.tTwoSided = (t, df) => ibeta(df / (df + t * t), df / 2, 0.5);
  D.tQuantile = function (p, df) {
    /* bisection on the CDF (monotone), precise to ~1e-12 */
    let lo = -1000, hi = 1000;
    for (let i = 0; i < 200; i++) { const mid = (lo + hi) / 2; if (D.tCdf(mid, df) < p) lo = mid; else hi = mid; if (hi - lo < 1e-12) break; }
    return (lo + hi) / 2;
  };
  D.fSf = (f, d1, d2) => (f <= 0 ? 1 : ibeta(d2 / (d2 + d1 * f), d2 / 2, d1 / 2));
  D.chi2Sf = (x, k) => (x <= 0 ? 1 : 1 - gammaP(k / 2, x / 2));
  D.normCdf = z => 0.5 * (1 + erf(z / Math.SQRT2));
  function erf(x) {
    /* high-precision erf via the incomplete gamma function */
    const s = x < 0 ? -1 : 1; return s * gammaP(0.5, x * x);
  }
  D.erf = erf;
  D.normQuantile = function (p) {
    if (p <= 0) return -Infinity; if (p >= 1) return Infinity;
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const pl = 0.02425; let q, r, x;
    if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    else if (p <= 1 - pl) { q = p - 0.5; r = q * q; x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
    else { q = Math.sqrt(-2 * Math.log(1 - p)); x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    /* one Halley refinement step */
    const e = D.normCdf(x) - p, u = e * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
    return x - u / (1 + x * u / 2);
  };

  /* ---------- matrices (arrays of rows) ---------- */
  const M = (MLS.mat = {});
  M.t = A => A[0].map((_, j) => A.map(r => r[j]));
  M.mul = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
  M.mulVec = (A, v) => A.map(r => r.reduce((s, x, k) => s + x * v[k], 0));
  M.xtx = X => { const k = X[0].length, out = Array.from({ length: k }, () => new Array(k).fill(0)); for (const r of X) for (let i = 0; i < k; i++) { const ri = r[i]; for (let j = i; j < k; j++) out[i][j] += ri * r[j]; } for (let i = 0; i < k; i++) for (let j = 0; j < i; j++) out[i][j] = out[j][i]; return out; };
  M.xty = (X, y) => { const k = X[0].length, out = new Array(k).fill(0); X.forEach((r, n) => { for (let i = 0; i < k; i++) out[i] += r[i] * y[n]; }); return out; };
  /* inverse by Gauss-Jordan with partial pivoting; returns null if singular */
  M.inv = function (A) {
    const n = A.length, a = A.map((r, i) => r.concat(Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))));
    const scale = Math.max(...A.map(r => Math.max(...r.map(Math.abs)))) || 1;
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
      if (Math.abs(a[p][c]) < 1e-12 * scale) return null;
      [a[c], a[p]] = [a[p], a[c]];
      const piv = a[c][c]; for (let j = 0; j < 2 * n; j++) a[c][j] /= piv;
      for (let r = 0; r < n; r++) if (r !== c) { const f = a[r][c]; if (f) for (let j = 0; j < 2 * n; j++) a[r][j] -= f * a[c][j]; }
    }
    return a.map(r => r.slice(n));
  };
  M.eigSym2 = function (A) { const a = A[0][0], b = A[0][1], d = A[1][1], tr = a + d, det = a * d - b * b, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det)); return [tr / 2 + disc, tr / 2 - disc]; };

  /* ---------- OLS ----------
     X: rows with the constant included as first column; names: column names
     opts.cov: 'classical' | 'HC1'
     returns null if X'X is singular (perfect collinearity) */
  MLS.olsFit = function (X, y, names, opts) {
    opts = opts || {};
    const n = X.length, k = X[0].length;
    if (n <= k) return { error: 'Need more observations (' + n + ') than coefficients (' + k + ').' };
    const XtX = M.xtx(X), XtXi = M.inv(XtX);
    if (!XtXi) return { error: 'The X variables are perfectly collinear (one is an exact combination of the others), so OLS has no unique solution.' };
    const b = M.mulVec(XtXi, M.xty(X, y));
    const fitted = X.map(r => r.reduce((s, v, i) => s + v * b[i], 0));
    const resid = y.map((v, i) => v - fitted[i]);
    const ssr = resid.reduce((s, e) => s + e * e, 0);
    const my = y.reduce((s, v) => s + v, 0) / n;
    const sst = y.reduce((s, v) => s + (v - my) * (v - my), 0);
    const dfr = n - k, sigma2 = ssr / dfr;
    const hasConst = X.every(r => r[0] === 1);
    const r2 = hasConst ? 1 - ssr / sst : 1 - ssr / y.reduce((s, v) => s + v * v, 0);
    const adjR2 = 1 - (1 - r2) * (n - (hasConst ? 1 : 0)) / dfr;
    /* covariance */
    let V;
    if (opts.cov === 'HC1') {
      const meat = Array.from({ length: k }, () => new Array(k).fill(0));
      X.forEach((r, t) => { const e2 = resid[t] * resid[t]; for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) meat[i][j] += e2 * r[i] * r[j]; });
      V = M.mul(M.mul(XtXi, meat), XtXi).map(r => r.map(v => v * n / dfr));
    } else V = XtXi.map(r => r.map(v => v * sigma2));
    const se = b.map((_, i) => Math.sqrt(Math.max(0, V[i][i])));
    const tval = b.map((v, i) => v / se[i]);
    const p = tval.map(t => D.tTwoSided(t, dfr));
    const tc = D.tQuantile(0.975, dfr);
    const ci = b.map((v, i) => [v - tc * se[i], v + tc * se[i]]);
    /* overall F-test (all slopes zero), Wald form with the chosen covariance (as statsmodels) */
    let F = NaN, pF = NaN, q = k - (hasConst ? 1 : 0);
    if (q > 0) {
      const idx = Array.from({ length: k }, (_, i) => i).filter(i => !hasConst || i > 0);
      const bs = idx.map(i => b[i]), Vs = idx.map(i => idx.map(j => V[i][j]));
      const Vi = M.inv(Vs);
      if (Vi) { const w = bs.reduce((s, bi, i) => s + bi * Vi[i].reduce((t, v, j) => t + v * bs[j], 0), 0); F = w / q; pF = D.fSf(F, q, dfr); }
    }
    const llf = -n / 2 * (Math.log(2 * Math.PI) + Math.log(ssr / n) + 1);
    const aic = -2 * llf + 2 * k, bic = -2 * llf + k * Math.log(n);
    return { n, k, names, b, se, t: tval, p, ci, V, fitted, resid, ssr, sst, r2, adjR2, sigma: Math.sqrt(sigma2), dfr, F, pF, q, llf, aic, bic, cov: opts.cov || 'classical', XtXi };
  };
  MLS.olsPredict = (fit, X) => X.map(r => r.reduce((s, v, i) => s + v * fit.b[i], 0));

  /* ---------- diagnostics ---------- */
  /* Breusch-Pagan (Koenker studentized version, as statsmodels het_breuschpagan default): LM = n * R² of e² on X */
  MLS.breuschPagan = function (X, resid) {
    const e2 = resid.map(e => e * e);
    const aux = MLS.olsFit(X, e2, null, {});
    if (!aux || aux.error) return null;
    const df = X[0].length - 1, lm = X.length * aux.r2;
    return { lm, df, p: D.chi2Sf(lm, df) };
  };
  /* Jarque-Bera (as statsmodels jarque_bera) */
  MLS.jarqueBera = function (resid) {
    const n = resid.length, m = resid.reduce((s, v) => s + v, 0) / n;
    let m2 = 0, m3 = 0, m4 = 0;
    resid.forEach(v => { const d = v - m; m2 += d * d; m3 += d * d * d; m4 += d * d * d * d; });
    m2 /= n; m3 /= n; m4 /= n;
    const skew = m3 / Math.pow(m2, 1.5), kurt = m4 / (m2 * m2);
    const jb = n / 6 * (skew * skew + (kurt - 3) * (kurt - 3) / 4);
    return { jb, skew, kurt, p: D.chi2Sf(jb, 2) };
  };
  /* prediction metrics */
  MLS.metrics = function (y, yhat) {
    const n = y.length; if (!n) return { rmse: NaN, mae: NaN, r2: NaN, n: 0 };
    let se = 0, ae = 0; const my = y.reduce((s, v) => s + v, 0) / n; let st = 0;
    y.forEach((v, i) => { const e = v - yhat[i]; se += e * e; ae += Math.abs(e); st += (v - my) * (v - my); });
    return { n, rmse: Math.sqrt(se / n), mae: ae / n, mse: se / n, r2: st > 0 ? 1 - se / st : NaN };
  };

  /* =================================================================
     least squares by Householder QR (stable for polynomial bases)
     X: rows (n × k), y: n. Returns { b, rank, ssr }. Columns that are
     numerically dependent on earlier ones get coefficient 0.
     ================================================================= */
  MLS.lstsq = function (X, y) {
    const n = X.length, k = X[0].length;
    const A = X.map(r => r.slice()), z = y.slice();
    const piv = [], rank = { v: 0 };
    const colNorm = j => Math.sqrt(A.reduce((s, r) => s + r[j] * r[j], 0));
    const scale = Math.max(1e-300, ...Array.from({ length: k }, (_, j) => colNorm(j)));
    const used = new Array(k).fill(false);
    let row = 0;
    for (let j = 0; j < k && row < n; j++) {
      let nrm = 0; for (let i = row; i < n; i++) nrm += A[i][j] * A[i][j];
      nrm = Math.sqrt(nrm);
      if (nrm < 1e-11 * scale) continue;                 /* dependent column */
      const alpha = A[row][j] > 0 ? -nrm : nrm;
      const v = new Array(n).fill(0);
      for (let i = row; i < n; i++) v[i] = A[i][j];
      v[row] -= alpha;
      const vv = v.reduce((s, t, i) => (i >= row ? s + t * t : s), 0);
      if (vv > 0) {
        for (let c = j; c < k; c++) { let d = 0; for (let i = row; i < n; i++) d += v[i] * A[i][c]; d = 2 * d / vv; for (let i = row; i < n; i++) A[i][c] -= d * v[i]; }
        let d = 0; for (let i = row; i < n; i++) d += v[i] * z[i]; d = 2 * d / vv; for (let i = row; i < n; i++) z[i] -= d * v[i];
      }
      piv.push([row, j]); used[j] = true; row++;
    }
    const b = new Array(k).fill(0);
    for (let t = piv.length - 1; t >= 0; t--) {
      const [r, j] = piv[t]; let sm = z[r];
      for (let u = t + 1; u < piv.length; u++) { const c = piv[u][1]; sm -= A[r][c] * b[c]; }
      b[j] = sm / A[r][j];
    }
    let ssr = 0; for (let i = row; i < n; i++) ssr += z[i] * z[i];
    return { b, rank: piv.length, ssr };
  };

  /* Legendre polynomials P0..Pd at t in [-1, 1] (a well-conditioned polynomial basis) */
  MLS.legendre = function (t, d) {
    const p = [1]; if (d >= 1) p.push(t);
    for (let m = 2; m <= d; m++) p.push(((2 * m - 1) * t * p[m - 1] - (m - 1) * p[m - 2]) / m);
    return p;
  };
  /* polynomial model of degree d fitted by least squares on x mapped to [-1, 1] over [lo, hi] */
  MLS.polyFit = function (x, y, d, lo, hi) {
    const tf = v => (2 * v - lo - hi) / (hi - lo);
    const X = x.map(v => MLS.legendre(tf(v), d));
    const r = MLS.lstsq(X, y);
    const f = v => { const p = MLS.legendre(tf(v), d); let s = 0; for (let i = 0; i <= d; i++) s += p[i] * r.b[i]; return s; };
    /* coefficients in the ordinary power basis of t (for display): convert Legendre -> monomials */
    const mono = new Array(d + 1).fill(0);
    const P = [[1], [0, 1]];
    for (let m = 2; m <= d; m++) { const a = new Array(m + 1).fill(0); for (let i = 0; i < m; i++) a[i + 1] += (2 * m - 1) / m * P[m - 1][i]; for (let i = 0; i <= m - 2; i++) a[i] -= (m - 1) / m * P[m - 2][i]; P.push(a); }
    for (let m = 0; m <= d; m++) for (let i = 0; i < P[m].length; i++) mono[i] += r.b[m] * P[m][i];
    return { f, b: r.b, rank: r.rank, ssr: r.ssr, k: d + 1, mono, tf };
  };

  /* ---------- scaling ---------- */
  /* column means and population standard deviations (as scikit-learn StandardScaler) */
  MLS.colStats = function (X) {
    const n = X.length, p = X[0].length, mu = new Array(p).fill(0), sd = new Array(p).fill(0);
    X.forEach(r => r.forEach((v, j) => (mu[j] += v / n)));
    X.forEach(r => r.forEach((v, j) => (sd[j] += (v - mu[j]) * (v - mu[j]) / n)));
    return { mu, sd: sd.map(v => Math.sqrt(v) || 1) };
  };
  MLS.applyScale = (X, st) => X.map(r => r.map((v, j) => (v - st.mu[j]) / st.sd[j]));

  /* =================================================================
     elastic net on standardized features (scikit-learn ElasticNet / glmnet form)
       minimize (1/2n)·||y − ȳ − Z b||² + λ·[ α·||b||₁ + (1−α)/2·||b||² ]
     α = 0 ridge, α = 1 lasso. Z: rows with centred columns. Returns b (no intercept).
     opts: { b0: warm start, tol, maxIter }
     ================================================================= */
  MLS.elasticNet = function (Z, y, lambda, alpha, opts) {
    opts = opts || {};
    const n = Z.length, p = Z[0].length, my = y.reduce((s, v) => s + v, 0) / n;
    const yc = y.map(v => v - my);
    const cols = Array.from({ length: p }, (_, j) => Z.map(r => r[j]));
    const zz = cols.map(c => c.reduce((s, v) => s + v * v, 0) / n);
    if (alpha <= 0) {
      /* ridge: closed form (Z'Z/n + λI) b = Z'y/n */
      const A = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => cols[i].reduce((s, v, t) => s + v * cols[j][t], 0) / n + (i === j ? lambda : 0)));
      const rhs = cols.map(c => c.reduce((s, v, t) => s + v * yc[t], 0) / n);
      const Ai = M.inv(A);
      const b = Ai ? M.mulVec(Ai, rhs) : new Array(p).fill(0);
      return { b, intercept: my, iter: 0 };
    }
    const b = opts.b0 ? opts.b0.slice() : new Array(p).fill(0);
    const r = yc.slice();
    for (let j = 0; j < p; j++) if (b[j]) for (let t = 0; t < n; t++) r[t] -= cols[j][t] * b[j];
    const l1 = lambda * alpha, l2 = lambda * (1 - alpha), tol = opts.tol || 1e-9, maxIter = opts.maxIter || 5000;
    let it = 0;
    for (; it < maxIter; it++) {
      let maxd = 0, maxb = 0;
      for (let j = 0; j < p; j++) {
        const c = cols[j], old = b[j];
        let rho = 0; for (let t = 0; t < n; t++) rho += c[t] * r[t];
        rho = rho / n + zz[j] * old;
        const nb = (rho > l1 ? rho - l1 : rho < -l1 ? rho + l1 : 0) / (zz[j] + l2);
        if (nb !== old) { const d = nb - old; for (let t = 0; t < n; t++) r[t] -= c[t] * d; b[j] = nb; maxd = Math.max(maxd, Math.abs(d)); }
        maxb = Math.max(maxb, Math.abs(nb));
      }
      if (maxd <= tol * Math.max(1, maxb)) break;
    }
    return { b, intercept: my, iter: it + 1 };
  };
  /* smallest λ at which the lasso (or elastic net with mixing α) sets every coefficient to zero */
  MLS.lambdaMax = function (Z, y, alpha) {
    const n = Z.length, my = y.reduce((s, v) => s + v, 0) / n;
    let m = 0; for (let j = 0; j < Z[0].length; j++) { let s = 0; for (let t = 0; t < n; t++) s += Z[t][j] * (y[t] - my); m = Math.max(m, Math.abs(s) / n); }
    return m / Math.max(alpha, 1e-3);
  };

  /* ---------- resampling ---------- */
  /* fold label (0..k-1) for each of n observations, balanced and shuffled with rng */
  MLS.kfold = function (n, k, rng) {
    const idx = Array.from({ length: n }, (_, i) => i); rng.shuffle(idx);
    const f = new Array(n); idx.forEach((v, i) => (f[v] = i % k)); return f;
  };
  MLS.mean = a => a.reduce((s, v) => s + v, 0) / (a.length || 1);
  MLS.rmse = (y, yh) => Math.sqrt(y.reduce((s, v, i) => s + (v - yh[i]) * (v - yh[i]), 0) / (y.length || 1));
  MLS.stars = p => (p < 0.01 ? '***' : p < 0.05 ? '**' : p < 0.1 ? '*' : '');
  MLS.fmtP = p => (!Number.isFinite(p) ? 'n/a' : p < 0.0001 ? '< 0.0001' : p.toFixed(4));
})();
