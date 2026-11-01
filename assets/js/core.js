/* =====================================================================
   4MLSim core engine
   Shared by every page: branding, theme, math levels, seeded random numbers,
   data import (CSV/XLSX), share links, remembered settings, help pop-ups,
   challenges, stepper + pseudocode, exports, version notice, install.
   (c) 2026 Periklis Gogas, EMEF Lab. MIT License.
   ===================================================================== */
(function () {
  'use strict';
  const MLS = (window.MLS = window.MLS || {});

  MLS.APP = {
    name: '4MLSim',
    longName: 'Machine Learning Simulation Lab',
    version: '1.0.0', stateVersion: 16, buildNo: 16,
    pill: 'v1.0.0 b16',
    build: 'Release 1.0.0 (build 16): 33 simulations and 2 reference tools',
    author: 'Professor Periklis Gogas',
    repo: 'PerryGogas/4MLSim',
    siteUrl: 'https://perrygogas.github.io/4MLSim/',
    emefUrl: 'https://sites.google.com/site/duthemef/home',
    fourCastUrl: 'https://github.com/PerryGogas/4Cast-Lab/releases',
    year: 2026
  };
  try { if (!/4MLSim b\d/.test(document.title)) document.title = document.title.replace('4MLSim', '4MLSim b' + MLS.APP.buildNo); } catch (e) { /* ignore */ }

  /* ---------------------------------------------------------------
     paths (works for /index.html, /modules/x.html and the offline bundle)
     --------------------------------------------------------------- */
  MLS.inBundle = !!window.MLS_IN_BUNDLE;
  MLS.base = (function () {
    if (typeof window.MLS_BASE === 'string') return window.MLS_BASE;
    const s = document.currentScript && document.currentScript.src;
    if (s) return new URL('../../', s).href;
    return './';
  })();
  MLS.asset = function (path) {
    const b = window.MLS_BUNDLE_ASSETS;
    if (b && b[path]) return b[path];
    return MLS.base + path;
  };
  MLS.pageUrl = function (path) { return MLS.base + path; };

  /* ---------------------------------------------------------------
     safe storage (private windows can throw)
     --------------------------------------------------------------- */
  const PFX = '4mlsim:';
  MLS.store = {
    get(key, dflt) {
      try { const v = localStorage.getItem(PFX + key); return v === null ? dflt : JSON.parse(v); } catch (e) { return dflt; }
    },
    set(key, val) { try { localStorage.setItem(PFX + key, JSON.stringify(val)); } catch (e) { /* ignore */ } },
    del(key) { try { localStorage.removeItem(PFX + key); } catch (e) { /* ignore */ } }
  };

  /* ---------------------------------------------------------------
     theme (applied immediately to avoid a flash)
     --------------------------------------------------------------- */
  MLS.getTheme = function () { return document.documentElement.getAttribute('data-theme') || 'light'; };
  MLS.setTheme = function (t, persist) {
    document.documentElement.setAttribute('data-theme', t);
    if (persist !== false) MLS.store.set('theme', t);
    const b = document.getElementById('mlsThemeBtn');
    if (b) b.innerHTML = t === 'dark' ? '&#9788; Light' : '&#9790; Dark';
    document.dispatchEvent(new CustomEvent('mls:theme', { detail: t }));
  };
  (function initTheme() {
    let t = MLS.store.get('theme', null);
    if (!t) { try { t = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'; } catch (e) { t = 'light'; } }
    document.documentElement.setAttribute('data-theme', t);
  })();
  MLS.css = function (name, el) {
    return getComputedStyle(el || document.documentElement).getPropertyValue(name).trim();
  };

  /* ---------------------------------------------------------------
     small DOM helpers
     --------------------------------------------------------------- */
  MLS.$ = (sel, root) => (root || document).querySelector(sel);
  MLS.$$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  MLS.el = function (tag, attrs, html) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === 'class') e.className = attrs[k];
      else if (k === 'style') e.style.cssText = attrs[k];
      else if (k.startsWith('on') && typeof attrs[k] === 'function') e.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] !== undefined && attrs[k] !== null && attrs[k] !== false) e.setAttribute(k, attrs[k]);
    }
    if (html !== undefined) e.innerHTML = html;
    return e;
  };
  MLS.esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  MLS.debounce = function (fn, ms) { let t; return function () { clearTimeout(t); const a = arguments; t = setTimeout(() => fn.apply(this, a), ms); }; };

  /* number formatting with a true minus sign */
  MLS.fmt = function (x, d) {
    if (x === null || x === undefined || Number.isNaN(x)) return 'n/a';
    if (!Number.isFinite(x)) return x > 0 ? '∞' : '−∞';
    if (d === undefined) {
      const a = Math.abs(x);
      d = a >= 1000 ? 0 : a >= 100 ? 1 : a >= 1 ? 2 : a >= 0.01 ? 3 : 4;
      if (a !== 0 && (a >= 1e7 || a < 1e-4)) return x.toExponential(2).replace('-', '−');
    }
    const s = x.toFixed(d);
    return (s.startsWith('-') ? '−' + s.slice(1) : s).replace(/^−(0\.?0*)$/, '$1');
  };
  MLS.fmtInt = x => Math.round(x).toLocaleString('en-US');
  /* significant-digit format for coefficients of any size (e.g. 0.000512, 12.35, 1234) */
  MLS.fmtSig = function (x, sig) {
    sig = sig || 4;
    if (!Number.isFinite(x)) return MLS.fmt(x);
    if (x === 0) return '0';
    const a = Math.abs(x);
    if (a >= 1e6 || a < 1e-5) return x.toExponential(sig - 1).replace('-', '−');
    const d = Math.max(0, sig - 1 - Math.floor(Math.log10(a)));
    const s = x.toFixed(Math.min(d, 8));
    return s.startsWith('-') ? '−' + s.slice(1) : s;
  };

  /* ---------------------------------------------------------------
     seeded random numbers (mulberry32 + Box-Muller)
     --------------------------------------------------------------- */
  MLS.rng = function (seed) {
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
      let u = 0, v = 0;
      while (u === 0) u = random();
      v = random();
      const r = Math.sqrt(-2 * Math.log(u));
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
  MLS.newSeed = () => 1 + Math.floor(Math.random() * 99999);

  /* ---------------------------------------------------------------
     basic statistics
     --------------------------------------------------------------- */
  MLS.stat = {
    mean: a => a.reduce((s, v) => s + v, 0) / a.length,
    variance(a) { const m = MLS.stat.mean(a); return a.reduce((s, v) => s + (v - m) * (v - m), 0) / a.length; },
    sd: a => Math.sqrt(MLS.stat.variance(a)),
    min: a => a.reduce((m, v) => (v < m ? v : m), Infinity),
    max: a => a.reduce((m, v) => (v > m ? v : m), -Infinity),
    /* simple OLS y = b0 + b1 x */
    ols(x, y) {
      const n = x.length, mx = MLS.stat.mean(x), my = MLS.stat.mean(y);
      let sxy = 0, sxx = 0;
      for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
      const b1 = sxx > 0 ? sxy / sxx : 0;
      return { b0: my - b1 * mx, b1 };
    },
    mse(x, y, b0, b1) { let s = 0; for (let i = 0; i < x.length; i++) { const e = y[i] - b0 - b1 * x[i]; s += e * e; } return s / x.length; },
    r2(x, y, b0, b1) { const v = MLS.stat.variance(y); return v > 0 ? 1 - MLS.stat.mse(x, y, b0, b1) / v : 0; }
  };

  /* ---------------------------------------------------------------
     toast, banners, tooltip
     --------------------------------------------------------------- */
  MLS.toast = function (msg, ms) {
    const t = MLS.el('div', { class: 'toast', role: 'status' }, MLS.esc(msg));
    document.body.appendChild(t);
    setTimeout(() => t.remove(), ms || 2600);
  };
  MLS.banner = function (html, type, id) {
    let host = document.getElementById('mlsBanners');
    if (!host) { host = MLS.el('div', { id: 'mlsBanners' }); const h = document.getElementById('mlsHeader'); if (h) h.after(host); else document.body.prepend(host); }
    if (id && document.getElementById(id)) document.getElementById(id).remove();
    const b = MLS.el('div', { class: 'banner' + (type === 'info' ? ' info' : ''), id: id || null, role: 'status' });
    b.innerHTML = '<span>' + html + '</span>';
    const x = MLS.el('button', { class: 'x', type: 'button', 'aria-label': 'Dismiss' }, '&times;');
    x.onclick = () => b.remove();
    b.appendChild(x);
    host.appendChild(b);
    return b;
  };
  MLS.tip = {
    el: null,
    show(html, x, y) {
      if (!this.el) { this.el = MLS.el('div', { id: 'mlsTip', role: 'tooltip' }); document.body.appendChild(this.el); }
      const t = this.el; t.innerHTML = html; t.style.display = 'block';
      const w = t.offsetWidth, h = t.offsetHeight;
      let lx = x + 14, ly = y + 14;
      if (lx + w > window.innerWidth - 6) lx = x - w - 14;
      if (ly + h > window.innerHeight - 6) ly = y - h - 14;
      t.style.left = Math.max(4, lx) + 'px'; t.style.top = Math.max(4, ly) + 'px';
    },
    hide() { if (this.el) this.el.style.display = 'none'; }
  };

  /* ---------------------------------------------------------------
     "?" help pop-ups: show on hover, pin on click (like 4CorrLab)
     --------------------------------------------------------------- */
  MLS.helpText = {};
  MLS.registerHelp = function (obj) { Object.assign(MLS.helpText, obj); };
  (function initHelp() {
    /* Hover shows a preview; a click pins it open until you click anywhere else (or press Esc),
       exactly as in 4Cast Lab. Links inside a pinned pop-up can be clicked. */
    let pinned = null, pop = null, hideT = null;
    function ensure() {
      if (!pop) {
        pop = MLS.el('div', { id: 'mlsHelpPop', role: 'dialog' });
        pop.addEventListener('mouseenter', () => clearTimeout(hideT));
        pop.addEventListener('mouseleave', () => { if (!pinned) hideSoon(); });
        document.body.appendChild(pop);
      }
      return pop;
    }
    function place(btn) {
      const key = btn.getAttribute('data-help');
      const h0 = MLS.helpText[key], h = h0 && { title: typeof h0.title === 'function' ? h0.title() : h0.title, html: typeof h0.html === 'function' ? h0.html() : h0.html };
      const p = ensure();
      p.innerHTML = h ? (h.title ? '<h5>' + h.title + '</h5>' : '') + '<div>' + h.html + '</div>' + (pinned === btn ? '' : '<div class="help-hint">Click the ? to keep this open.</div>')
                      : '<div class="note">No explanation yet for “' + MLS.esc(key) + '”.</div>';
      p.style.display = 'block';
      MLS.renderMath(p);
      const r = btn.getBoundingClientRect();
      const w = Math.min(380, window.innerWidth - 16);
      p.style.maxWidth = w + 'px';
      let left = r.left + window.scrollX - 12;
      if (left + p.offsetWidth > window.scrollX + window.innerWidth - 8) left = window.scrollX + window.innerWidth - p.offsetWidth - 8;
      p.style.left = Math.max(8, left) + 'px';
      let top = r.bottom + window.scrollY + 8;
      if (r.bottom + p.offsetHeight + 16 > window.innerHeight && r.top - p.offsetHeight - 8 > 0) top = r.top + window.scrollY - p.offsetHeight - 8;
      p.style.top = top + 'px';
    }
    function hide() { clearTimeout(hideT); if (pop) pop.style.display = 'none'; }
    function hideSoon() { clearTimeout(hideT); hideT = setTimeout(hide, 220); }
    function unpin() { if (pinned) pinned.classList.remove('open'); pinned = null; hide(); }
    document.addEventListener('mouseover', e => { const b = e.target.closest && e.target.closest('.help'); if (b && !pinned) { clearTimeout(hideT); place(b); } });
    document.addEventListener('mouseout', e => { const b = e.target.closest && e.target.closest('.help'); if (b && !pinned) hideSoon(); });
    document.addEventListener('click', e => {
      const b = e.target.closest && e.target.closest('.help');
      if (b) {
        e.preventDefault(); e.stopPropagation();
        if (pinned === b) { unpin(); return; }
        if (pinned) pinned.classList.remove('open');
        pinned = b; b.classList.add('open'); place(b); return;
      }
      if (pinned && pop && !pop.contains(e.target)) unpin();
    }, true);
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && pinned) unpin(); });
  })();
  MLS.helpBtn = key => '<button type="button" class="help" data-help="' + key + '" aria-label="Help">?</button>';

  /* ---------------------------------------------------------------
     math rendering (TeX -> MathML via Temml, fully offline)
     usage: <span class="tex">\beta_0</span>  or  <div class="tex-block">...</div>
     --------------------------------------------------------------- */
  MLS.renderMath = function (root) {
    if (!window.temml) return;
    MLS.$$('.tex, .tex-block', root || document).forEach(el => {
      if (el.dataset.rendered) return;
      const src = el.dataset.tex || el.textContent;
      el.dataset.tex = src;
      try { window.temml.render(src, el, { displayMode: el.classList.contains('tex-block'), throwOnError: false }); el.dataset.rendered = '1'; }
      catch (e) { el.textContent = src; }
    });
  };
  MLS.tex = function (src, display) {
    if (!window.temml) return MLS.esc(src);
    try { return window.temml.renderToString(src, { displayMode: !!display, throwOnError: false }); } catch (e) { return MLS.esc(src); }
  };

  /* ---------------------------------------------------------------
     math level: 0 = intuition, 1 = formulas, 2 = deep dive
     --------------------------------------------------------------- */
  MLS.getMath = () => Number(document.body.dataset.math || 0);
  MLS.setMath = function (lvl, persist) {
    document.body.dataset.math = String(lvl);
    if (persist !== false) MLS.store.set('mathLevel', lvl);
    MLS.$$('[data-math-btn]').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.mathBtn) === lvl)));
    MLS.renderMath();
    document.dispatchEvent(new CustomEvent('mls:math', { detail: lvl }));
  };

  /* ---------------------------------------------------------------
     navigation (normal site and offline single-file bundle)
     --------------------------------------------------------------- */
  MLS.go = function (path) {
    if (MLS.inBundle && window.parent && window.parent !== window) { window.parent.postMessage({ mls: 'nav', href: path }, '*'); return; }
    location.href = MLS.base + path;
  };
  document.addEventListener('click', e => {
    if (!MLS.inBundle) return;
    const a = e.target.closest && e.target.closest('a[data-nav]');
    if (a) { e.preventDefault(); MLS.go(a.getAttribute('data-nav')); }
  });
  MLS.navAttrs = path => 'href="' + MLS.base + path + '" data-nav="' + path + '"';

  /* ---------------------------------------------------------------
     header & footer
     --------------------------------------------------------------- */
  MLS.renderHeader = function () {
    const h = document.getElementById('mlsHeader');
    if (!h) return;
    const compact = h.dataset.variant === 'compact';
    const A = MLS.APP;
    h.className = 'app-header' + (compact ? ' compact' : '');
    h.innerHTML =
      '<a ' + MLS.navAttrs('index.html') + ' aria-label="4MLSim home"><img class="app-logo" src="' + MLS.asset('assets/img/icon.svg') + '" alt=""></a>' +
      '<div class="app-title-block">' +
        '<h1 class="app-name"><a ' + MLS.navAttrs('index.html') + '>4ML<span class="lab">Sim</span></a><span class="version-pill" title="' + MLS.esc(A.build) + '">' + A.pill + '</span></h1>' +
        '<span class="app-byline">' + A.longName + ' &bull; by ' + A.author + '</span>' +
      '</div>' +
      '<div class="header-right">' +
        '<button id="mlsInstallBtn" class="btn-navy btn-small" type="button" hidden title="Install 4MLSim on this computer so it opens in its own window, also offline">&#11015; Install 4MLSim</button>' +
        '<button id="mlsThemeBtn" class="btn-ghost btn-small" type="button" title="Switch between light and dark theme"></button>' +
        '<a class="emef-link" href="' + A.emefUrl + '" target="_blank" rel="noopener" title="EMEF Research Lab"><img src="' + MLS.asset('assets/img/emef-lab-logo.png') + '" alt="E|M|E|F Research Lab"></a>' +
      '</div>';
    const tb = document.getElementById('mlsThemeBtn');
    tb.innerHTML = MLS.getTheme() === 'dark' ? '&#9788; Light' : '&#9790; Dark';
    tb.onclick = () => MLS.setTheme(MLS.getTheme() === 'dark' ? 'light' : 'dark');
    const ib = document.getElementById('mlsInstallBtn');
    ib.onclick = MLS.install;
    if (MLS._installEvent) ib.hidden = false;
  };
  MLS.renderFooter = function () {
    const f = document.getElementById('mlsFooter');
    if (!f) return;
    const A = MLS.APP;
    f.className = 'app-footer';
    f.innerHTML = '<span>&copy; ' + A.year + ' Periklis Gogas &bull; EMEF Lab, Department of Economics, Democritus University of Thrace</span>' +
      '<span>MIT License</span><span>v' + A.version + '</span>' +
      '<a ' + MLS.navAttrs('about.html') + '>About &amp; how to cite</a>' +
      '<a href="https://github.com/' + A.repo + '" target="_blank" rel="noopener">GitHub</a>';
  };

  /* ---------------------------------------------------------------
     install (PWA) and offline helper
     --------------------------------------------------------------- */
  window.addEventListener('beforeinstallprompt', e => {
    e.preventDefault();
    MLS._installEvent = e;
    const b = document.getElementById('mlsInstallBtn');
    if (b) b.hidden = false;
  });
  window.addEventListener('appinstalled', () => {
    MLS._installEvent = null;
    const b = document.getElementById('mlsInstallBtn');
    if (b) b.hidden = true;
  });
  MLS.install = async function () {
    const ev = MLS._installEvent;
    if (!ev) { MLS.banner('To install, use the browser menu: in Edge <b>Apps &gt; Install this site as an app</b>; in Chrome <b>Cast, save and share &gt; Install page as app</b>.', 'info'); return; }
    ev.prompt();
    try { await ev.userChoice; } catch (e) { /* ignore */ }
    MLS._installEvent = null;
    const b = document.getElementById('mlsInstallBtn'); if (b) b.hidden = true;
  };
  if (!MLS.inBundle && 'serviceWorker' in navigator && (location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1')) {
    window.addEventListener('load', () => { navigator.serviceWorker.register(MLS.base + 'sw.js').catch(() => { /* offline support unavailable */ }); });
  }

  /* ---------------------------------------------------------------
     new-version notice (checks GitHub releases at most twice a day)
     --------------------------------------------------------------- */
  function cmpVer(a, b) {
    const pa = String(a).replace(/^v/i, '').split(/[.-]/).map(n => parseInt(n, 10) || 0);
    const pb = String(b).replace(/^v/i, '').split(/[.-]/).map(n => parseInt(n, 10) || 0);
    for (let i = 0; i < 3; i++) { if ((pa[i] || 0) !== (pb[i] || 0)) return (pa[i] || 0) - (pb[i] || 0); }
    return 0;
  }
  MLS.cmpVer = cmpVer;
  MLS.checkVersion = async function () {
    if (MLS.inBundle || !navigator.onLine || !/^https?:$/.test(location.protocol)) return;
    const now = Date.now();
    let info = MLS.store.get('latestRelease', null);
    if (!info || now - info.t > 12 * 3600 * 1000) {
      try {
        const ctl = new AbortController(); const tm = setTimeout(() => ctl.abort(), 6000);
        const r = await fetch('https://api.github.com/repos/' + MLS.APP.repo + '/releases/latest', { signal: ctl.signal, headers: { Accept: 'application/vnd.github+json' } });
        clearTimeout(tm);
        if (!r.ok) { MLS.store.set('latestRelease', { t: now, tag: null }); return; }
        const j = await r.json();
        info = { t: now, tag: j.tag_name, url: j.html_url };
        MLS.store.set('latestRelease', info);
      } catch (e) { return; }
    }
    if (info && info.tag && cmpVer(info.tag, MLS.APP.version) > 0 && MLS.store.get('dismissedVersion', '') !== info.tag) {
      const b = MLS.banner('A newer version of 4MLSim is available (<b>' + MLS.esc(info.tag) + '</b>; you have v' + MLS.APP.version + '). ' +
        '<a href="#" id="mlsReload">Reload to update</a> &bull; <a href="' + info.url + '" target="_blank" rel="noopener">What is new</a>', 'info', 'mlsVersionBanner');
      const rl = document.getElementById('mlsReload');
      if (rl) rl.onclick = async ev => {
        ev.preventDefault();
        try { const regs = await navigator.serviceWorker.getRegistrations(); for (const g of regs) await g.update(); } catch (e) { /* ignore */ }
        location.reload();
      };
      b.querySelector('.x').addEventListener('click', () => MLS.store.set('dismissedVersion', info.tag));
    }
  };

  /* ---------------------------------------------------------------
     lazy script loading (SheetJS is only loaded when needed)
     --------------------------------------------------------------- */
  const loaded = {};
  MLS.loadScript = function (path) {
    if (loaded[path]) return loaded[path];
    loaded[path] = new Promise((resolve, reject) => {
      const src = window.MLS_BUNDLE_SCRIPTS && window.MLS_BUNDLE_SCRIPTS[path];
      const s = document.createElement('script');
      if (src) { s.textContent = src; document.head.appendChild(s); resolve(); return; }
      s.src = MLS.base + path; s.onload = () => resolve(); s.onerror = () => reject(new Error('Could not load ' + path));
      document.head.appendChild(s);
    });
    return loaded[path];
  };
  MLS.needXLSX = () => (window.XLSX ? Promise.resolve() : MLS.loadScript('assets/vendor/xlsx.core.min.js'));

  /* ---------------------------------------------------------------
     downloads & exports
     --------------------------------------------------------------- */
  MLS.download = function (blobOrUrl, filename) {
    const url = typeof blobOrUrl === 'string' ? blobOrUrl : URL.createObjectURL(blobOrUrl);
    const a = MLS.el('a', { href: url, download: filename });
    document.body.appendChild(a); a.click(); a.remove();
    if (typeof blobOrUrl !== 'string') setTimeout(() => URL.revokeObjectURL(url), 4000);
  };
  /* sheets: { 'Sheet name': [[header...], [row...], ...] } */
  MLS.exportXLSX = async function (sheets, filename) {
    try {
      await MLS.needXLSX();
      const wb = window.XLSX.utils.book_new();
      for (const name in sheets) window.XLSX.utils.book_append_sheet(wb, window.XLSX.utils.aoa_to_sheet(sheets[name]), name.slice(0, 31));
      const out = window.XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
      MLS.download(new Blob([out], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }), filename);
    } catch (e) { MLS.toast('Export failed: ' + e.message); }
  };
  MLS.slug = s => String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

  /* ---------------------------------------------------------------
     data import: CSV / XLSX with column mapping
     roles: [{ key:'x', label:'X (feature)', numeric:true }, ...]
     resolves { cols:{key:[...]}, names:{key:'col name'}, n, dropped, file }
     --------------------------------------------------------------- */
  MLS.importTable = function (opts) {
    const roles = opts.roles, maxRows = opts.maxRows || 5000;
    return new Promise(resolve => {
      const back = MLS.el('div', { class: 'modal-back' });
      const m = MLS.el('div', { class: 'modal', role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Load your data' });
      m.innerHTML = '<h3>Load your data (CSV or Excel)</h3>' +
        '<p class="note">The first row should hold the column names. Only numeric values are used; rows with missing or non-numeric values in the chosen columns are skipped. ' +
        (opts.note || '') + '</p>' +
        '<input type="file" accept=".csv,.txt,.xlsx,.xls" style="margin-top:12px">' +
        '<div class="imp-body" style="margin-top:10px"></div>' +
        '<div class="foot"><button type="button" class="btn-ghost imp-cancel">Cancel</button><button type="button" class="imp-ok" disabled>Use these columns</button></div>';
      back.appendChild(m); document.body.appendChild(back);
      const body = m.querySelector('.imp-body'), ok = m.querySelector('.imp-ok');
      let table = null, fileName = '';
      function close(result) { back.remove(); resolve(result || null); }
      m.querySelector('.imp-cancel').onclick = () => close(null);
      back.addEventListener('click', e => { if (e.target === back) close(null); });
      m.querySelector('input[type=file]').onchange = async ev => {
        const f = ev.target.files[0]; if (!f) return;
        fileName = f.name;
        body.innerHTML = '<p class="note">Reading ' + MLS.esc(f.name) + '...</p>';
        try {
          await MLS.needXLSX();
          const buf = await f.arrayBuffer();
          const wb = window.XLSX.read(buf, { type: 'array' });
          const sh = wb.Sheets[wb.SheetNames[0]];
          const aoa = window.XLSX.utils.sheet_to_json(sh, { header: 1, raw: true, blankrows: false });
          if (!aoa.length || aoa.length < 2) throw new Error('The file has no data rows.');
          const header = aoa[0].map((h, i) => (h === undefined || h === null || h === '' ? 'Column ' + (i + 1) : String(h)));
          const rows = aoa.slice(1);
          table = { header, rows };
          const num = v => (typeof v === 'number' ? v : (typeof v === 'string' && v.trim() !== '' && !isNaN(Number(v.replace(',', '.'))) ? Number(v.replace(',', '.')) : NaN));
          const numericCols = header.map((h, i) => rows.slice(0, 200).filter(r => Number.isFinite(num(r[i]))).length >= Math.min(3, rows.length));
          let html = '<p class="note"><b>' + rows.length + '</b> rows, <b>' + header.length + '</b> columns. Choose which column plays each role:</p>';
          roles.forEach((r, ri) => {
            if (r.multi) {
              html += '<label class="field">' + r.label + '</label><div class="var-list" data-multi="' + r.key + '">' +
                header.map((h, i) => (r.numeric !== false && !numericCols[i]) ? '' : '<label class="check"><input type="checkbox" value="' + i + '"' + (i === Math.min(ri, header.length - 1) ? ' checked' : '') + '> ' + MLS.esc(h) + '</label>').join('') + '</div>';
              return;
            }
            html += '<label class="field">' + r.label + '</label><select data-role="' + r.key + '">';
            header.forEach((h, i) => {
              if (r.numeric !== false && !numericCols[i]) return;
              html += '<option value="' + i + '"' + (i === Math.min(ri, header.length - 1) ? ' selected' : '') + '>' + MLS.esc(h) + '</option>';
            });
            html += '</select>';
          });
          html += '<div class="fig-table" style="margin-top:12px"><table class="tbl"><thead><tr>' + header.map(h => '<th>' + MLS.esc(h) + '</th>').join('') + '</tr></thead><tbody>' +
            rows.slice(0, 6).map(r => '<tr>' + header.map((_, i) => '<td>' + MLS.esc(r[i] === undefined ? '' : r[i]) + '</td>').join('') + '</tr>').join('') + '</tbody></table></div>';
          body.innerHTML = html;
          ok.disabled = false;
          ok.onclick = () => {
            const idx = {}; m.querySelectorAll('select[data-role]').forEach(s => { idx[s.dataset.role] = Number(s.value); });
            const midx = {}; m.querySelectorAll('[data-multi]').forEach(d => { midx[d.dataset.multi] = Array.from(d.querySelectorAll('input:checked')).map(c => Number(c.value)); });
            for (const r of roles) if (r.multi && !midx[r.key].length) { MLS.toast('Choose at least one column for ' + r.label + '.'); return; }
            for (const r of roles) if (r.multi && roles.some(q => !q.multi && midx[r.key].includes(idx[q.key]))) { MLS.toast('A column cannot be both Y and an X variable.'); return; }
            const cols = {}, names = {};
            roles.forEach(r => { if (r.multi) { cols[r.key] = midx[r.key].map(() => []); names[r.key] = midx[r.key].map(i => header[i]); } else { cols[r.key] = []; names[r.key] = header[idx[r.key]]; } });
            let dropped = 0, kept = 0;
            for (const row of table.rows) {
              const vals = roles.map(r => r.multi ? midx[r.key].map(i => num(row[i])) : num(row[idx[r.key]]));
              if (vals.some(v => Array.isArray(v) ? v.some(w => !Number.isFinite(w)) : !Number.isFinite(v))) { dropped++; continue; }
              if (kept >= maxRows) { dropped++; continue; }
              kept++;
              roles.forEach((r, k) => { if (r.multi) vals[k].forEach((w, j) => cols[r.key][j].push(w)); else cols[r.key].push(vals[k]); });
            }
            if (!kept) { MLS.toast('No complete numeric rows in those columns.'); return; }
            close({ cols, names, n: kept, dropped, file: fileName });
            return;
          };
        } catch (e) { body.innerHTML = '<div class="box-warn">Could not read the file: ' + MLS.esc(e.message) + '</div>'; ok.disabled = true; }
      };
    });
  };

  /* ---------------------------------------------------------------
     module state: remembered settings + share links
     --------------------------------------------------------------- */
  function b64e(obj) { const s = JSON.stringify(obj); return btoa(unescape(encodeURIComponent(s))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''); }
  function b64d(str) { str = str.replace(/-/g, '+').replace(/_/g, '/'); while (str.length % 4) str += '='; return JSON.parse(decodeURIComponent(escape(atob(str)))); }
  MLS.state = {
    cfg: null,
    /* cfg: { id, get:()=>obj, set:(obj)=>void } ; returns 'link' | 'saved' | 'default' */
    init(cfg) {
      this.cfg = cfg;
      const m = location.hash.match(/[#&]s=([A-Za-z0-9_-]+)/);
      if (m) {
        try { cfg.set(b64d(m[1])); this.fromLink = true; MLS.banner('This page opened with a <b>shared setup</b> (data, seed and settings chosen by your instructor).', 'info', 'mlsSharedBanner'); return 'link'; }
        catch (e) { MLS.toast('The shared link could not be read; using defaults.'); }
      }
      const saved = MLS.store.get('state:' + cfg.id, null);
      /* settings saved by an older build are dropped, so a new build always opens with its own defaults */
      if (saved && saved._v !== MLS.APP.stateVersion) { MLS.store.del('state:' + cfg.id); return 'default'; }
      if (saved) { try { delete saved._v; cfg.set(saved); return 'saved'; } catch (e) { /* fall through */ } }
      return 'default';
    },
    save: MLS.debounce(function () {
      const c = MLS.state.cfg; if (!c || MLS.state.resetting) return;
      try { MLS.store.set('state:' + c.id, Object.assign({ _v: MLS.APP.stateVersion }, c.get())); } catch (e) { /* ignore */ }
    }, 400),
    reset() {
      const c = this.cfg; if (!c) return;
      this.resetting = true;                              /* no save may write the old settings back */
      MLS.store.del('state:' + c.id); MLS.store.del('ch:' + c.id + ':open');
      if (MLS.inBundle) { MLS.go('modules/' + c.id + '.html'); return; }   /* the offline file: reload the page from the bundle */
      try { history.replaceState(null, '', location.pathname + location.search); } catch (e) { /* ignore */ }
      location.reload();
    },
    link() {
      const c = this.cfg; if (!c) return '';
      const st = c.get();
      const page = MLS.inBundle || !/^https?:$/.test(location.protocol) ? MLS.APP.siteUrl + 'modules/' + c.id + '.html' : location.origin + location.pathname;
      return page + '#s=' + b64e(st);
    },
    async share() {
      const url = this.link();
      let copied = false;
      try { await navigator.clipboard.writeText(url); copied = true; } catch (e) { /* clipboard blocked */ }
      const b = MLS.banner((copied ? '<b>Link copied.</b> ' : '<b>Copy this link:</b> ') +
        'Anyone who opens it sees this module with the same data, seed and settings.' +
        '<input type="text" readonly style="margin-top:8px" value="' + MLS.esc(url) + '">' +
        (MLS.state.cfg && MLS.state.cfg.uploaded && MLS.state.cfg.uploaded() ? '<div class="note">Uploaded files are not included in links; the link opens with the built-in dataset instead.</div>' : ''), 'info', 'mlsShareBanner');
      const inp = b.querySelector('input'); inp.focus(); inp.select();
    }
  };

  /* ---------------------------------------------------------------
     module top area (frozen while scrolling):
       row 1: back, title, tags, intro, share, reset
       row 2: tabs (Simulation | Lesson | Challenges | On real data)
     meta: { id, title, section, level }
     opts: { tabs:[{id,label}], onTab(id) }
     --------------------------------------------------------------- */
  MLS.moduleBar = function (meta, opts) {
    opts = opts || {};
    const host = document.getElementById('mlsModBar');
    if (!host) return;
    host.className = 'mod-bar';
    const lvlClass = 'lvl-' + String(meta.level).toLowerCase();
    host.innerHTML =
      '<a class="back proj-hide" ' + MLS.navAttrs('index.html') + '>&larr; All modules</a>' +
      '<h2 class="mod-title">' + meta.title + '</h2>' +
      '<span class="tag proj-hide ' + lvlClass + '" title="' + MLS.esc(meta.section) + '">' + meta.level + '</span>' +
      '<span class="spacer"></span>' +

      '<button type="button" class="btn-ghost btn-small proj-hide" id="mlsIntroBtn" title="Show or hide the short introduction">&#9432; Intro</button>' +
      '<button type="button" class="btn-ghost btn-small proj-hide" id="mlsShareBtn" title="Copy a link that opens this module with the current data, seed and settings">&#128279; Share</button>' +

      '<button type="button" class="btn-ghost btn-small proj-hide" id="mlsResetBtn" title="Forget my settings for this module and start fresh">&#8635; Reset</button>';
    MLS.registerHelp({ _mathlevel: { title: 'How much math to show', html: '<b>Intuition</b>: ideas and pictures only. <b>Formulas</b>: adds the key equations next to the ideas. <b>Deep dive</b>: adds derivations, conditions and references, for advanced and PhD students. The choice is remembered for all modules.' } });
    /* math level control lives at the top of the Lesson tab (it only affects the Lesson) */
    const mc = document.getElementById('mlsMathCtl');
    if (mc) {
      mc.className = 'math-ctl';
      mc.innerHTML = '<span class="ctl-label">Level of mathematics' + MLS.helpBtn('_mathlevel') + '</span>' +
        '<div class="seg" role="group" aria-label="How much math to show">' +
          '<button type="button" data-math-btn="0" title="Ideas and pictures only">Intuition</button>' +
          '<button type="button" data-math-btn="1" title="Also show the formulas">Formulas</button>' +
          '<button type="button" data-math-btn="2" title="Formulas plus derivations and references">Deep dive</button>' +
        '</div>';
      MLS.$$('[data-math-btn]', mc).forEach(b => b.onclick = () => MLS.setMath(Number(b.dataset.mathBtn)));
    }
    MLS.setMath(Number(MLS.store.get('mathLevel', 0)), false);
    document.getElementById('mlsShareBtn').onclick = () => MLS.state.share();
    document.getElementById('mlsResetBtn').onclick = () => { if (confirmReset()) MLS.state.reset(); };
    /* intro box (hide/show remembered per module) */
    const intro = document.getElementById('mlsIntro');
    const ib = document.getElementById('mlsIntroBtn');
    if (intro) {
      const title = (intro.querySelector('h3') || {}).textContent || 'the introduction';
      const show = MLS.el('button', { type: 'button', class: 'intro-show', title: 'Show the introduction again' }, '&#9656; Show the introduction: <b>' + MLS.esc(title) + '</b>');
      intro.after(show);
      const setHidden = h => { intro.hidden = h; show.hidden = !h; ib.setAttribute('aria-pressed', String(!h)); MLS.store.set('introHidden:' + meta.id, h); };
      const close = MLS.el('button', { type: 'button', class: 'btn-ghost btn-small intro-close', title: 'Hide the introduction' }, 'Hide &#9652;');
      close.onclick = () => setHidden(true);
      intro.prepend(close);
      show.onclick = () => setHidden(false);
      ib.onclick = () => { setHidden(!intro.hidden); if (!intro.hidden) window.scrollTo({ top: 0, behavior: 'smooth' }); };
      setHidden(MLS.store.get('introHidden:' + meta.id, false));
    } else ib.hidden = true;
    /* tabs in the frozen area */
    const tabsHost = document.getElementById('mlsTabs');
    if (tabsHost && opts.tabs) {
      tabsHost.className = 'tabs top-tabs';
      tabsHost.setAttribute('role', 'tablist');
      tabsHost.innerHTML = opts.tabs.map(t => '<button type="button" role="tab" data-tab="' + t.id + '">' + t.label + '</button>').join('');
      const pick = id => {
        MLS.$$('button[data-tab]', tabsHost).forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === id)));
        MLS.$$('[data-panel]').forEach(p => { p.hidden = p.dataset.panel !== id; });
        MLS.renderMath();
        MLS.store.set('tab:' + meta.id, id);
        document.dispatchEvent(new CustomEvent('mls:tab', { detail: id }));
        if (opts.onTab) opts.onTab(id);
      };
      MLS.$$('button[data-tab]', tabsHost).forEach(b => b.onclick = () => { pick(b.dataset.tab); const top = document.getElementById('mlsTop'); if (top && top.getBoundingClientRect().top <= 1) window.scrollTo({ top: document.getElementById('mlsTop').offsetTop, behavior: 'auto' }); });
      MLS.pickTab = pick;
      const saved = MLS.store.get('tab:' + meta.id, opts.tabs[0].id);
      pick(opts.tabs.some(t => t.id === saved) ? saved : opts.tabs[0].id);
    }
    /* keep the frozen-area height available to CSS (for the side panel) */
    const top = document.getElementById('mlsTop');
    if (top) {
      const setH = () => document.documentElement.style.setProperty('--top-h', top.offsetHeight + 'px');
      setH(); new ResizeObserver(setH).observe(top);
    }
    /* side panel scrolls on its own: show a 'more below' hint while part of it is hidden */
    MLS.$$('.side').forEach(side => {
      const hint = MLS.el('div', { class: 'side-more', role: 'button', tabindex: '0' }, 'More below: scroll this panel &#9662;');
      side.appendChild(hint);
      const upd = () => side.classList.toggle('has-more', side.scrollHeight - side.clientHeight - side.scrollTop > 24);
      hint.onclick = () => side.scrollBy({ top: side.clientHeight * 0.8, behavior: 'smooth' });
      side.addEventListener('scroll', upd, { passive: true }); window.addEventListener('resize', upd);
      new ResizeObserver(upd).observe(side); MLS.$$(':scope > *', side).forEach(ch => new ResizeObserver(upd).observe(ch));
      setTimeout(upd, 300);
    });
    document.title = meta.title + ' | 4MLSim b' + MLS.APP.buildNo;
    MLS.store.set('lastModule', meta.id);
  };
  function confirmReset() {
    /* no blocking dialogs: use a two-click confirm on the button */
    const b = document.getElementById('mlsResetBtn');
    if (b.dataset.armed) return true;
    b.dataset.armed = '1'; const old = b.innerHTML; b.innerHTML = 'Click again to reset';
    setTimeout(() => { delete b.dataset.armed; b.innerHTML = old; }, 3000);
    return false;
  }

  /* ---------------------------------------------------------------
     stat card with optional "?" help
     --------------------------------------------------------------- */
  MLS.statCard = function (label, value, sub, cls, help) {
    return '<div class="stat' + (cls ? ' ' + cls : '') + '"><div class="label">' + label + (help ? MLS.helpBtn(help) : '') + '</div><div class="value">' + value + '</div>' + (sub ? '<div class="sub">' + sub + '</div>' : '') + '</div>';
  };

  /* ---------------------------------------------------------------
     simple tabs inside a card
     --------------------------------------------------------------- */
  MLS.tabs = function (root, onChange) {
    const btns = MLS.$$('.tabs button[data-tab]', root);
    function pick(name) {
      btns.forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
      MLS.$$('[data-panel]', root).forEach(p => { p.hidden = p.dataset.panel !== name; });
      MLS.renderMath(root);
      if (onChange) onChange(name);
    }
    btns.forEach(b => { b.setAttribute('role', 'tab'); b.onclick = () => pick(b.dataset.tab); });
    if (btns.length) pick(btns[0].dataset.tab);
    return { pick };
  };

  /* ---------------------------------------------------------------
     challenges: "predict first" questions and checked tasks
     items: { type:'predict', q, options:[], answer:i, explain, tryIt?:fn }
            { type:'task', q, hint?, check:()=>bool, setup?:fn }
     --------------------------------------------------------------- */
  MLS.challenges = function (host, id, items) {
    const key = 'ch:' + id;
    const done = MLS.store.get(key, {});
    host.innerHTML = '';
    const score = MLS.el('div', { class: 'ch-score' });
    const list = MLS.el('div', { class: 'ch-list' });
    host.appendChild(score); host.appendChild(list);
    const L = 'ABCDEFGH';
    const tasks = [];
    function updScore() {
      const n = items.length, k = items.filter((_, i) => done[i]).length;
      score.textContent = k + ' of ' + n + ' completed' + (k === n ? '. Well done!' : '.');
    }
    items.forEach((it, i) => {
      const box = MLS.el('div', { class: 'ch' });
      const status = MLS.el('span', { class: 'ch-status' + (done[i] ? ' done' : '') }, done[i] ? '&#10003; Done' : (it.type === 'predict' ? 'Answer' : 'Open'));
      const head = MLS.el('div', { class: 'ch-head' });
      head.innerHTML = '<span class="ch-badge ' + it.type + '">' + (it.type === 'predict' ? 'Predict first' : 'Challenge') + '</span><div class="ch-q">' + it.q + '</div>';
      head.appendChild(status);
      box.appendChild(head);
      if (it.type === 'predict') {
        const opts = MLS.el('div', { class: 'ch-opts' });
        const fb = MLS.el('div', { class: 'ch-feedback' });
        it.options.forEach((o, k) => {
          const b = MLS.el('button', { type: 'button', class: 'ch-opt' }, '<span class="ltr">' + L[k] + '</span>' + o);
          b.onclick = () => {
            if (k === it.answer) {
              b.classList.add('right');
              fb.innerHTML = '<b>Correct.</b> ' + (it.explain || '');
              done[i] = true; MLS.store.set(key, done); status.className = 'ch-status done'; status.innerHTML = '&#10003; Done'; updScore();
              opts.querySelectorAll('button').forEach(x => (x.disabled = true));
            } else {
              b.classList.add('wrong'); b.disabled = true;
              fb.innerHTML = '<b>Not quite.</b> Try another answer' + (it.tryIt ? ', or test it in the simulation first.' : '.');
            }
            MLS.renderMath(fb);
          };
          opts.appendChild(b);
        });
        box.appendChild(opts); box.appendChild(fb);
        if (done[i]) { fb.innerHTML = '<b>Answered.</b> ' + (it.explain || ''); opts.querySelectorAll('button').forEach((x, k) => { x.disabled = true; if (k === it.answer) x.classList.add('right'); }); }
        if (it.tryIt) { const a = MLS.el('div', { class: 'ch-actions' }); const tb = MLS.el('button', { type: 'button', class: 'btn-secondary btn-small' }, '&#9654; Test it in the simulation'); tb.onclick = it.tryIt; a.appendChild(tb); box.appendChild(a); }
      } else {
        if (it.hint) box.appendChild(MLS.el('div', { class: 'note' }, it.hint));
        if (it.setup) { const a = MLS.el('div', { class: 'ch-actions' }); const sb = MLS.el('button', { type: 'button', class: 'btn-secondary btn-small' }, 'Set up this challenge'); sb.onclick = it.setup; a.appendChild(sb); box.appendChild(a); }
        tasks.push({ i, it, status });
      }
      list.appendChild(box);
    });
    MLS.renderMath(host);
    updScore();
    return {
      check() {
        let changed = false;
        tasks.forEach(t => {
          if (done[t.i]) return;
          let ok = false; try { ok = !!t.it.check(); } catch (e) { ok = false; }
          if (ok) { done[t.i] = true; changed = true; t.status.className = 'ch-status done'; t.status.innerHTML = '&#10003; Done'; MLS.toast('Challenge completed: nice work!'); }
        });
        if (changed) { MLS.store.set(key, done); updScore(); }
      },
      resetProgress() { MLS.store.del(key); }
    };
  };

  /* ---------------------------------------------------------------
     pseudocode panel
     lines: array of strings; use **word** for keywords and // for comments
     --------------------------------------------------------------- */
  MLS.pseudo = function (host, lines) {
    const ol = MLS.el('ol', { class: 'pseudo', 'aria-label': 'Pseudocode' });
    lines.forEach(l => {
      let h = MLS.esc(l).replace(/\*\*(.+?)\*\*/g, '<span class="kw">$1</span>').replace(/(\/\/.*)$/, '<span class="cm">$1</span>');
      ol.appendChild(MLS.el('li', null, h));
    });
    host.innerHTML = ''; host.appendChild(ol);
    return {
      hl(idx) { const want = [].concat(idx); Array.from(ol.children).forEach((li, k) => li.classList.toggle('hl', want.includes(k + 1))); },
      clear() { Array.from(ol.children).forEach(li => li.classList.remove('hl')); }
    };
  };

  /* ---------------------------------------------------------------
     stepper: Reset / Step / Play-Pause / Run to end, with speed
     opts: { step:()=>bool (false when finished), reset:()=>void, runMax:number, label }
     --------------------------------------------------------------- */
  MLS.stepper = function (host, opts) {
    host.innerHTML =
      '<div class="stepper">' +
        '<button type="button" class="btn-ghost" data-a="reset" title="Back to the start">&#8634; Reset</button>' +
        '<button type="button" class="btn-secondary" data-a="step" title="One step">&#10142; Step</button>' +
        '<button type="button" data-a="play" title="Run continuously">&#9654; Play</button>' +
        '<button type="button" class="btn-ghost" data-a="end" title="Run to the end at once">&#9197; End</button>' +
      '</div>' +
      '<div class="speed-row"><span>Speed</span><input type="range" min="1" max="60" value="' + (opts.speed || 12) + '" aria-label="Steps per second"><span class="val sp"></span></div>';
    const [bReset, bStep, bPlay, bEnd] = ['reset', 'step', 'play', 'end'].map(a => host.querySelector('[data-a="' + a + '"]'));
    const sp = host.querySelector('input[type=range]'), spv = host.querySelector('.sp');
    let timer = null, acc = 0, last = 0, running = false;
    function showSp() { spv.textContent = sp.value + '/s'; }
    showSp(); sp.oninput = showSp;
    function stop() { running = false; if (timer) cancelAnimationFrame(timer); timer = null; bPlay.innerHTML = '&#9654; Play'; }
    function frame(t) {
      if (!running) return;
      const dt = last ? (t - last) / 1000 : 0; last = t;
      acc += dt * Number(sp.value);
      let n = Math.floor(acc); acc -= n;
      if (n > 0) {
        n = Math.min(n, 50);
        for (let k = 0; k < n; k++) { if (opts.step() === false) { stop(); if (opts.after) opts.after(); return; } }
        if (opts.after) opts.after();
      }
      timer = requestAnimationFrame(frame);
    }
    bPlay.onclick = () => {
      if (running) { stop(); return; }
      running = true; last = 0; acc = 1; bPlay.innerHTML = '&#10074;&#10074; Pause';
      timer = requestAnimationFrame(frame);
    };
    bStep.onclick = () => { stop(); opts.step(); if (opts.after) opts.after(); };
    bReset.onclick = () => { stop(); opts.reset(); if (opts.after) opts.after(); };
    bEnd.onclick = () => {
      stop();
      const max = opts.runMax || 1000;
      for (let k = 0; k < max; k++) if (opts.step() === false) break;
      if (opts.after) opts.after();
    };
    return { stop, isRunning: () => running };
  };

  /* ---------------------------------------------------------------
     page boot: header, footer, math, version check
     --------------------------------------------------------------- */
  document.addEventListener('DOMContentLoaded', () => {
    if (!document.body.dataset.math) document.body.dataset.math = String(MLS.store.get('mathLevel', 0));
    MLS.renderHeader();
    MLS.renderFooter();
    MLS.renderMath();
    setTimeout(MLS.checkVersion, 1500);
  });
})();
