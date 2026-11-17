/* Interactive threshold explorer. All model outputs are precomputed by src/pipeline.py
   (site_data/results.json); nothing here trains a model. */
(function () {
  const $ = (id) => document.getElementById(id);
  const res = window.RESULTS;
  if (!res) { $('loading').textContent = 'Model outputs are missing. Rebuild the page with python src/build_site.py.'; return; }
  const M = res.models, T = res.test, R = res.meta.ratios, y = T.y, N = y.length;
  const NPOS = y.reduce((a, b) => a + b, 0), NNEG = N - NPOS;
  const css = (v) => getComputedStyle($('app')).getPropertyValue(v).trim();
  const C = { ink: css('--ink'), alarm: css('--alarm'), calm: css('--calm'), grey: css('--grey') };
  const TXT = css('--muted'), GRID = css('--line'), BG = '#0E1C30';
  const fmtR = (r) => (Number.isInteger(r) ? String(r) : r.toFixed(1));
  const pct = (x, d = 1) => (100 * x).toFixed(d) + '%';
  const f3 = (x) => x.toFixed(3);
  const labelOf = (k) => M[k].label + (M[k].feat === 'eng' ? ' with physics features' : ' on raw features');
  const refIdx = R.indexOf(res.meta.r_ref);

  const st = { model: res.meta.winner, scores: 'raw', ri: refIdx, r: res.meta.r_ref, fa: 500, t: 0.5, follow: true, tab: 'miss' };

  /* ---------- static tables ---------- */
  function buildFacts() {
    const e = res.eda;
    $('dataFacts').innerHTML = [[e.n.toLocaleString(), 'operating snapshots'], [String(e.cols), 'columns, none missing'],
      [e.failures + ' (' + pct(e.rate, 2) + ')', 'machine failures'], [e.ntrain.toLocaleString() + ' / ' + e.ntest.toLocaleString(), 'train / test rows']]
      .map(([a, b]) => `<div><b>${a}</b><span>${b}</span></div>`).join('');
  }
  function buildModelTable() {
    const b = res.meta.baseline, rows = [];
    rows.push(`<tr><td class="l">Majority class (baseline)</td><td class="l">none</td><td>${f3(res.eda.ptrain / res.eda.ntrain)}</td><td>${f3(b.test_ap)}</td><td>0.0%</td><td>${(res.meta.r_ref * res.eda.ptest).toFixed(0)}</td><td>n/a</td></tr>`);
    const keys = Object.keys(M);
    for (const k of keys) {
      const m = M[k], b5 = m.fpr_budget[2];
      rows.push(`<tr class="${k === res.meta.winner ? 'win' : ''}"><td class="l">${m.label}${m.model === 'logreg' ? ' (baseline 2)' : ''}</td><td class="l">${m.feat === 'eng' ? 'raw + physics' : 'raw'}</td>
        <td>${f3(m.cv_ap_mean)} ± ${f3(m.cv_ap_std)}</td><td>${f3(m.test.ap)}</td><td>${pct(b5.recall)}</td><td>${m.ref.cost.toFixed(0)}</td>
        <td>${m.test.brier_raw.toFixed(4)} → ${m.test.brier_cal.toFixed(4)}</td></tr>`);
    }
    $('modelTable').innerHTML = `<thead><tr><th class="l">Model</th><th class="l">Features</th><th>CV PR-AUC (mean ± sd)</th><th>Test PR-AUC</th><th>Recall at 5% false-alarm budget</th><th>Test cost at 10:1</th><th>Brier, raw → calibrated</th></tr></thead><tbody>${rows.join('')}</tbody>`;
    const w = M[res.meta.winner], bt = res.meta.boot;
    $('modelNote').textContent = `Shaded row: the model chosen by cross-validated PR-AUC (${labelOf(res.meta.winner).toLowerCase()}). Its test PR-AUC is ${f3(w.test.ap)} (95% bootstrap interval ${bt.ap[0].toFixed(2)} to ${bt.ap[1].toFixed(2)}). Cost = 10 × missed failures + false alarms on the 2,000 test machines; never alarming costs ${10 * res.eda.ptest} and always alarming costs ${NNEG}. Thresholds for budget and cost columns were chosen on training folds.`;
  }
  function buildBudgetTable() {
    const w = M[res.meta.winner];
    $('budgetTable').innerHTML = `<thead><tr><th>Healthy snapshots allowed to alarm</th><th>Threshold (training folds)</th><th>Failures caught on test</th><th>Recall</th><th>Actual test false-alarm rate</th></tr></thead><tbody>` +
      w.fpr_budget.map((b) => `<tr><td>${pct(b.budget, 0)}</td><td>${b.thr.toFixed(3)}</td><td>${b.tp} of ${NPOS}</td><td>${pct(b.recall)}</td><td>${pct(b.test_fpr)}</td></tr>`).join('') + '</tbody>';
  }
  function buildMissTable() {
    const rows = res.errors.fn.map((r) => `<tr><td>${r.UDI}</td><td>${r.Type}</td><td>${r['Rotational speed']}</td><td>${r.Torque}</td><td>${r['Tool wear']}</td><td>${(r['Process temperature'] - r['Air temperature']).toFixed(1)}</td><td>${r.p.toFixed(3)}</td><td class="l">${r.ftype}</td></tr>`);
    $('missTable').innerHTML = `<thead><tr><th>Machine row</th><th>Type</th><th>Speed (rpm)</th><th>Torque (Nm)</th><th>Tool wear (min)</th><th>Temp gap (K)</th><th>Model score</th><th class="l">Failure flag</th></tr></thead><tbody>${rows.join('')}</tbody>`;
  }
  buildFacts(); buildModelTable(); buildBudgetTable(); buildMissTable();

  /* ---------- interaction ---------- */
  const sel = $('model');
  for (const k of Object.keys(M)) sel.add(new Option(labelOf(k) + (k === res.meta.winner ? ' (chosen)' : ''), k));
  sel.value = st.model; $('ratio').max = R.length - 1; $('ratio').value = st.ri;

  const probs = () => T[st.scores][st.model];
  const chosenT = () => (st.scores === 'raw' ? M[st.model].thr_raw : M[st.model].thr_cal)[st.ri];
  const lower = (arr, v) => { let a = 0, b = arr.length; while (a < b) { const m = (a + b) >> 1; if (arr[m] < v) a = m + 1; else b = m; } return a; };

  let sp, sn, jit = [];            // sorted scores of failures / healthy, plus fixed jitter for the strip plot
  let seed = 7; for (let i = 0; i < N; i++) { seed = (seed * 16807) % 2147483647; jit.push(seed / 2147483647); }

  function prep() {
    const p = probs(); const pos = [], neg = [];
    for (let i = 0; i < N; i++) (y[i] ? pos : neg).push(p[i]);
    sp = pos.sort((a, b) => a - b); sn = neg.sort((a, b) => a - b);
  }
  const fnAt = (t) => lower(sp, t);                 // failures scoring below t (missed)
  const fpAt = (t) => sn.length - lower(sn, t);     // healthy scoring at or above t (false alarms)

  function setT(t, fromUser) {
    st.t = Math.round(Math.min(1, Math.max(0, t)) * 1000) / 1000;
    $('thr').value = st.t;
    if (fromUser) { st.follow = false; $('follow').checked = false; }
  }

  function render() {
    prep();
    const r = st.r;
    if (st.follow) setT(chosenT(), false);
    $('ratioOut').textContent = fmtR(r);
    $('tOut').textContent = st.t.toFixed(3);
    const fn = fnAt(st.t), fp = fpAt(st.t), tp = NPOS - fn, tn = NNEG - fp;
    $('tp').textContent = tp; $('fn').textContent = fn; $('fp').textContent = fp; $('tn').textContent = tn.toLocaleString();
    $('recall').textContent = pct(tp / NPOS); $('precision').textContent = tp + fp ? pct(tp / (tp + fp)) : 'no alarms';
    $('far').textContent = pct(fp / NNEG); const cost = r * fn + fp;
    const never = r * NPOS, saved = 1 - cost / never;
    const usd = (n) => '$' + Math.round(n).toLocaleString();
    $('cost').textContent = usd(fn * r * st.fa + fp * st.fa);
    $('costNote').textContent = `${fn} missed × ${usd(r * st.fa)} + ${fp} false alarms × ${usd(st.fa)}. Never alarming would cost ${usd(never * st.fa)}; alarming on every machine ${usd(NNEG * st.fa)}. At this threshold you ${saved >= 0 ? 'save ' + pct(saved, 0) + ' of' : 'spend ' + pct(-saved, 0) + ' more than'} the never-alarm cost.`;
    drawStrip(); drawCost(r); drawThr(); drawErrors();
    window.dispatchEvent(new CustomEvent('tm:render'));
  }

  /* strip plot: one dot per test machine */
  function drawStrip() {
    const cv = $('strip'), dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = 240;
    cv.width = w * dpr; cv.height = h * dpr; const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const L = 14, Rr = w - 14, X = (p) => L + p * (Rr - L), p = probs(), t = st.t;
    g.clearRect(0, 0, w, h); g.fillStyle = 'rgba(255,106,43,0.13)'; g.fillRect(X(t), 0, Rr - X(t) + 14, h - 26);
    const yF = (j) => 30 + j * 56, yH = (j) => 108 + j * 92;
    g.fillStyle = '#5FA9B5';
    for (let i = 0; i < N; i++) if (!y[i] && p[i] < t) { g.beginPath(); g.arc(X(p[i]), yH(jit[i]), 1.7, 0, 6.3); g.fill(); }
    g.fillStyle = C.ink;
    for (let i = 0; i < N; i++) if (!y[i] && p[i] >= t) { g.beginPath(); g.arc(X(p[i]), yH(jit[i]), 2.6, 0, 6.3); g.fill(); }
    for (let i = 0; i < N; i++) if (y[i]) {
      g.beginPath(); g.arc(X(p[i]), yF(jit[i]), 4.2, 0, 6.3);
      if (p[i] >= t) { g.fillStyle = C.alarm; g.fill(); } else { g.fillStyle = BG; g.fill(); g.lineWidth = 2; g.strokeStyle = C.alarm; g.stroke(); }
    }
    g.strokeStyle = C.ink; g.lineWidth = 2; g.beginPath(); g.moveTo(X(t), 4); g.lineTo(X(t), h - 26); g.stroke();
    g.fillStyle = TXT; g.font = '13px "Instrument Sans", system-ui, sans-serif';
    g.textAlign = 'left'; g.fillText('Failed machines (' + NPOS + ')', L + 2, 14); g.fillText('Healthy machines (' + NNEG.toLocaleString() + ')', L + 2, 100);
    g.textAlign = 'center'; for (let v = 0; v <= 1.001; v += 0.25) g.fillText(v.toFixed(2), Math.min(Math.max(X(v), L + 8), Rr - 8), h - 8);
  }

  /* generic small svg line chart helpers */
  const svgOpen = (w, h) => `<svg viewBox="0 0 ${w} ${h}" role="img" xmlns="http://www.w3.org/2000/svg" font-family="Instrument Sans, system-ui, sans-serif" font-size="11">`;
  function drawCost(r) {
    const W = 420, H = 250, l = 42, rt = 10, tp = 12, bt = 36, pw = W - l - rt, ph = H - tp - bt, ymax = 1.5;
    const X = (v) => l + v * pw, Y = (v) => tp + ph - Math.min(v, ymax) / ymax * ph, never = r * NPOS;
    let best = 1e9, bestT = [], pts = [];
    for (let k = 0; k <= 200; k++) { const t = k / 200, c = (r * fnAt(t) + fpAt(t)); if (c < best) { best = c; bestT = [t]; } else if (c === best) bestT.push(t); pts.push(`${X(t).toFixed(1)},${Y(c / never).toFixed(1)}`); }
    const hind = bestT[bestT.length >> 1], cur = (r * fnAt(st.t) + fpAt(st.t)) / never;
    let s = svgOpen(W, H) + `<title>Cost against threshold</title>`;
    for (const v of [0, 0.5, 1, 1.5]) s += `<line x1="${l}" x2="${W - rt}" y1="${Y(v)}" y2="${Y(v)}" stroke="${GRID}"/><text x="${l - 6}" y="${Y(v) + 4}" text-anchor="end" fill="${TXT}">${v}</text>`;
    for (const v of [0, 0.25, 0.5, 0.75, 1]) s += `<text x="${X(v)}" y="${H - 16}" text-anchor="middle" fill="${TXT}">${v}</text>`;
    s += `<line x1="${l}" x2="${W - rt}" y1="${Y(1)}" y2="${Y(1)}" stroke="${C.grey}" stroke-dasharray="4 3"/><text x="${W - rt - 2}" y="${Y(1) - 5}" text-anchor="end" fill="${TXT}">never alarm = 1</text>`;
    s += `<polyline points="${pts.join(' ')}" fill="none" stroke="${C.calm}" stroke-width="2.2"/>`;
    s += `<circle cx="${X(hind)}" cy="${Y(best / never)}" r="7" fill="none" stroke="${C.alarm}" stroke-width="2"/>`;
    s += `<line x1="${X(st.t)}" x2="${X(st.t)}" y1="${tp}" y2="${tp + ph}" stroke="${C.ink}" stroke-width="1.2" stroke-dasharray="2 3"/><circle cx="${X(st.t)}" cy="${Y(cur)}" r="4.5" fill="${C.ink}"/>`;
    s += `<text x="${l + pw / 2}" y="${H - 2}" text-anchor="middle" fill="${TXT}">alarm threshold</text><text transform="rotate(-90)" x="${-(tp + ph / 2)}" y="11" text-anchor="middle" fill="${TXT}">cost ÷ never-alarm cost</text></svg>`;
    $('costChart').innerHTML = s;
  }
  function drawThr() {
    const W = 420, H = 250, l = 42, rt = 10, tp = 12, bt = 36, pw = W - l - rt, ph = H - tp - bt, lr = Math.log(R[R.length - 1]);
    const X = (r) => l + Math.log(Math.max(1, r)) / lr * pw, Y = (v) => tp + ph - v * ph;
    const arr = st.scores === 'raw' ? M[st.model].thr_raw : M[st.model].thr_cal;
    let s = svgOpen(W, H) + `<title>Best threshold against cost ratio</title>`;
    for (const v of [0, 0.25, 0.5, 0.75, 1]) s += `<line x1="${l}" x2="${W - rt}" y1="${Y(v)}" y2="${Y(v)}" stroke="${GRID}"/><text x="${l - 6}" y="${Y(v) + 4}" text-anchor="end" fill="${TXT}">${v}</text>`;
    for (const v of [1, 2, 5, 10, 20, 50, 100, 200]) s += `<text x="${X(v)}" y="${H - 16}" text-anchor="middle" fill="${TXT}">${v}</text>`;
    if (st.scores === 'cal') {
      s += `<polyline points="${R.map((r) => `${X(r).toFixed(1)},${Y(1 / (1 + r)).toFixed(1)}`).join(' ')}" fill="none" stroke="${C.grey}" stroke-dasharray="5 4" stroke-width="1.6"/><text x="${X(30)}" y="${Y(0.2)}" fill="${TXT}">rule 1 ÷ (1 + ratio)</text>`;
    }
    s += `<polyline points="${R.map((r, i) => `${X(r).toFixed(1)},${Y(arr[i]).toFixed(1)}`).join(' ')}" fill="none" stroke="${C.calm}" stroke-width="2.2"/>`;
    s += `<circle cx="${X(st.r)}" cy="${Y(arr[st.ri])}" r="5" fill="${C.alarm}"/>`;
    s += `<text x="${l + pw / 2}" y="${H - 2}" text-anchor="middle" fill="${TXT}">cost of a missed failure ÷ cost of a false alarm</text><text transform="rotate(-90)" x="${-(tp + ph / 2)}" y="11" text-anchor="middle" fill="${TXT}">best threshold</text></svg>`;
    $('thrChart').innerHTML = s;
  }
  function drawErrors() {
    const p = probs(), rows = T.rows, miss = st.tab === 'miss', out = [];
    for (let i = 0; i < N; i++) if (miss ? (y[i] && p[i] < st.t) : (!y[i] && p[i] >= st.t)) out.push(i);
    out.sort((a, b) => miss ? p[a] - p[b] : p[b] - p[a]);
    const body = out.slice(0, 10).map((i) => { const r = rows[i]; return `<tr><td>${T.udi[i]}</td><td>${'LMH'[r[5]]}</td><td>${r[2]}</td><td>${r[3]}</td><td>${r[4]}</td><td>${(r[1] - r[0]).toFixed(1)}</td><td>${p[i].toFixed(3)}</td><td class="l">${r[6]}</td></tr>`; }).join('');
    $('errTable').innerHTML = out.length
      ? `<table class="data"><thead><tr><th>Machine row</th><th>Type</th><th>Speed (rpm)</th><th>Torque (Nm)</th><th>Tool wear (min)</th><th>Temp gap (K)</th><th>Score</th><th class="l">Failure flag</th></tr></thead><tbody>${body}</tbody></table><p class="small">${out.length > 10 ? `Showing 10 of ${out.length}. ` : ''}${miss ? 'Lowest scores first: the failures the model was most sure were fine.' : 'Highest scores first: healthy machines the model was most sure were failing.'}</p>`
      : `<p class="note">${miss ? 'No missed failures at this threshold.' : 'No false alarms at this threshold.'}</p>`;
  }

  /* events */
  $('model').addEventListener('change', (e) => { st.model = e.target.value; render(); });
  document.querySelectorAll('input[name=scores]').forEach((el) => el.addEventListener('change', (e) => { st.scores = e.target.value; render(); }));
  const nearest = (r) => R.reduce((b, v, i) => (Math.abs(v - r) < Math.abs(R[b] - r) ? i : b), 0);
  const setRatio = (r, keepMoney) => { st.r = Math.min(200, Math.max(1, r)); st.ri = nearest(st.r); $('ratio').value = st.ri; if (!keepMoney) $('cMiss').value = Math.round(st.r * st.fa); };
  $('ratio').addEventListener('input', (e) => { setRatio(R[+e.target.value]); render(); });
  const money = () => { const m = +$('cMiss').value, f = +$('cFalse').value; if (m > 0 && f > 0) { st.fa = f; setRatio(m / f, true); render(); } };
  $('cMiss').addEventListener('input', money); $('cFalse').addEventListener('input', money);
  document.querySelectorAll('[data-r]').forEach((b) => b.addEventListener('click', () => { st.follow = true; $('follow').checked = true; setRatio(+b.dataset.r); render(); }));
  let raf; const animate = (from, to, ms, fn) => { cancelAnimationFrame(raf); const t0 = performance.now(); const step = (now) => { const k = Math.min(1, (now - t0) / ms); fn(from + (to - from) * k); if (k < 1) raf = requestAnimationFrame(step); }; raf = requestAnimationFrame(step); };
  $('playT').addEventListener('click', () => { st.follow = false; $('follow').checked = false; animate(0, 1, 6000, (v) => { setT(v, false); render(); }); });
  $('playR').addEventListener('click', () => { st.follow = true; $('follow').checked = true; animate(0, R.length - 1, 8000, (v) => { const i = Math.round(v); if (i !== st.ri || v === 0) { setRatio(R[i]); render(); } }); });
  $('thr').addEventListener('input', (e) => { setT(+e.target.value, true); render(); });
  $('follow').addEventListener('change', (e) => { st.follow = e.target.checked; render(); });
  $('tabMiss').addEventListener('click', () => { st.tab = 'miss'; $('tabMiss').setAttribute('aria-selected', 'true'); $('tabFalse').setAttribute('aria-selected', 'false'); drawErrors(); });
  $('tabFalse').addEventListener('click', () => { st.tab = 'false'; $('tabFalse').setAttribute('aria-selected', 'true'); $('tabMiss').setAttribute('aria-selected', 'false'); drawErrors(); });
  const fromPointer = (ev) => { const b = $('strip').getBoundingClientRect(); setT((ev.clientX - b.left - 14) / (b.width - 28), true); render(); };
  let drag = false;
  $('strip').addEventListener('pointerdown', (ev) => { drag = true; $('strip').setPointerCapture(ev.pointerId); fromPointer(ev); });
  $('strip').addEventListener('pointermove', (ev) => { if (drag) fromPointer(ev); });
  $('strip').addEventListener('pointerup', () => { drag = false; });
  window.addEventListener('resize', drawStrip);

  $('loading').hidden = true; $('app').hidden = false; setRatio(st.r); render();
  window.TM = { st, res, M, T, R, y, N, NPOS, NNEG, css, probs, labelOf, chosenT };
})();
