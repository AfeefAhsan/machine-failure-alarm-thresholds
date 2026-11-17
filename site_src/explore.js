/* Lab: build-a-machine (tree + risk map) and model comparison. Reads shared state from app.js (window.TM). */
(function () {
  const TM = window.TM, tree = window.TREE; if (!TM || !tree) return;
  const { st, res, M, T, R, y, N } = TM, $ = (id) => document.getElementById(id), css = TM.css;
  const TXT = css('--muted'), GRID = css('--line'), INK = css('--ink'), ALARM = css('--alarm'), CALM = css('--calm');
  $('treeAp').textContent = tree.test_ap.toFixed(2);

  /* ---------- the tree ---------- */
  const NAMES = ['Quality type', 'Air temperature (K)', 'Process temperature (K)', 'Speed (rpm)', 'Torque (Nm)', 'Tool wear (min)', 'Power (W)', 'Temperature gap (K)', 'Wear × torque'];
  const vec = (m) => [m.type, m.air, m.proc, m.rpm, m.torque, m.wear, m.torque * m.rpm * 2 * Math.PI / 60, m.proc - m.air, m.wear * m.torque];
  function evalTree(m) {
    const x = vec(m), path = []; let n = 0;
    while (tree.left[n] !== -1) {
      const f = tree.feat[n], go = x[f] <= tree.thr[n]; path.push({ f, thr: tree.thr[n], le: go, v: x[f] });
      n = go ? tree.left[n] : tree.right[n];
    }
    return { p: tree.p[n], n: tree.n[n], path };
  }

  /* ---------- machine state + sliders ---------- */
  const DEF = { type: 1, air: 300, proc: 310, rpm: 1500, torque: 40, wear: 100 };
  const m = { ...DEF };
  const SL = [['air', 'Air temperature (K)', 295, 305, 0.1], ['proc', 'Process temperature (K)', 305, 314, 0.1], ['rpm', 'Speed (rpm)', 1100, 2900, 10], ['torque', 'Torque (Nm)', 3, 77, 0.1], ['wear', 'Tool wear (min)', 0, 255, 1]];
  $('sliders').innerHTML = `<div class="s"><label for="s_type"><span>Quality type</span></label><select id="s_type"><option value="0">L (low)</option><option value="1" selected>M (medium)</option><option value="2">H (high)</option></select></div>` +
    SL.map(([k, lab, lo, hi, step]) => `<div class="s"><label for="s_${k}"><span>${lab}</span><output id="o_${k}"></output></label><input type="range" id="s_${k}" min="${lo}" max="${hi}" step="${step}"></div>`).join('');
  const syncUI = () => { $('s_type').value = m.type; for (const [k, , , , step] of SL) { $('s_' + k).value = m[k]; $('o_' + k).textContent = step < 1 ? (+m[k]).toFixed(1) : Math.round(m[k]); } };
  SL.forEach(([k]) => $('s_' + k).addEventListener('input', (e) => { m[k] = +e.target.value; $('loaded').textContent = ''; update(); }));
  $('s_type').addEventListener('change', (e) => { m.type = +e.target.value; update(); });

  const rows = T.rows, fails = [], oks = [];
  for (let i = 0; i < N; i++) (y[i] ? fails : oks).push(i);
  const load = (i) => { const r = rows[i]; Object.assign(m, { air: r[0], proc: r[1], rpm: r[2], torque: r[3], wear: r[4], type: r[5] }); $('loaded').textContent = `Loaded test machine row ${T.udi[i]} (${y[i] ? 'a real failure, flag: ' + r[6] : 'a healthy machine'}). Note the tree was trained on other rows.`; update(); };
  const pick = (a) => a[Math.floor(Math.random() * a.length)];
  $('bFail').addEventListener('click', () => load(pick(fails))); $('bOk').addEventListener('click', () => load(pick(oks)));
  $('bReset').addEventListener('click', () => { Object.assign(m, DEF); $('loaded').textContent = ''; update(); });

  /* ---------- verdict ---------- */
  function verdict() {
    const e = evalTree(m), cut = 1 / (1 + st.r), on = e.p >= cut, v = vec(m);
    $('lamp').classList.toggle('on', on);
    $('lampText').textContent = on ? 'Alarm rings' : 'Stays quiet';
    $('lampSub').textContent = `Risk ${(100 * e.p).toFixed(1)}% against an alarm cutoff of ${(100 * cut).toFixed(1)}% (1 ÷ (1 + ${st.r % 1 ? st.r.toFixed(1) : st.r})).`;
    $('gFill').style.width = (100 * e.p) + '%'; $('gCut').style.left = `calc(${100 * cut}% - 1px)`; $('gCutLbl').textContent = `cutoff ${(100 * cut).toFixed(0)}%`;
    $('derived').innerHTML = [['Power', Math.round(v[6]).toLocaleString() + ' W'], ['Temperature gap', v[7].toFixed(1) + ' K'], ['Wear × torque', Math.round(v[8]).toLocaleString()]].map(([a, b]) => `<div><dt>${a}</dt><dd>${b}</dd></div>`).join('');
    const fmt = (p) => { if (p.f === 0) return `Quality type is ${'LMH'[p.v]}`; const big = p.f === 6 || p.f === 8, f = (x) => big ? Math.round(x).toLocaleString() : (+x).toFixed(1); return `${NAMES[p.f]} is ${f(p.v)}, which is ${p.le ? 'at most' : 'above'} ${f(p.thr)}`; };
    $('path').innerHTML = e.path.map((p) => `<li>${fmt(p)}</li>`).join('') + `<li><b>Result:</b> ${e.n} similar training machines, ${(100 * e.p).toFixed(1)}% of them failed.</li>`;
  }

  /* ---------- risk map ---------- */
  const AX = { air: ['Air temperature (K)', 295, 305], proc: ['Process temperature (K)', 305, 314], rpm: ['Speed (rpm)', 1100, 2900], torque: ['Torque (Nm)', 3, 77], wear: ['Tool wear (min)', 0, 255] };
  const ROWIDX = { air: 0, proc: 1, rpm: 2, torque: 3, wear: 4 };
  for (const id of ['ax', 'ay']) for (const k in AX) $(id).add(new Option(AX[k][0], k));
  $('ax').value = 'rpm'; $('ay').value = 'torque';
  $('ax').addEventListener('change', () => { if ($('ay').value === $('ax').value) $('ay').value = $('ax').value === 'rpm' ? 'torque' : 'rpm'; drawMap(); });
  $('ay').addEventListener('change', () => { if ($('ay').value === $('ax').value) $('ax').value = $('ay').value === 'rpm' ? 'torque' : 'rpm'; drawMap(); });
  $('showPts').addEventListener('change', drawMap);
  const mapGeom = () => { const cv = $('map'); return { cv, w: cv.clientWidth, h: 360, L: 46, B: 34, T: 10, Rr: cv.clientWidth - 12 }; };
  function drawMap() {
    const { cv, w, h, L, B, T: Tp, Rr } = mapGeom(), dpr = window.devicePixelRatio || 1; cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const kx = $('ax').value, ky = $('ay').value, [lx, x0, x1] = AX[kx], [ly, y0, y1] = AX[ky], pw = Rr - L, ph = h - B - Tp, cut = 1 / (1 + st.r);
    const X = (v) => L + (v - x0) / (x1 - x0) * pw, Y = (v) => Tp + ph - (v - y0) / (y1 - y0) * ph;
    const NX = 64, NY = 40, probe = { ...m };
    for (let i = 0; i < NX; i++) for (let j = 0; j < NY; j++) {
      probe[kx] = x0 + (i + 0.5) / NX * (x1 - x0); probe[ky] = y0 + (j + 0.5) / NY * (y1 - y0);
      const p = evalTree(probe).p; g.fillStyle = p >= cut ? `rgba(255,106,43,${0.22 + 0.5 * p})` : `rgba(60,195,210,${0.03 + 0.22 * p})`;
      g.fillRect(L + i / NX * pw, Tp + ph - (j + 1) / NY * ph, pw / NX + 0.6, ph / NY + 0.6);
    }
    if ($('showPts').checked) for (let i = 0; i < N; i++) {
      const r = rows[i], px = X(r[ROWIDX[kx]]), py = Y(r[ROWIDX[ky]]);
      if (y[i]) { g.fillStyle = '#fff'; g.beginPath(); g.arc(px, py, 3.2, 0, 6.3); g.fill(); g.strokeStyle = ALARM; g.lineWidth = 1.6; g.stroke(); }
      else { g.fillStyle = 'rgba(190,225,235,.35)'; g.beginPath(); g.arc(px, py, 1.3, 0, 6.3); g.fill(); }
    }
    g.fillStyle = TXT; g.font = '12.5px "Instrument Sans", system-ui, sans-serif'; g.textAlign = 'center';
    for (let k = 0; k <= 4; k++) { const v = x0 + k / 4 * (x1 - x0); g.fillText(v.toFixed(x1 - x0 < 20 ? 1 : 0), X(v), h - 16); }
    g.fillText(lx, L + pw / 2, h - 2); g.textAlign = 'right';
    for (let k = 0; k <= 4; k++) { const v = y0 + k / 4 * (y1 - y0); g.fillText(v.toFixed(y1 - y0 < 20 ? 1 : 0), L - 6, Y(v) + 4); }
    g.save(); g.translate(11, Tp + ph / 2); g.rotate(-Math.PI / 2); g.textAlign = 'center'; g.fillText(ly, 0, 0); g.restore();
    const cx = X(m[kx]), cy = Y(m[ky]); g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.setLineDash([4, 4]);
    g.beginPath(); g.moveTo(cx, Tp); g.lineTo(cx, Tp + ph); g.moveTo(L, cy); g.lineTo(L + pw, cy); g.stroke(); g.setLineDash([]);
    g.beginPath(); g.arc(cx, cy, 8, 0, 6.3); g.lineWidth = 3; g.strokeStyle = '#fff'; g.stroke(); g.fillStyle = evalTree(m).p >= cut ? ALARM : CALM; g.fill();
  }
  let drag = false;
  const mapPoint = (ev) => {
    const { cv, L, Rr, h, B, T: Tp } = mapGeom(), b = cv.getBoundingClientRect(), kx = $('ax').value, ky = $('ay').value;
    const fx = Math.min(1, Math.max(0, (ev.clientX - b.left - L) / (Rr - L))), fy = Math.min(1, Math.max(0, 1 - (ev.clientY - b.top - Tp) / (h - B - Tp)));
    for (const [k, f] of [[kx, fx], [ky, fy]]) { const [, lo, hi] = AX[k]; const step = SL.find((s) => s[0] === k)[4]; m[k] = Math.round((lo + f * (hi - lo)) / step) * step; }
    $('loaded').textContent = ''; update();
  };
  $('map').addEventListener('pointerdown', (e) => { drag = true; $('map').setPointerCapture(e.pointerId); mapPoint(e); });
  $('map').addEventListener('pointermove', (e) => { if (drag) mapPoint(e); });
  $('map').addEventListener('pointerup', () => { drag = false; });
  function update() { syncUI(); verdict(); drawMap(); }

  /* ---------- model comparison ---------- */
  const PAL = ['#8FA3BF', '#B58AF2', '#F2B134', '#3CC3D2', '#5E8BFF', '#7BD88F', '#FF6A2B', '#FF8FB1'];
  const keys = Object.keys(M), on = new Set([res.meta.winner, 'logreg__eng', 'tree__eng', 'forest__eng']);
  $('cmpBoxes').innerHTML = keys.map((k, i) => `<label><input type="checkbox" data-k="${k}" ${on.has(k) ? 'checked' : ''}><i style="background:${PAL[i]}"></i>${TM.labelOf(k)}</label>`).join('');
  $('cmpBoxes').addEventListener('change', (e) => { e.target.checked ? on.add(e.target.dataset.k) : on.delete(e.target.dataset.k); drawCmp(); });
  const lower = (a, v) => { let lo = 0, hi = a.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] < v) lo = mid + 1; else hi = mid; } return lo; };
  function sorted(k) { const p = T[st.scores][k], pos = [], neg = []; for (let i = 0; i < N; i++) (y[i] ? pos : neg).push(p[i]); return [pos.sort((a, b) => a - b), neg.sort((a, b) => a - b)]; }
  function drawCmp() {
    const W = 760, H = 330, l = 46, rt = 12, tp = 12, bt = 40, pw = W - l - rt, ph = H - tp - bt, ymax = 1.2, r = st.r, never = r * TM.NPOS;
    const X = (v) => l + v * pw, Y = (v) => tp + ph - Math.min(v, ymax) / ymax * ph;
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Cost against threshold for the selected models" xmlns="http://www.w3.org/2000/svg" font-family="Instrument Sans, system-ui, sans-serif" font-size="12">`;
    for (const v of [0, 0.25, 0.5, 0.75, 1, 1.2]) s += `<line x1="${l}" x2="${W - rt}" y1="${Y(v)}" y2="${Y(v)}" stroke="${GRID}"/><text x="${l - 6}" y="${Y(v) + 4}" text-anchor="end" fill="${TXT}">${v}</text>`;
    for (const v of [0, 0.25, 0.5, 0.75, 1]) s += `<text x="${X(v)}" y="${H - 20}" text-anchor="middle" fill="${TXT}">${v}</text>`;
    s += `<line x1="${l}" x2="${W - rt}" y1="${Y(1)}" y2="${Y(1)}" stroke="#fff" stroke-dasharray="4 4" opacity=".5"/><text x="${W - rt - 2}" y="${Y(1) - 5}" text-anchor="end" fill="${TXT}">never alarm = 1</text>`;
    const tbl = [];
    keys.forEach((k, i) => {
      if (!on.has(k)) return; const [pos, neg] = sorted(k), pts = [];
      for (let q = 0; q <= 200; q++) { const t = q / 200, c = r * lower(pos, t) + (neg.length - lower(neg, t)); pts.push(`${X(t).toFixed(1)},${Y(c / never).toFixed(1)}`); }
      s += `<polyline points="${pts.join(' ')}" fill="none" stroke="${PAL[i]}" stroke-width="${k === res.meta.winner ? 3 : 1.8}"/>`;
      const t = (st.scores === 'raw' ? M[k].thr_raw : M[k].thr_cal)[st.ri], fn = lower(pos, t), fp = neg.length - lower(neg, t);
      s += `<circle cx="${X(t)}" cy="${Y((r * fn + fp) / never)}" r="5" fill="${PAL[i]}" stroke="#fff" stroke-width="1.5"/>`;
      tbl.push(`<tr><td class="l"><i style="background:${PAL[i]};display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px"></i>${TM.labelOf(k)}</td><td>${t.toFixed(3)}</td><td>${TM.NPOS - fn}</td><td>${fn}</td><td>${fp}</td><td>$${Math.round((r * fn + fp) * st.fa).toLocaleString()}</td></tr>`);
    });
    s += `<text x="${l + pw / 2}" y="${H - 3}" text-anchor="middle" fill="${TXT}">alarm threshold (dots: each model’s training-fold choice)</text><text transform="rotate(-90)" x="${-(tp + ph / 2)}" y="12" text-anchor="middle" fill="${TXT}">cost ÷ never-alarm cost</text></svg>`;
    $('cmpChart').innerHTML = s;
    $('cmpTable').innerHTML = `<thead><tr><th class="l">Model</th><th>Threshold</th><th>Caught</th><th>Missed</th><th>False alarms</th><th>Cost</th></tr></thead><tbody>${tbl.join('')}</tbody>`;
  }
  window.addEventListener('tm:render', () => { verdict(); drawMap(); drawCmp(); });
  window.addEventListener('resize', drawMap);
  update(); drawCmp();
})();
