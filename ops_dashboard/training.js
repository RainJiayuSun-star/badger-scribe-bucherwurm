'use strict';
// Training tab: loss and held-out CER charts as inline SVG. Uses $ and esc from app.js.
// Colors are the first three slots of the validated dark categorical palette (blue, orange, aqua),
// checked against the panel color #181a19; each follows a category, never a rank.
const CAT_COLOR = {kade_letters: '#3987e5', survey_notes: '#d95926', dominy_accounts: '#199e70'};
const CAT_NAME = {kade_letters: 'German letters', survey_notes: 'Survey notes', dominy_accounts: 'Account books'};
const ONE = '#3987e5', INK = '#a2aa99', GRID = '#2a2f28', SURFACE = '#181a19';
let training = null;
const charts = {};

function smooth(points) {  // centered 3-point rolling mean; keeps the logged value for the tooltip
  return points.map((p, i) => {
    const w = points.slice(Math.max(0, i - 1), i + 2);
    return [p[0], w.reduce((s, q) => s + q[1], 0) / w.length, p[1]];
  });
}
function lossPoints(run) {  // x in epochs: logged step over the last step, times the last epoch
  const last = Math.max(...run.loss.map(l => l[0])), epochs = Math.max(...run.loss.map(l => l[1]));
  return smooth(run.loss.map(([step, , loss]) => [step / last * epochs, loss]));
}
function niceMax(v) {
  const p = Math.pow(10, Math.floor(Math.log10(v || 1))), f = v / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
}

function lineChart(id, {title, cap, series, xTitle, digits, markers}) {
  const W = 560, H = 250, L = 52, R = series.length > 1 ? 110 : 18, T = 12, B = 40;
  const all = series.flatMap(s => s.pts);
  if (!all.length) return `<div class="chart"><h3>${esc(title)}</h3><p class="cap">${esc(cap)}</p><div class="empty none">No data for this chart.</div></div>`;
  const xMax = Math.max(1, Math.ceil(Math.max(...all.map(p => p[0])))), yMax = niceMax(Math.max(...all.map(p => p[1])) * 1.05);
  const sx = x => L + x / xMax * (W - L - R), sy = y => T + (1 - y / yMax) * (H - T - B);
  const fmt = v => v.toFixed(digits);
  let g = '';
  for (let i = 0; i <= 4; i++) {
    const v = yMax * i / 4;
    g += `<line x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}" stroke="${GRID}" stroke-width="1"/><text x="${L - 8}" y="${sy(v) + 4}" text-anchor="end" fill="${INK}" font-size="11">${+v.toFixed(3)}</text>`;
  }
  for (let x = 0; x <= xMax; x++) g += `<text x="${sx(x)}" y="${H - B + 18}" text-anchor="middle" fill="${INK}" font-size="11">${x}</text>`;
  g += `<line x1="${L}" x2="${W - R}" y1="${sy(0)}" y2="${sy(0)}" stroke="${INK}" stroke-width="1"/>`;
  g += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle" fill="${INK}" font-size="11">${esc(xTitle)}</text>`;
  const labels = [];
  for (const s of series) {
    if (!s.pts.length) continue;
    g += `<path d="${s.pts.map((p, i) => (i ? 'L' : 'M') + sx(p[0]).toFixed(1) + ' ' + sy(p[1]).toFixed(1)).join(' ')}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    if (markers) g += s.pts.map(p => `<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="4.5" fill="${s.color}" stroke="${SURFACE}" stroke-width="2"/>`).join('');
    const p = s.pts[s.pts.length - 1];
    labels.push({x: sx(p[0]) + 8, y: sy(p[1]) + 4, text: s.short});
  }
  if (series.length > 1) {  // direct labels at the line ends, pushed apart so they never overlap
    labels.sort((a, b) => a.y - b.y);
    labels.forEach((l, i) => { if (i && l.y - labels[i - 1].y < 13) l.y = labels[i - 1].y + 13; });
    g += labels.map(l => `<text x="${l.x}" y="${l.y}" fill="${INK}" font-size="11">${esc(l.text)}</text>`).join('');
  }
  g += `<line class="cross" x1="0" x2="0" y1="${T}" y2="${H - B}" stroke="${INK}" stroke-width="1" stroke-dasharray="3 3" visibility="hidden"/>`;
  g += `<rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"/>`;
  charts[id] = {series, sx, ix: px => (px - L) / (W - L - R) * xMax, W, fmt};
  const legend = series.length > 1 ? `<div class="legend">${series.map(s => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join('')}</div>` : '';
  const rows = series.flatMap(s => s.pts.map(p => `<tr><td>${esc(s.name)}</td><td class="mono">${p[0].toFixed(2)}</td><td class="mono">${fmt(p[1])}</td>${p[2] != null ? `<td class="mono">${fmt(p[2])}</td>` : ''}</tr>`)).join('');
  const logged = series.some(s => s.pts.some(p => p[2] != null)) ? '<th>Logged</th>' : '';
  return `<div class="chart"><h3>${esc(title)}</h3><p class="cap">${esc(cap)}</p>
    <svg viewBox="0 0 ${W} ${H}" data-chart="${id}" role="img" aria-label="${esc(title)}">${g}</svg>${legend}<div class="tip"></div>
    <details><summary>DATA TABLE</summary><div class="tablewrap"><table><thead><tr><th>Series</th><th>${esc(xTitle)}</th><th>Value</th>${logged}</tr></thead><tbody>${rows}</tbody></table></div></details></div>`;
}

function barChart(id, {title, cap, items}) {
  const W = 560, H = 250, L = 52, R = 18, T = 22, B = 40;
  if (!items.length) return `<div class="chart"><h3>${esc(title)}</h3><div class="empty none">No public scores recorded yet.</div></div>`;
  const yMax = niceMax(Math.max(...items.map(i => i.value)) * 1.1), band = (W - L - R) / items.length, bw = Math.min(56, band * 0.5);
  const sy = y => T + (1 - y / yMax) * (H - T - B);
  let g = '';
  for (let i = 0; i <= 4; i++) {
    const v = yMax * i / 4;
    g += `<line x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}" stroke="${GRID}"/><text x="${L - 8}" y="${sy(v) + 4}" text-anchor="end" fill="${INK}" font-size="11">${v.toFixed(3)}</text>`;
  }
  items.forEach((it, i) => {
    const x = L + band * i + (band - bw) / 2, y = sy(it.value), h = sy(0) - y, r = Math.min(4, h);
    // 4px rounded top, square at the baseline
    g += `<path d="M${x} ${sy(0)}V${y + r}Q${x} ${y} ${x + r} ${y}H${x + bw - r}Q${x + bw} ${y} ${x + bw} ${y + r}V${sy(0)}Z" fill="${ONE}"><title>${esc(it.label)}: public CER ${it.value.toFixed(3)}</title></path>`;
    g += `<text x="${x + bw / 2}" y="${y - 6}" text-anchor="middle" fill="#e4e4d7" font-size="11">${it.value.toFixed(3)}</text>`;
    g += `<text x="${x + bw / 2}" y="${H - B + 18}" text-anchor="middle" fill="${INK}" font-size="11">${esc(it.label.replace('submission-', 'sub-'))}</text>`;
  });
  g += `<line x1="${L}" x2="${W - R}" y1="${sy(0)}" y2="${sy(0)}" stroke="${INK}"/>`;
  return `<div class="chart"><h3>${esc(title)}</h3><p class="cap">${esc(cap)}</p><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">${g}</svg>
    <details><summary>DATA TABLE</summary><div class="tablewrap"><table><thead><tr><th>Submission</th><th>Public CER</th></tr></thead><tbody>${items.map(i => `<tr><td>${esc(i.label)}</td><td class="mono">${i.value.toFixed(3)}</td></tr>`).join('')}</tbody></table></div></details></div>`;
}

function hover(e) {
  const svg = e.target.closest('svg[data-chart]');
  if (!svg) return;
  const c = charts[svg.dataset.chart], box = svg.getBoundingClientRect(), wrap = svg.parentElement, tip = wrap.querySelector('.tip');
  const x = c.ix((e.clientX - box.left) / box.width * c.W);
  const hits = c.series.map(s => [s, s.pts.reduce((b, p) => !b || Math.abs(p[0] - x) < Math.abs(b[0] - x) ? p : b, null)]).filter(h => h[1]);
  if (!hits.length) return;
  const cross = svg.querySelector('.cross'), at = hits[0][1][0];
  cross.setAttribute('x1', c.sx(at)); cross.setAttribute('x2', c.sx(at)); cross.setAttribute('visibility', 'visible');
  tip.innerHTML = `<div>Epoch ${at.toFixed(2)}</div>` + hits.map(([s, p]) => `<div><i style="display:inline-block;width:10px;height:2px;background:${s.color};vertical-align:middle;margin-right:6px"></i>${esc(s.name)}: ${c.fmt(p[1])}${p[2] != null ? ' (logged ' + c.fmt(p[2]) + ')' : ''}</div>`).join('');
  const wb = wrap.getBoundingClientRect();
  tip.style.display = 'block';
  tip.style.left = Math.min(e.clientX - wb.left + 14, wb.width - tip.offsetWidth - 4) + 'px';
  tip.style.top = (e.clientY - wb.top + 14) + 'px';
}
function unhover(e) {
  const svg = e.target.closest('svg[data-chart]');
  if (!svg || svg.contains(e.relatedTarget)) return;  // moving between marks inside the chart
  svg.querySelector('.cross').setAttribute('visibility', 'hidden');
  svg.parentElement.querySelector('.tip').style.display = 'none';
}

// Fold runs from before the 11-02 code review used different augmentation, so they don't stand in for today's adapters.
const SAME_CODE_FROM = '2026-11-03';
function estimate(adapter, category) {
  // Held-out CER of fold runs trained like this adapter (same base, settings and category), at the adapter's last epoch.
  const same = training.runs.filter(r => r.category === category && r.model === adapter.model && (r.started || '') >= SAME_CODE_FROM
    && JSON.stringify(r.setup) === JSON.stringify(adapter.setup) && r.val.some(([e]) => e === adapter.epochs));
  const folds = same.map(r => [r.fold, r.val.find(([e]) => e === adapter.epochs)[1]]).sort();
  return folds.length ? {cer: folds.reduce((s, [, c]) => s + c, 0) / folds.length, folds} : null;
}
function collectionsHtml() {
  const byId = Object.fromEntries(training.runs.map(r => [r.id, r]));
  const subs = training.runs.filter(r => r.id.startsWith('submission-') && r.adapters).sort((a, b) => a.id.localeCompare(b.id));
  const latest = subs.filter(s => training.public[s.id] != null).pop() || subs[subs.length - 1];
  if (!latest) return '<div class="none">No submission runs recorded yet.</div>';
  const rows = Object.keys(CAT_COLOR).filter(c => latest.adapters[c]).map(c => {
    const run = byId[latest.adapters[c]], est = run && estimate(run, c);
    const basis = est ? est.folds.map(([f, v]) => `${esc(f)} ${v.toFixed(3)}`).join(', ') + ` (epoch ${run.epochs})` : 'No matching held-out runs';
    return `<tr><td><i style="display:inline-block;width:10px;height:10px;background:${CAT_COLOR[c]};margin-right:8px"></i>${CAT_NAME[c]}</td><td>${esc(run ? run.model : '?')}</td><td class="id">${esc(latest.adapters[c])}</td><td class="mono">${est ? '~' + est.cer.toFixed(3) : '—'}</td><td style="color:var(--muted)">${basis}</td></tr>`;
  }).join('');
  const score = training.public[latest.id];
  return `<div class="tablewrap"><table><thead><tr><th>Collection</th><th>Model</th><th>Adapter</th><th>Held-out CER</th><th>Estimated from</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="notice">${esc(latest.id)}${score != null ? ` · Kaggle public CER <b>${score.toFixed(3)}</b> (the average of the three; Kaggle doesn't split it)` : ''}. Held-out CER is the mean of fold runs trained the same way, scored on documents they never saw, at the adapter's last epoch. The submitted adapters trained on every page, so this is an estimate. Survey and account labels are mostly machine drafts.</p>`;
}

function latestHtml() {
  const byId = Object.fromEntries(training.runs.map(r => [r.id, r]));
  const subs = training.runs.filter(r => r.id.startsWith('submission-') && r.adapters).sort((a, b) => a.id.localeCompare(b.id));
  const latest = subs.filter(s => training.public[s.id] != null).pop() || subs[subs.length - 1];
  const scores = Object.entries(training.public).sort(([a], [b]) => a.localeCompare(b)).map(([label, value]) => ({label, value}));
  const bars = barChart('public', {title: 'Kaggle public CER by submission', cap: 'Lower is better. Scored on a fixed subset of test pages.', items: scores});
  if (!latest) return `<div class="charts"><div class="chart"><div class="empty none">No submission runs recorded yet.</div></div>${bars}</div>`;
  const runs = {};  // adapter run -> the categories it decodes
  for (const [cat, run] of Object.entries(latest.adapters)) (runs[run] = runs[run] || []).push(cat);
  const order = Object.keys(CAT_COLOR);
  const series = Object.entries(runs).filter(([run]) => byId[run]?.loss?.length).map(([run, cats]) => {
    cats.sort((a, b) => order.indexOf(a) - order.indexOf(b));
    const label = cats.map(c => CAT_NAME[c]).join(' + ');
    return {name: `${label} · ${byId[run].model} (${run})`, short: label.split(' ')[0], color: CAT_COLOR[cats[0]], pts: lossPoints(byId[run])};
  });
  const score = training.public[latest.id];
  return `<p class="notice" style="margin-top:0">${esc(latest.id)}${score != null ? ` · public CER <b>${score.toFixed(3)}</b>` : ' · no public score recorded yet'}</p>
    <div class="charts">${lineChart('latestloss', {title: `Training loss, ${latest.id}'s adapters`, cap: 'All-data runs, so no held-out test curve. The public CER on the right is their test.', series, xTitle: 'Epoch', digits: 3})}${bars}</div>`;
}

function runHtml(id) {
  const run = training.runs.find(r => r.id === id);
  if (!run) return '';
  const loss = run.loss.length ? [{name: 'Training loss', color: ONE, pts: lossPoints(run)}] : [];
  const val = run.val.length ? [{name: 'Held-out CER', color: ONE, pts: run.val.map(([e, c]) => [e, c])}] : [];
  const valCap = run.val.length ? `CER on ${run.fold || 'the held-out pages'} at the end of each epoch.` : run.fold === 'all_train' ? 'Trained on all 619 training pages, so nothing is held out.' : 'No held-out scores logged.';
  return `<div class="charts">${lineChart('runloss', {title: `Training loss · ${run.model}`, cap: `${run.id}${run.fold ? ', fold ' + run.fold : ''}`, series: loss, xTitle: 'Epoch', digits: 3})}${lineChart('runval', {title: 'Held-out CER by epoch', cap: valCap, series: val, xTitle: 'Epoch', digits: 4, markers: true})}</div>`;
}

function drawTraining() {
  if (!training) return;
  $('collections').innerHTML = $('collections2').innerHTML = collectionsHtml();
  $('latest').innerHTML = latestHtml();
  const pick = $('trainrun'), keep = pick.value;
  const runs = training.runs.filter(r => !r.id.startsWith('submission-') && (r.loss.length || r.val.length))
    .sort((a, b) => (b.started || '').localeCompare(a.started || ''));
  pick.innerHTML = runs.map(r => `<option value="${esc(r.id)}">${esc(r.id)} · ${esc(r.model)}</option>`).join('');
  if (runs.some(r => r.id === keep)) pick.value = keep;
  $('onerun').innerHTML = runHtml(pick.value);
}
async function loadTraining() {
  try {
    const r = await fetch('/api/training');
    if (!r.ok) throw Error();
    training = await r.json();
    drawTraining();
  } catch (e) {
    if (!training) $('latest').innerHTML = '<div class="loading">Training data is not available from the dashboard server.</div>';
  }
}
$('trainrun').onchange = () => { $('onerun').innerHTML = runHtml($('trainrun').value); };
document.addEventListener('mousemove', hover);
document.addEventListener('mouseout', unhover);
loadTraining(); setInterval(loadTraining, 600000);
