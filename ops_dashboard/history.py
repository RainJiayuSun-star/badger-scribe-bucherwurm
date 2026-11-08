"""Build history.html, a log of every fine-tuning, decode and screen run.

Run from the repo root: python3 ops_dashboard/history.py [--offline] [--no-open]
Run metadata comes from your EFabric workspaces (config.json) through collect_history.py,
numbers and identifiers only, and is cached in history_runs.json for when they are gone.
Public scores and hand-written notes go in history_notes.json.
"""
import argparse
import json
import pathlib
import re
import shlex
import subprocess
import webbrowser
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent
CACHE, NOTES, OUT = HERE / 'history_runs.json', HERE / 'history_notes.json', HERE / 'history.html'
CATEGORIES = {'kade_letters': 'German letters', 'survey_notes': 'Survey notes', 'dominy_accounts': 'Account books'}
MODELS = {'stanford-oval/churro-3B': 'Churro-3B', 'allenai/olmOCR-2-7B-1025': 'olmOCR-2-7B',
          'lightonai/LightOnOCR-3-4B': 'LightOnOCR-3-4B',
          'yale-cultural-heritage/LightOnOCR-3-4B-cultural-heritage-eng': "Yale's LightOnOCR-3-4B"}
FAMILIES = [  # first matching prefix wins
    ('submission-', 'Test-set prediction', 'Decoded all 208 test pages for Kaggle with the adapters listed. Test labels are hidden, so only the public score applies.'),
    ('check-', 'Adapter check', 'Reloaded a saved adapter from disk and re-scored the fold it held out, to prove it loads.'),
    ('screen-', 'Base screen', 'Plan step 6: zero-shot decode of another open-weight model on the account books and survey_011.'),
    ('dec-base-', 'Decode test', 'Plan step 2 baseline: re-decode with the default settings on the A5000.'),
    ('dec-rp10-', 'Decode test', 'Plan step 2: repetition penalty off (1.0) instead of the model default of 1.05.'),
    ('dec-2mp-', 'Decode test', 'Plan step 3: decode at a 2 MP image budget.'),
    ('dec-4mp-', 'Decode test', 'Plan step 3: decode at a 4 MP image budget.'),
    ('dec-noretry-', 'Decode test', 'Plan step 5 baseline: decode without the sampled retry on loop-cut pages.'),
    ('dec-retry-', 'Decode test', 'Plan step 5: sampled retry on pages the loop guard cut.'),
    ('ft11-vis-', 'Fine-tune', 'Plan step 11: LoRA on the vision encoder as well as the language layers.'),
    ('ft10-lr2e4e6-', 'Fine-tune', 'Plan step 10: learning rate 2e-4 for 6 epochs.'),
    ('ft10-lr1e4e6-', 'Fine-tune', 'Plan step 10: the default learning rate for 6 epochs.'),
    ('ft10-lr2e4-', 'Fine-tune', 'Plan step 10: learning rate 2e-4 instead of 1e-4.'),
    ('ft10-base-', 'Fine-tune', 'Plan step 10 baseline: default settings with the fixed augmentation, on the A5000.'),
    ('ft9-seed', 'Fine-tune', 'Plan step 9: same settings with a different random seed, for voting and noise.'),
    ('ft8-repeats-', 'Fine-tune', 'Plan step 8: account-book pages repeated in each epoch.'),
    ('ft6-olmocr2-', 'Fine-tune', 'Plan step 6 follow-up: LoRA on olmOCR-2-7B instead of Churro, compared with ft5 on the same fold.'),
    ('ft5-', 'Fine-tune', 'Plan step 5: account-book fold on the current code, the baseline for account-book tests.'),
    ('ft4-', 'Fine-tune', 'First full-length run of the 11-02 code, with the fixed augmentation.'),
    ('churro-bf16-ft-', 'Fine-tune', 'First LoRA fine-tunes (10-17), one held-out document each.'),
    ('churro-bf16-zeroshot-kade-guard', 'Zero-shot', 'Churro-3B with no fine-tuning, with the loop guard on.'),
    ('churro-bf16-zeroshot-', 'Zero-shot', 'Churro-3B with no fine-tuning.'),
    ('churro-bf16-retry-', 'Zero-shot', 'Churro-3B with no fine-tuning, plus a sampled retry on looping pages.'),
    ('smoke-', 'Pipeline test', 'Short run to test code, not an experiment.'),
    ('ckpt-test', 'Pipeline test', 'Short run to test checkpoint resume, not an experiment.'),
    ('selftest', 'Pipeline test', 'Queue runner self-test, not an experiment.'),
    ('ft', 'Fine-tune', 'LoRA fine-tune with one document held out.'),
]


def fetch():
    import settings
    config = settings.load()
    script = (HERE / 'collect_history.py').read_text()
    command = ' '.join(['python3 -c', shlex.quote(script), shlex.quote(config['remote_runs'].rstrip('/') + '/finetune')])
    for workspace in config['workspaces']:  # the workspaces share a home, so the first that answers is enough
        try:
            out = subprocess.run(settings.ssh(workspace, command, 120), capture_output=True, text=True, timeout=120, check=True).stdout
            return json.loads(out)
        except (subprocess.SubprocessError, ValueError, OSError):
            continue
    return None


def update_cache(fresh):
    """Merge a fresh run list into history_runs.json, keeping runs the workspace no longer has."""
    cached = {r['id']: r for r in json.loads(CACHE.read_text())} if CACHE.exists() else {}
    if fresh is not None:
        cached.update({r['id']: r for r in fresh})
        CACHE.write_text(json.dumps(list(cached.values())))
    return cached


def category(fold):
    fold = fold or ''
    return next((key for prefix, key in (('kade', 'kade_letters'), ('survey', 'survey_notes'), ('dominy', 'dominy_accounts'),
                                         ('group', 'dominy_accounts')) if fold.startswith(prefix)), None)


def short(model_id):
    cached = re.search(r'models--([^/]+?)--([^/]+)', model_id or '')  # a local Hugging Face cache path
    model_id = f'{cached[1]}/{cached[2]}' if cached else model_id
    return MODELS.get(model_id, model_id or 'unknown model')


def best_epoch(epochs):
    scored = [e for e in epochs if isinstance(e.get('val_macro_cer'), (int, float))]
    return min(scored, key=lambda e: e['val_macro_cer']) if scored else None


def approach(run, kind):
    c, parts = run['config'], []
    if kind == 'Fine-tune' or kind == 'Pipeline test' and run['epochs']:
        layers = 'language and vision layers' if c.get('adapter_policy') == 'vision_inclusive' else 'language layers'
        lr = f"learning rate {c['learning_rate']:g}" if 'learning_rate' in c else 'default learning rate'
        if c.get('adapter_policy') == 'vision_inclusive' and 'vision_learning_rate' in c:
            lr += f" (vision {c['vision_learning_rate']:g})"
        parts.append(f"LoRA rank {c.get('lora_rank', '?')}, alpha {c.get('lora_alpha', '?')}, on the {layers}; {lr}; "
                     f"{c.get('epochs', '?')} epochs; batch {c.get('effective_batch_size', '?')}; seed {c.get('seed', '?')}.")
        if run.get('fold'):
            parts.append(f"Held out {run['fold']} ({CATEGORIES.get(category(run['fold']), 'unknown data')}).")
        if (c.get('human_repeats') or 1) > 1:
            parts.append(f"Human-labelled pages repeated {c['human_repeats']} times per epoch.")
        if c.get('category_repeats'):
            parts.append('Category repeats: ' + ', '.join(f'{k} x{v}' for k, v in c['category_repeats'].items()) + '.')
    elif kind in ('Decode test', 'Base screen', 'Adapter check', 'Test-set prediction'):
        target = f"held-out fold {run['fold']}" if run.get('fold') else 'training pages' if kind == 'Base screen' else 'the test set'
        parts.append(f'No training. Decoded {target}.')
        if c.get('repetition_penalty') is not None:
            parts.append(f"Repetition penalty {c['repetition_penalty']}.")
        if c.get('loop_retry_temperatures'):
            parts.append('Retry temperatures ' + ', '.join(map(str, c['loop_retry_temperatures'])) + '.')
    elif kind == 'Zero-shot':
        parts.append('No training. Decoded every training page, one document at a time.')
    mp = c.get('max_pixels')
    if mp:
        extra = ', '.join(f'{CATEGORIES.get(k, k)} {v / 1e6:.1f} MP' for k, v in (c.get('max_pixels_by_category') or {}).items())
        parts.append(f'Image budget {mp / 1e6:.1f} MP' + (f' ({extra}).' if extra else '.'))
    return ' '.join(parts)


def metrics(run, kind):
    """Rows of (model label, {category: CER}) on held-out training pages."""
    base = short(run['config'].get('model_id') or run['summary'].get('model_id'))
    rows = []
    best = best_epoch(run['epochs'])
    if best:
        cells = {k: best[f'val_cer_{k}'] for k in CATEGORIES if isinstance(best.get(f'val_cer_{k}'), (int, float))}
        if not cells and category(run.get('fold')):
            cells = {category(run['fold']): best['val_macro_cer']}
        rows.append((f"{base} + this run's LoRA (best epoch {best.get('epoch')})", cells))
        human = best.get('val_human_macro_cer')
        if isinstance(human, (int, float)) and category(run.get('fold')) and abs(human - best['val_macro_cer']) > 1e-9:
            rows.append(('Same, human-labelled pages only', {category(run['fold']): best['val_human_macro_cer']}))
    summary = run['summary']
    if summary.get('cer_by_category') or isinstance(summary.get('cer'), (int, float)):
        cells = {k: v for k, v in (summary.get('cer_by_category') or {}).items() if k in CATEGORIES}
        if not cells and category(run.get('fold')):
            cells = {category(run['fold']): summary['cer']}
        names = run.get('adapters') or {}
        label = base + (' + adapters ' + ', '.join(sorted(set(filter(None, names.values())))) if names else ' (no adapter)' if kind == 'Base screen' else '')
        rows.append((label, cells))
    if run['zero_shot'] and not summary.get('cer_by_category'):  # screens keep a copy of the same scores there
        totals = {}
        for fold, vals in run['zero_shot'].items():
            cat, cer, pages = category(fold), vals.get('val_macro_cer'), vals.get('pages') or 0
            if cat and isinstance(cer, (int, float)) and pages:
                n, s = totals.get(cat, (0, 0.0))
                totals[cat] = (n + pages, s + cer * pages)
        rows.append((f'{base} (no adapter), page-weighted over documents', {k: s / n for k, (n, s) in totals.items()}))
    return rows


def describe(run, notes):
    kind, note = next(((k, n) for prefix, k, n in FAMILIES if run['id'].startswith(prefix)), ('Other', ''))
    when = run.get('started') or (datetime.fromtimestamp(run['mtime'], timezone.utc).isoformat() if run.get('mtime') else None)
    summary, best = run['summary'], best_epoch(run['epochs'])
    models = [short(run['config'].get('model_id') or summary.get('model_id'))]
    if kind == 'Fine-tune':
        models[0] += ' + LoRA adapter trained in this run'
    for cat, name in sorted((run.get('adapters') or {}).items()):
        models.append(f'{CATEGORIES.get(cat, cat)}: adapter from {name}')
    peak = summary.get('peak_gpu_gib') or max((e.get('peak_gpu_memory_mb', 0) for e in run['epochs']), default=0) / 1024 or None
    seconds = summary.get('wall_seconds') or max((e.get('elapsed_seconds', 0) for e in run['epochs']), default=0) or None
    rows = metrics(run, kind)
    by_category = summary.get('cer_by_category') or {}
    outcome = run.get('outcome')
    if not outcome:  # older runs have no events log, so judge by what they recorded
        planned = run['config'].get('epochs')
        outcome = ('finished' if not planned or len(run['epochs']) >= planned else 'incomplete') if run['epochs'] else \
            'finished' if summary or run['zero_shot'] else 'unknown'
    data = CATEGORIES.get(category(run.get('fold')), '') if run.get('fold') else \
        'Test set' if kind == 'Test-set prediction' else ', '.join(CATEGORIES[k] for k in CATEGORIES if any(k in c for _, c in rows))
    return {'id': run['id'], 'base': short(run['config'].get('model_id') or summary.get('model_id')), 'kind': kind, 'note': notes.get('notes', {}).get(run['id']) or note, 'when': when,
            'outcome': outcome, 'error': run.get('error'), 'data': data, 'fold': run.get('fold'),
            'models': models, 'approach': approach(run, kind), 'metrics': rows,
            'epochs': [{k: e.get(k) for k in ('epoch', 'val_macro_cer', 'val_human_macro_cer', 'val_wer', 'val_loop_cuts')} for e in run['epochs']],
            'best': best['val_macro_cer'] if best else summary.get('cer', next(iter(by_category.values())) if len(by_category) == 1 else None),
            'public': notes.get('public_scores', {}).get(run['id']),
            'facts': {'GPU host': run.get('host'), 'Commit': (summary.get('git_version') or '')[:12] or None,
                      'Wall time (min)': round(seconds / 60, 1) if seconds else None,
                      'Peak GPU memory (GiB)': round(peak, 1) if peak else None, 'Pages': summary.get('pages'),
                      'Loop cuts': summary.get('loop_cuts'), 'Empty pages': summary.get('empty_pages')}}


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fine-tuning log</title>
<style>:root{--bg:#101111;--panel:#181a19;--line:#343a32;--ink:#e4e4d7;--muted:#a2aa99;--amber:#efac51;--red:#ee5947;--green:#b9d69b}*{box-sizing:border-box}
body{margin:0;background-color:var(--bg);background-image:radial-gradient(ellipse at top,rgba(16,17,17,0) 0%,rgba(16,17,17,.85) 75%),url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='100'%3E%3Cg fill='none' stroke='%23efac51' stroke-opacity='.11' stroke-width='1'%3E%3Cpath d='M28 66L0 50V16L28 0l28 16v34L28 66v34'/%3E%3Cpath d='M28 0v34L0 50v34l28 16 28-16V50L28 34'/%3E%3C/g%3E%3C/svg%3E");background-attachment:fixed;color:var(--ink);font:14px/1.5 Arial,sans-serif}.shell{max-width:1200px;margin:auto;padding:0 32px 32px}
.top{border-bottom:1px solid var(--line);display:flex;justify-content:space-between;padding:20px 0;font:800 16px Arial;letter-spacing:3px}.top b{color:var(--red)}.top span{font:11px monospace;color:var(--muted);letter-spacing:1px}
h1{font-size:clamp(30px,4vw,48px);margin:30px 0 6px;letter-spacing:-1px}.lede{color:var(--muted);margin:0 0 22px}.eyebrow{font:11px monospace;letter-spacing:2px;color:var(--amber)}
.filters{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}input,select{font:inherit;color:var(--ink);background:#20231f;border:1px solid var(--line);padding:9px 12px}input{flex:1;min-width:160px}
input:focus-visible,select:focus-visible,summary:focus-visible{outline:2px solid var(--amber);outline-offset:3px}
details{border:1px solid var(--line);background:var(--panel);margin-bottom:8px}summary{cursor:pointer;display:grid;grid-template-columns:minmax(0,2.2fr) 1.2fr 1fr 1fr .8fr .8fr;gap:12px;padding:13px 16px;align-items:center;list-style:none}summary::-webkit-details-marker{display:none}
summary .id{font:13px monospace;overflow-wrap:anywhere}summary .id:before{content:'\\25B8  ';color:var(--red)}details[open] summary .id:before{content:'\\25BE  '}summary small{color:var(--muted);font:11px monospace}.cer{font:15px monospace;color:var(--amber)}
.badge{font:10px monospace;text-transform:uppercase;letter-spacing:1px;padding:3px 6px;border:1px solid currentColor;color:var(--amber);justify-self:start}.badge.good{color:var(--green)}.badge.bad{color:var(--red)}.badge.dim{color:var(--muted)}
.body{border-top:1px solid var(--line);padding:16px 18px 18px;display:grid;gap:14px}.body h3{font:11px monospace;letter-spacing:1.5px;color:var(--amber);margin:0 0 6px}.body h3:before{content:'// ';color:var(--red)}.body p,.body li{margin:0;color:var(--ink);font-size:13px}.body ul{margin:0;padding-left:18px}
table{border-collapse:collapse;width:100%;font-size:12px}th{font:10px monospace;letter-spacing:1px;color:var(--muted);text-align:left;background:#1d201b}th,td{padding:8px 10px;border-bottom:1px solid var(--line)}td.n{font-family:monospace}.wrap{overflow:auto}
.facts{display:flex;flex-wrap:wrap;gap:6px 22px;font:11px monospace;color:var(--muted)}.facts b{color:var(--ink);font-weight:normal}.note{font:11px/1.6 monospace;color:var(--muted);border-left:2px solid var(--line);padding-left:12px;margin:14px 0}
@media(max-width:700px){.shell{padding:0 16px 24px}summary{grid-template-columns:1fr 1fr}summary .id{grid-column:1/-1}}</style></head><body><div class="shell">
<div class="top"><div><b>&#11042;</b> B&Uuml;CHERWURM <span>/ FINE-TUNING LOG</span></div><span>BUILT __BUILT__</span></div>
<div class="eyebrow">EXPERIMENT HISTORY / __COUNT__ RUNS</div><h1>Every run, one record.</h1>
<p class="lede">Scores are character error rate on held-out training pages, lower is better, unless marked public. Each run is built from its own run files.</p>
<div class="filters"><input id="q" placeholder="Filter by run ID, model or data&hellip;" aria-label="Filter runs"><select id="kind" aria-label="Run type"><option value="">All run types</option></select><select id="cat" aria-label="Data type"><option value="">All data</option><option>German letters</option><option>Survey notes</option><option>Account books</option></select></div>
<div id="list"></div><p class="note">Fine-tune scores are the saved adapter's best epoch on the one held-out document, so runs on different documents aren't directly comparable. Zero-shot rows are page-weighted over documents.</p></div>
<script>
const RUNS=__DATA__;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=v=>typeof v==='number'?v.toFixed(4):'&mdash;';
const CATS=['German letters','Survey notes','Account books'],KEYS={'German letters':'kade_letters','Survey notes':'survey_notes','Account books':'dominy_accounts'};
const when=s=>s?new Date(s).toLocaleString('en-US',{timeZone:'America/Chicago',month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit'})+' CST':'date unknown';
const kinds=[...new Set(RUNS.map(r=>r.kind))].sort();document.getElementById('kind').innerHTML+=kinds.map(k=>`<option>${esc(k)}</option>`).join('');
function badge(r){const c=r.outcome==='finished'?'good':r.outcome==='failed'?'bad':'dim';return `<span class="badge ${c}">${esc(r.outcome==='failed'?'failed: '+(r.error||'error'):r.outcome)}</span>`}
function card(r){
 const score=r.public!=null?`${fmt(r.public)} <small>public</small>`:r.best!=null?fmt(r.best):'&mdash;';
 const mrows=r.metrics.map(([label,cells])=>`<tr><td>${esc(label)}</td>${CATS.map(c=>`<td class="n">${fmt(cells[KEYS[c]])}</td>`).join('')}</tr>`).join('');
 const erows=r.epochs.map(e=>`<tr><td class="n">${esc(e.epoch)}</td><td class="n">${fmt(e.val_macro_cer)}</td><td class="n">${fmt(e.val_human_macro_cer)}</td><td class="n">${fmt(e.val_wer)}</td><td class="n">${e.val_loop_cuts??'&mdash;'}</td></tr>`).join('');
 const facts=Object.entries(r.facts).filter(([k,v])=>v!=null&&v!=='').map(([k,v])=>`<span>${esc(k)} <b>${esc(v)}</b></span>`).join('');
 return `<details><summary><span class="id">${esc(r.id)}</span><span>${esc(r.base)}</span><small>${esc(r.kind)}</small><small>${esc(r.data||'')}</small><span class="cer">${score}</span>${badge(r)}</summary><div class="body">
 <div><h3>Models</h3><ul>${r.models.map(m=>`<li>${esc(m)}</li>`).join('')}</ul></div>
 <div><h3>Approach</h3><p>${esc(r.note)} ${esc(r.approach)}</p></div>
 <div><h3>Performance by data type</h3>${mrows?`<div class="wrap"><table><thead><tr><th>Model</th>${CATS.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>${mrows}</tbody></table></div>`:'<p>No scores recorded for this run.</p>'}${r.public!=null?`<p style="margin-top:8px">Public leaderboard (macro over all three): <b>${fmt(r.public)}</b></p>`:''}</div>
 ${erows?`<div><h3>By epoch</h3><div class="wrap"><table><thead><tr><th>Epoch</th><th>CER</th><th>Human pages CER</th><th>WER</th><th>Loop cuts</th></tr></thead><tbody>${erows}</tbody></table></div></div>`:''}
 <div class="facts"><span>Started <b>${esc(when(r.when))}</b></span>${facts}</div></div></details>`}
function draw(){const q=document.getElementById('q').value.toLowerCase(),k=document.getElementById('kind').value,c=document.getElementById('cat').value;
 const rows=RUNS.filter(r=>(!k||r.kind===k)&&(!c||r.data.includes(c)||r.metrics.some(([,cells])=>cells[KEYS[c]]!=null))&&(r.id+' '+r.base+' '+r.models.join(' ')+' '+r.data).toLowerCase().includes(q));
 document.getElementById('list').innerHTML=rows.length?rows.map(card).join(''):'<p class="note">No runs match.</p>'}
['q','kind','cat'].forEach(id=>document.getElementById(id).addEventListener('input',draw));draw();
</script></body></html>"""


def build(runs, notes):
    records = sorted((describe(r, notes) for r in runs), key=lambda r: r['when'] or '', reverse=True)
    data = json.dumps(records).replace('</', '<\\/')
    built = datetime.now().strftime('%b %d %H:%M')
    return PAGE.replace('__DATA__', data).replace('__COUNT__', str(len(records))).replace('__BUILT__', built.upper())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true', help='use the cached run list without contacting EFabric')
    parser.add_argument('--no-open', action='store_true', help="build the page without opening it in the browser")
    args = parser.parse_args()
    fresh = None if args.offline else fetch()
    cached = update_cache(fresh)
    if not cached:
        raise SystemExit('EFabric is unreachable and there is no cache yet.')
    notes = json.loads(NOTES.read_text()) if NOTES.exists() else {}
    OUT.write_text(build(list(cached.values()), notes))
    print(f"{OUT} with {len(cached)} runs ({'fresh from EFabric' if fresh is not None else 'from cache'})")
    if not args.no_open:
        webbrowser.open(OUT.as_uri())


if __name__ == '__main__':
    main()
