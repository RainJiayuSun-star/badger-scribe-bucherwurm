"""Local-only dashboard. Read-only SSH polls; no model or transcription access."""
import concurrent.futures
import copy
import csv
import json
import shlex
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from telemetry import snapshot
from live_state import enrich
import settings
import history

ROOT = Path(__file__).resolve().parent
CONFIG = settings.load()
state = {'gpus': [], 'refreshing': True, 'kaggle': None, 'kaggle_checked': 0, 'history_checked': 0}
KAGGLE_EVERY = 600  # seconds between Kaggle and run-history checks; Kaggle's API is rate limited
lock = threading.Lock()
trigger = threading.Event()


def poll(workspace):
    command = ' '.join(['python3 -c', shlex.quote((ROOT / 'collect_remote.py').read_text()),
                        shlex.quote(workspace['queue']), shlex.quote(CONFIG['remote_runs'])])
    result = subprocess.run(settings.ssh(workspace, command, 50), capture_output=True, text=True, timeout=50, check=True)
    return json.loads(result.stdout)


def kaggle_uploads():
    """Our Kaggle uploads, oldest first, without errored ones; None if the CLI has no token (~/.kaggle)."""
    try:
        out = subprocess.run(['kaggle', 'competitions', 'submissions', CONFIG.get('kaggle_competition', 'badger-scribe'), '--csv'],
                             capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = [{k.lower().replace('_', ''): v for k, v in row.items() if k} for row in csv.DictReader(out.splitlines())]
    return sorted((r for r in rows if r.get('date') and 'error' not in r.get('status', '').lower()), key=lambda r: r['date'])


def refresh():
    while True:
        if time.time() - state['kaggle_checked'] > KAGGLE_EVERY:
            uploads = kaggle_uploads()
            with lock:
                state['kaggle_checked'] = time.time()
                if uploads is not None:  # a failed check keeps the last good list
                    state['kaggle'] = uploads
        if time.time() - state['history_checked'] > KAGGLE_EVERY:
            state['history_checked'] = time.time()
            try:
                history.update_cache(history.fetch())  # training curves for the Training tab
            except (OSError, ValueError):
                pass  # an unreadable cache must not stop the GPU polling
        with lock:
            state['refreshing'] = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(CONFIG['workspaces']))) as pool:
            jobs = {w['queue']: pool.submit(poll, w) for w in CONFIG['workspaces']}
            for queue, future in jobs.items():
                try:
                    item = future.result()
                    item['error'] = None
                except Exception:
                    with lock:
                        item = next((dict(g) for g in state['gpus'] if g['queue'] == queue), {'queue': queue, 'runs': []})
                    item['error'] = 'Workspace unreachable. Last successful snapshot retained.'
                with lock:
                    state['gpus'] = sorted([g for g in state['gpus'] if g['queue'] != queue] + [item], key=lambda g: g['queue'])
        with lock:
            state['refreshing'] = False
        trigger.wait(120)
        trigger.clear()


def submission_status(gpus, kaggle=None):
    """Merge the workspaces' submission runs with the local copies, history_notes.json and Kaggle's upload list."""
    try:
        notes = json.loads((ROOT / 'history_notes.json').read_text())
    except (OSError, ValueError):
        notes = {}
    scores, uploaded = dict(notes.get('public_scores', {})), set(notes.get('submitted', [])) | set(notes.get('public_scores', {}))
    remote = {s['id']: s for g in gpus for s in g.get('submissions', [])}
    local_runs = ROOT.parent / CONFIG.get('local_runs', 'runs')  # relative to the repo root
    local = {p.parent.name for p in (local_runs / 'finetune').glob('submission-*/submission.csv')}
    ids = sorted(set(remote) | local)
    # Kaggle names every upload submission.csv, so the nth upload is taken to be the nth submission folder.
    # shortcut: breaks if a folder is skipped or uploaded twice; match on description if that ever happens.
    kaggle_status = {}
    for sid, row in zip(ids, kaggle or []):
        uploaded.add(sid)
        kaggle_status[sid] = row.get('status', '')
        try:
            scores[sid] = float(row.get('publicscore') or '')
        except ValueError:
            pass  # not scored yet
    rows, previous = [], None
    for sid in ids:
        s = dict(remote.get(sid, {'id': sid, 'csv': True}))
        evaluation = notes.get('evaluations', {}).get(sid)
        if evaluation is None and sid in scores and previous:
            delta = scores[sid] - scores[previous]
            verdict = 'better' if delta < -0.005 else 'worse' if delta > 0.005 else "a tie under the plan's 0.005 rule"
            evaluation = f'public {scores[sid]:.3f} against {previous} {scores[previous]:.3f}: {delta:+.3f}, {verdict}'
        s.update(on_mac=sid in local, uploaded=sid in uploaded, public_score=scores.get(sid),
                 kaggle_status=kaggle_status.get(sid), evaluation=evaluation)
        rows.append(s)
        if sid in scores:
            previous = sid
    return rows


def training_payload():
    """Loss and held-out CER curves per run, from history_runs.json (numbers and identifiers only)."""
    try:
        runs = json.loads(history.CACHE.read_text())
        notes = json.loads((ROOT / 'history_notes.json').read_text())
    except (OSError, ValueError):
        runs, notes = [], {}
    rows = []
    for r in runs:
        val = [[e['epoch'], e['val_macro_cer']] for e in r.get('epochs', [])
               if isinstance(e.get('epoch'), (int, float)) and isinstance(e.get('val_macro_cer'), (int, float))]
        if r.get('loss') or val or r['id'].startswith('submission-'):
            c = r['config']
            rows.append({'id': r['id'], 'model': history.short(c.get('model_id') or r['summary'].get('model_id')),
                         'fold': r.get('fold'), 'category': history.category(r.get('fold')), 'epochs': c.get('epochs'),
                         'loss': r.get('loss', []), 'val': val, 'adapters': r.get('adapters'), 'started': r.get('started'),
                         # the settings that must match for a fold run to stand in for an all-data adapter
                         'setup': [c.get('adapter_policy'), c.get('learning_rate'), c.get('human_repeats') or 1, c.get('category_repeats') or None]})
    return {'runs': rows, 'public': notes.get('public_scores', {})}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/status':
            with lock:
                payload = copy.deepcopy(state)
            payload['agents'] = snapshot()
            payload['models'] = enrich(payload['gpus'], payload['agents'], time.time())
            kaggle = payload.pop('kaggle')
            payload['submissions'] = submission_status(payload['gpus'], kaggle)
            payload['kaggle_connected'] = kaggle is not None
            payload['roles'], payload['names'] = CONFIG.get('roles', {}), CONFIG.get('names', {})
            payload['served_at'] = time.time()
            body = json.dumps(payload).encode()
            kind = 'application/json'
        elif self.path == '/api/training':
            body = json.dumps(training_payload()).encode()
            kind = 'application/json'
        elif self.path == '/training.js':
            body = (ROOT / 'training.js').read_bytes()
            kind = 'text/javascript; charset=utf-8'
        elif self.path in ('/', '/index.html'):
            body = (ROOT / 'index.html').read_bytes()
            kind = 'text/html; charset=utf-8'
        elif self.path == '/app.js':
            body = (ROOT / 'app.js').read_bytes()
            kind = 'text/javascript; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != '/api/refresh':
            self.send_error(404)
            return
        trigger.set()
        self.send_response(202)
        self.end_headers()

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    threading.Thread(target=refresh, daemon=True).start()
    port = int(CONFIG.get('port', 8766))
    print(f'Dashboard: http://127.0.0.1:{port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
