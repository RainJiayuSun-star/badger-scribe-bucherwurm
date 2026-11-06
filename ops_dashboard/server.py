"""Local-only dashboard. Read-only SSH polls; no model or transcription access."""
import concurrent.futures
import copy
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

ROOT = Path(__file__).resolve().parent
CONFIG = settings.load()
state = {'gpus': [], 'refreshing': True}
lock = threading.Lock()
trigger = threading.Event()


def poll(workspace):
    command = ' '.join(['python3 -c', shlex.quote((ROOT / 'collect_remote.py').read_text()),
                        shlex.quote(workspace['queue']), shlex.quote(CONFIG['remote_runs'])])
    result = subprocess.run(settings.ssh(workspace, command, 50), capture_output=True, text=True, timeout=50, check=True)
    return json.loads(result.stdout)


def refresh():
    while True:
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


def submission_status(gpus):
    """Merge EFabric's submission runs with the Mac copies and what has been uploaded (history_notes.json)."""
    try:
        notes = json.loads((ROOT / 'history_notes.json').read_text())
    except (OSError, ValueError):
        notes = {}
    scores, uploaded = notes.get('public_scores', {}), set(notes.get('submitted', [])) | set(notes.get('public_scores', {}))
    remote = {s['id']: s for g in gpus for s in g.get('submissions', [])}
    local_runs = ROOT.parent / CONFIG.get('local_runs', 'runs')  # relative to the repo root
    local = {p.parent.name for p in (local_runs / 'finetune').glob('submission-*/submission.csv')}
    rows = []
    for sid in sorted(set(remote) | local):
        s = dict(remote.get(sid, {'id': sid, 'csv': True}))
        s.update(on_mac=sid in local, uploaded=sid in uploaded, public_score=scores.get(sid))
        rows.append(s)
    return rows


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/status':
            with lock:
                payload = copy.deepcopy(state)
            payload['agents'] = snapshot()
            payload['models'] = enrich(payload['gpus'], payload['agents'], time.time())
            payload['submissions'] = submission_status(payload['gpus'])
            payload['roles'], payload['names'] = CONFIG.get('roles', {}), CONFIG.get('names', {})
            payload['served_at'] = time.time()
            body = json.dumps(payload).encode()
            kind = 'application/json'
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
