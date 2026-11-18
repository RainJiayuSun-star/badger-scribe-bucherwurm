"""BadgerBrain section of the dashboard: gateway reachability, usage and local gateway runs. Numbers only."""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = 'https://llm-gw01.doit.wisc.edu'
USAGE_FIELDS = ('spend', 'model_spend', 'max_budget', 'rpm_limit', 'tpm_limit', 'expires', 'last_active')  # never the key or its names
RUNNING_SECONDS = 180  # a run with no finished page for this long is reported as stopped


def get(path, key, timeout=8):
    """(HTTP status, JSON body); status None means the gateway could not be reached."""
    request = urllib.request.Request(BASE + path, headers={'Authorization': 'Bearer ' + key} if key else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, None
    except (OSError, ValueError):
        return None, None


def gateway(key):
    started = time.monotonic()
    code, body = get('/v1/models', key)
    result = {'reachable': code is not None, 'latency_ms': round((time.monotonic() - started) * 1000) if code else None,
              'key_loaded': bool(key), 'key_accepted': code == 200 if key and code else None, 'models': [], 'usage': None}
    if code == 200:
        result['models'] = sorted(model['id'] for model in body.get('data', []))
        code, info = get('/key/info', key)
        if code == 200:
            result['usage'] = {field: info.get('info', {}).get(field) for field in USAGE_FIELDS}
    return result


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def fold_pages(runs_root, names):
    folds = [read_json(Path(runs_root) / 'folds' / f'{name}.json') for name in names]
    return sum(len(fold['eval_ids']) for fold in folds) if all(folds) else None


def gateway_runs(runs_root, now=None, limit=4):
    """Newest gateway screens first, from each run's config.json, pages.jsonl and summary.json."""
    now, rows = now or time.time(), []
    for path in (Path(runs_root) / 'finetune').glob('*/config.json'):
        config = read_json(path)
        log = path.with_name('pages.jsonl')
        if not isinstance(config, dict) or 'gateway' not in config or not log.is_file() or path.parent.name.startswith('smoke-'):
            continue
        pages = {}
        for line in log.read_text().splitlines():
            try:
                record = json.loads(line)
                pages[record['page_id']] = record
            except (ValueError, KeyError, TypeError):
                continue  # a page being written right now
        summary, updated = read_json(path.with_name('summary.json')) or {}, log.stat().st_mtime
        records = list(pages.values())
        rows.append({'id': path.parent.name, 'model': config.get('model'), 'pages': len(pages), 'total': fold_pages(runs_root, config.get('folds', [])),
                     'state': 'complete' if summary else 'running' if now - updated < RUNNING_SECONDS else 'stopped', 'updated': updated,
                     'seconds_per_page': round(sum(r.get('seconds', 0) for r in records) / len(records), 1) if records else None,
                     'tokens': sum((r.get('usage') or {}).get('completion_tokens') or 0 for r in records),
                     'hit_limit': sum(r.get('finish_reason') == 'length' for r in records), 'retries': sum(r.get('attempt', 1) > 1 for r in records),
                     'cer_by_category': summary.get('cer_by_category'), 'loop_cuts': summary.get('loop_cuts'), 'wall_seconds': summary.get('wall_seconds')})
    return sorted(rows, key=lambda row: row['updated'], reverse=True)[:limit]


def status(key, runs_root):
    return {**gateway(key), 'runs': gateway_runs(runs_root), 'checked': time.time()}
