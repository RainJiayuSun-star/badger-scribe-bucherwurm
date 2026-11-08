"""Emit only queue IDs, numeric metrics and allowlisted status fields."""
import json
import pathlib
import re
import subprocess
import sys
import os
import socket
from datetime import datetime, timezone

RUN = re.compile(r'[A-Za-z0-9_.-]+')
ESTIMATES = {}  # planned minutes per job from <runs>/pm/estimates.json, if that file exists; loaded in main


def adapter_run(path):
    """runs/finetune/<run>/best_adapter -> <run>."""
    parts = [p for p in str(path).split('/') if p and p != 'best_adapter']
    return parts[-1] if parts and RUN.fullmatch(parts[-1]) else None


def training_time(root, run):
    """Seconds a fine-tune has run (so far, if still going) and its phase, from its own files."""
    folder = root / 'finetune' / run
    try:
        status = json.loads((folder / 'status.json').read_text())
    except (OSError, ValueError):
        status = {}
    if status.get('started'):
        start = datetime.fromisoformat(status['started']).timestamp()
        done = status.get('phase') in ('finished', 'failed')
        end = datetime.fromisoformat(status['updated']).timestamp() if done and status.get('updated') else datetime.now(timezone.utc).timestamp()
        return {'run': run, 'seconds': round(end - start), 'phase': status.get('phase'),
                **{k: status[k] for k in ('started', 'step', 'total_steps', 'eval_page', 'eval_pages', 'phase_started') if k in status}}
    try:  # older runs have no status.json; their metrics log the elapsed time per epoch
        rows = [json.loads(l) for l in (folder / 'metrics.jsonl').read_text().splitlines() if l.strip()]
        return {'run': run, 'seconds': round(max(r.get('elapsed_seconds', 0) for r in rows)), 'phase': 'finished'}
    except (OSError, ValueError):
        return {'run': run, 'seconds': None, 'phase': None}


def training_for(root, adapters):
    return [training_time(root, run) for run in sorted(set(filter(None, adapters)))]


def submissions(root):
    """Test-set prediction runs: existence and summary numbers only. The CSV itself is never opened."""
    rows = []
    for folder in sorted((root / 'finetune').glob('submission-*')):
        if not re.fullmatch(r'submission-[A-Za-z0-9_-]+', folder.name):
            continue
        try:
            summary = json.loads((folder / 'summary.json').read_text())
        except (OSError, ValueError):
            summary = {}
        csv_file = folder / 'submission.csv'
        adapters = [adapter_run(p) for p in (summary.get('adapters') or {}).values()]
        rows.append({'id': folder.name, 'csv': csv_file.exists(),
                     'finished_at': datetime.fromtimestamp(csv_file.stat().st_mtime, timezone.utc).isoformat() if csv_file.exists() else None,
                     'pages': summary.get('pages') if isinstance(summary.get('pages'), int) else None,
                     'empty_pages': len(summary['empty_pages']) if isinstance(summary.get('empty_pages'), list) else None,
                     'commit': str(summary.get('git_version', ''))[:12] or None,
                     'dirty': str(summary.get('git_version', '')).endswith('-dirty'),
                     'predict_seconds': round(summary['wall_seconds']) if isinstance(summary.get('wall_seconds'), (int, float)) else None,
                     'training': training_for(root, adapters)})
    return rows


def collect(root, queue):
    done = {}
    path = root / 'pm' / ('queue-' + queue + '.txt.done')
    for line in path.read_text().splitlines() if path.exists() else []:
        match = re.match(r'^(\S+) rc=(-?\d+)', line)
        if match:
            done[match[1]] = int(match[2])
    rows = []
    for line in (root / 'pm' / ('queue-' + queue + '.txt')).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        run = line.split()[0]
        if not re.fullmatch(r'[A-Za-z0-9_-]+', run):
            continue
        folder = root / 'finetune' / run
        try:
            status = json.loads((folder / 'status.json').read_text())
        except (OSError, ValueError):
            status = {}
        item = {'id': run, 'rc': done.get(run), 'status': {k: status[k] for k in
                ('phase', 'epoch', 'step', 'total_steps', 'eval_page', 'eval_pages', 'updated', 'started', 'phase_started') if k in status}, 'scores': []}
        if isinstance(ESTIMATES.get(run), (int, float)):
            item['planned_minutes'] = ESTIMATES[run]
        if run.startswith('submission-'):  # its fine-tunes, from the --adapter arguments on its queue line
            item['training'] = training_for(root, [adapter_run(a.split('=', 1)[1]) for a in re.findall(r'--adapter (\S+=\S+)', line)])
        item['process_alive'] = None
        if status.get('host') == socket.gethostname() and isinstance(status.get('pid'), int):
            try:
                os.kill(status['pid'], 0)
                item['process_alive'] = True
            except ProcessLookupError:
                item['process_alive'] = False
            except PermissionError:
                pass
        events = folder / 'events.jsonl'
        for event in events.read_text().splitlines() if events.exists() else []:
            try:
                e = json.loads(event)
                if e.get('event') == 'evaluated' and isinstance(e.get('cer'), (float, int)):
                    item['scores'].append({'epoch': e.get('epoch'), 'cer': e['cer']})
                # the first event of the current epoch marks when that epoch started
                if 'epoch_started' not in item and e.get('epoch') is not None and e.get('epoch') == status.get('epoch') \
                        and isinstance(e.get('time'), str):
                    item['epoch_started'] = e['time']
            except ValueError:
                pass
        rows.append(item)
    return rows


if __name__ == '__main__':
    queue = sys.argv[1]
    assert re.fullmatch(r'[A-Za-z0-9_-]+', queue)
    root = pathlib.Path(os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else '~/badger-scribe-bucherwurm/runs'))
    try:
        ESTIMATES.update(json.loads((root / 'pm' / 'estimates.json').read_text()))
    except (OSError, ValueError):
        pass
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    # container PID 1 starts with the workspace; etimes is wrong on these hosts, lstart isn't
    started = subprocess.check_output(['ps', '-o', 'lstart=', '-p', '1'], text=True, env={**os.environ, 'TZ': 'UTC'}).strip()
    print(json.dumps({'queue': queue, 'checked': datetime.now(timezone.utc).isoformat(),
                      'workspace_started': datetime.strptime(started, '%a %b %d %H:%M:%S %Y').replace(tzinfo=timezone.utc).isoformat(),
                      'gpu': {'name': gpu[0].strip(), 'used': int(gpu[1]), 'total': int(gpu[2]), 'utilization': int(gpu[3])},
                      'runs': collect(root, queue), 'submissions': submissions(root)}))
