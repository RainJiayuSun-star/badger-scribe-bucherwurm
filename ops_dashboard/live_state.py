"""Derive display state from observed runs and reported tasks, never fixed stage text."""
from datetime import datetime, timezone

MODEL_NAMES = {'churro': 'Churro / 3B', 'olmocr': 'olmOCR-2 / 7B',
               'lighton': 'LightOnOCR / 4B', 'weight-space': 'Weight-space pilot'}


def age(timestamp, now):
    try:
        return max(0, now - datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp())
    except (ValueError, TypeError, AttributeError):
        return None


def model_for(run_id):
    if 'wsp-' in run_id:
        return 'weight-space'
    if 'olmocr' in run_id:
        return 'olmocr'
    if 'lighton' in run_id:
        return 'lighton'
    return 'churro'


def run_state(run, stale=False):
    if run.get('rc') is not None:
        return 'complete' if run['rc'] == 0 else 'failed'
    phase = run.get('status', {}).get('phase')
    if phase == 'finished':
        return 'complete'
    if phase == 'failed':
        return 'failed'
    if not phase:
        return 'queued'
    if stale or run.get('process_alive') is False:
        return 'unknown'
    return 'active'


def enrich(gpus, registry, now):
    runs = []
    for gpu in gpus:
        elapsed = age(gpu.get('checked'), now)
        gpu['stale'] = bool(gpu.get('error')) or elapsed is None or elapsed > 300
        for run in gpu['runs']:
            heartbeat_age = age(run.get('status', {}).get('updated'), now)
            run['model'] = model_for(run['id'])
            run['display_state'] = run_state(run, gpu['stale'] or (heartbeat_age is not None and heartbeat_age > 600))
            run['unit'] = gpu['queue']
            runs.append(run)
    models = []
    for key, name in MODEL_NAMES.items():
        jobs = [r for r in runs if r['model'] == key]
        tasks = [t for t in registry['tasks'] if t['model'] == key]
        active_jobs = [r for r in jobs if r['display_state'] == 'active']
        active_tasks = [t for t in tasks if t['display_state'] == 'active']
        unknown = [r for r in jobs if r['display_state'] == 'unknown'] + [t for t in tasks if t['display_state'] == 'unknown']
        waiting = [t for t in tasks if t['display_state'] in ('waiting', 'blocked')]
        complete = [r for r in jobs if r['display_state'] == 'complete']
        if active_jobs:
            stages = sorted({'evaluation' if r['status'].get('phase') == 'evaluating' else 'validation' if 'smoke' in r['id'] else 'fine-tuning' for r in active_jobs})
            stage, status = ' + '.join(stages), 'active'
            note = ' · '.join(r['id'] for r in active_jobs)
        elif active_tasks:
            stage, status = active_tasks[-1]['stage'], 'active'
            note = active_tasks[-1]['title']
        elif unknown:
            stage, status, note = 'unknown', 'unknown', 'An active task or run stopped reporting. Current stage is unconfirmed.'
        elif waiting:
            task = waiting[0]
            stage, status, note = task['stage'], task['display_state'], task['title']
        elif any(r['display_state'] == 'queued' for r in jobs):
            stage, status, note = 'fine-tuning', 'waiting', 'Jobs are queued; no active run is confirmed.'
        elif complete or any(t['display_state'] == 'completed' for t in tasks):
            stage, status, note = 'review', 'waiting', 'Work completed. No active run or release approval is reported.'
        else:
            stage, status, note = 'unknown', 'unknown', 'No execution or task event is available.'
        models.append({'id': key, 'name': name, 'stage': stage, 'state': status,
                       'note': note, 'completed_runs': len(complete), 'queued_runs': sum(r['display_state'] == 'queued' for r in jobs)})
    return models
