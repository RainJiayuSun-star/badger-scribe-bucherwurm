"""Emit allowlisted settings and numeric metrics for every run folder.

Never opens prediction CSVs or pages.jsonl, and keeps only numbers and short
identifiers, so no transcription text can leave the workspace.
"""
import json
import pathlib
import re
import sys

CONFIG_KEYS = ('model_id', 'revision', 'adapter_policy', 'lora_rank', 'lora_alpha', 'learning_rate',
               'vision_learning_rate', 'epochs', 'effective_batch_size', 'seed', 'human_repeats',
               'category_repeats', 'curriculum', 'augmentation_profile', 'max_pixels', 'max_pixels_by_category',
               'repetition_penalty', 'loop_retry_temperatures', 'quantization', 'dtype')
SUMMARY_KEYS = ('cer', 'cer_by_category', 'loop_cuts', 'pages', 'peak_gpu_gib', 'wall_seconds', 'model_id',
                'revision', 'git_version', 'gpu')
KEY = re.compile(r'[\w.-]{1,80}')
VALUE = re.compile(r'[\w./@+:-]{1,120}')  # no spaces, so no sentence of text fits


def clean(value):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if VALUE.fullmatch(value) else None
    if isinstance(value, list):
        return [clean(v) for v in value[:20] if not isinstance(v, (dict, list))]
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if KEY.fullmatch(str(k))}
    return None


def ident(value):
    """A fold or adapter path reduced to its run or fold name."""
    parts = [p for p in str(value).split('/') if p and p != 'best_adapter']
    name = re.sub(r'\.(json|yaml)$', '', parts[-1]) if parts else ''
    return name if KEY.fullmatch(name) else None


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def lines(path):
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


def adapters(value):
    return {k: ident(v) for k, v in value.items() if KEY.fullmatch(str(k))} if isinstance(value, dict) else None


def collect_run(folder):
    config, summary = read_json(folder / 'config.json'), read_json(folder / 'summary.json')
    run = {'id': folder.name,
           'config': {k: clean(config[k]) for k in CONFIG_KEYS if k in config},
           'summary': {k: clean(summary[k]) for k in SUMMARY_KEYS if k in summary},
           'fold': ident(config.get('fold') or summary.get('fold') or '') or None,
           'adapters': adapters(summary.get('adapters')) or adapters(config.get('adapters'))}
    if isinstance(summary.get('empty_pages'), list):
        run['summary']['empty_pages'] = len(summary['empty_pages'])
    metric_rows = lines(folder / 'metrics.jsonl')
    run['epochs'] = [{k: v for k, v in row.items() if isinstance(v, (int, float)) and
                      (k in ('epoch', 'elapsed_seconds', 'peak_gpu_memory_mb') or k.startswith('val_'))}
                     for row in metric_rows if 'val_macro_cer' in row]
    run['loss'] = [[row['step'], row['epoch'], round(row['train_loss'], 5)] for row in metric_rows
                   if all(isinstance(row.get(k), (int, float)) for k in ('step', 'epoch', 'train_loss'))]
    events = lines(folder / 'events.jsonl')
    if not run['epochs']:  # a run that crashed can have evals in events.jsonl but no metrics.jsonl
        run['epochs'] = [{'epoch': e.get('epoch'), 'val_macro_cer': e['cer'], 'val_human_macro_cer': e.get('human_cer'),
                          'val_loop_cuts': e.get('loop_cuts')} for e in events
                         if e.get('event') == 'evaluated' and e.get('epoch') is not None and isinstance(e.get('cer'), (int, float))]
    zero_shot = read_json(folder / 'zero_shot' / 'summary.json')
    run['zero_shot'] = {fold: {k: v for k, v in vals.items() if isinstance(v, (int, float))}
                        for fold, vals in zero_shot.items() if isinstance(vals, dict) and KEY.fullmatch(fold)}
    times =[clean(e.get('time')) for e in events if isinstance(e.get('time'), str)]
    run['started'] = times[0] if times else None
    run['ended'] = next((clean(e.get('time')) for e in reversed(events) if e.get('event') in ('finished', 'failed')), None)
    failed = [e for e in events if e.get('event') == 'failed']
    run['outcome'] = 'failed' if failed else 'finished' if any(e.get('event') == 'finished' for e in events) else None
    if failed:
        match = re.match(r'(\w+(?:Error|Exception))', str(failed[-1].get('error', '')))
        run['error'] = match[1] if match else 'error'
    host = read_json(folder / 'status.json').get('host')
    run['host'] = host if isinstance(host, str) and KEY.fullmatch(host) else None
    try:
        run['mtime'] = (folder / 'config.json').stat().st_mtime
    except OSError:
        run['mtime'] = None
    return run


def collect(root):
    return [collect_run(d) for d in sorted(pathlib.Path(root).iterdir()) if d.is_dir() and KEY.fullmatch(d.name)]


if __name__ == '__main__':
    import os
    print(json.dumps(collect(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/badger-scribe-bucherwurm/runs/finetune'))))
