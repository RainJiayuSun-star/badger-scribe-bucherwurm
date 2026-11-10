"""Upload finished submission CSVs to Kaggle by themselves, once each, within the daily limit.

Run on the Mac from the repo root, in the background:
    nohup python3 runs/pm/bin/auto_submit.py submission-03 submission-04 > runs/pm/auto_submit.out 2>&1 &

For each run ID it waits until EFabric has a complete prediction (208 pages, none empty, a clean
commit), copies the CSV to runs/finetune/<id>/, checks it against sample_submission.csv by page ID
and hash, and uploads it only if fewer than 2 submissions were made today (Kaggle counts by UTC
day). The text column is never printed or logged. Uploaded IDs are added to "submitted" in
runs/ops-dashboard/history_notes.json. Gives up on an ID once the workspaces are gone and it
never appeared. Needs the kaggle CLI and ~/.kaggle/kaggle.json.
"""
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HOST = ['-p', '2200', 'igloege_ian-badgerscribe_0@128.105.218.242']
REMOTE = '~/badger-scribe-bucherwurm/runs/finetune'
NOTES = REPO / 'runs/ops-dashboard/history_notes.json'
LOG = REPO / 'runs/pm/auto_submit.log'
COMPETITION = 'badger-scribe'
DAILY_LIMIT = 2
MESSAGES = {
    'submission-03': 'Churro-3B with adapters trained on all training pages: German 4 epochs, English with the same settings as submission 2.',
    'submission-04': 'Churro-3B for German letters, olmOCR-2-7B LoRA for survey notes and account books, adapters trained on all training pages.',
}


def log(message):
    line = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {message}"
    print(line, flush=True)
    with LOG.open('a') as handle:
        handle.write(line + '\n')


def ssh(command, binary=False):
    result = subprocess.run(['ssh', '-n', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', *HOST, command],
                            capture_output=True, timeout=300)
    if result.returncode == 255:
        raise ConnectionError('workspace unreachable')
    return result.stdout if binary else result.stdout.decode()


def remote_ready(run_id):
    """Summary numbers and the CSV's hash if the run finished cleanly, else None."""
    out = ssh(f"cd {REMOTE}/{run_id} 2>/dev/null && test -f submission.csv && test -f summary.json && "
              "python3 -c \"import json;s=json.load(open('summary.json'));"
              "print(json.dumps({k:(len(s[k]) if k=='empty_pages' else s.get(k)) for k in ('pages','empty_pages','git_version')}))\" && "
              "sha256sum submission.csv | cut -d' ' -f1")
    lines = out.split()
    if len(lines) < 2:
        return None
    summary = json.loads(out.rsplit('\n', 2)[0])
    summary['sha256'] = lines[-1]
    return summary


def check_csv(path, expected_sha):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        return 'hash differs from EFabric'
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['page_id', 'text']:
            return f'columns are {reader.fieldnames}'
        ids = [row['page_id'] for row in reader]
    with (REPO / 'badger-scribe-data/sample_submission.csv').open(encoding='utf-8-sig', newline='') as handle:
        expected = [row['page_id'] for row in csv.DictReader(handle)]
    if ids != expected:
        return f'page IDs differ from sample_submission.csv ({len(ids)} rows)'
    return None


def submitted_today():
    out = subprocess.run(['kaggle', 'competitions', 'submissions', COMPETITION, '--csv'],
                         capture_output=True, text=True, timeout=120, check=True).stdout
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    rows = list(csv.DictReader(out.splitlines()))
    return sum(1 for row in rows if str(row.get('date', '')).startswith(today))


def public_score(message):
    """The public score of our upload with this description, once Kaggle has scored it."""
    out = subprocess.run(['kaggle', 'competitions', 'submissions', COMPETITION, '--csv'],
                         capture_output=True, text=True, timeout=120, check=True).stdout
    for row in csv.DictReader(out.splitlines()):
        row = {k.lower().replace('_', ''): v for k, v in row.items() if k}
        if row.get('description') == message:
            try:
                return float(row.get('publicscore') or '')
            except ValueError:
                return None  # uploaded but not scored yet
    return None


def evaluate(run_id, score):
    """Record the score and compare it with the previous scored submission (lower is better)."""
    notes = json.loads(NOTES.read_text())
    earlier = {k: v for k, v in notes.get('public_scores', {}).items() if k < run_id and isinstance(v, (int, float))}
    notes.setdefault('public_scores', {})[run_id] = score
    if earlier:
        base = max(earlier)
        delta = score - earlier[base]
        verdict = ('better' if delta < -0.005 else 'worse' if delta > 0.005 else "a tie under the plan's 0.005 rule")
        text = f'public {score:.3f} against {base} {earlier[base]:.3f}: {delta:+.3f}, {verdict}'
    else:
        text = f'public {score:.3f}; no earlier score recorded to compare with'
    notes.setdefault('evaluations', {})[run_id] = text
    NOTES.write_text(json.dumps(notes, indent=2) + '\n')
    log(f'{run_id} {text}')


def mark_submitted(run_id):
    notes = json.loads(NOTES.read_text())
    if run_id not in notes.setdefault('submitted', []):
        notes['submitted'].append(run_id)
        NOTES.write_text(json.dumps(notes, indent=2) + '\n')


def main(targets):
    if not shutil.which('kaggle'):
        sys.exit('kaggle CLI not found; install it first')
    if not (Path.home() / '.kaggle/kaggle.json').exists():
        sys.exit('~/.kaggle/kaggle.json is missing; add your Kaggle API token first')
    pending, unreachable = [t for t in targets if t in MESSAGES], 0
    verified = {}  # run ID -> facts, once its Mac copy has passed the checks; no EFabric needed after that
    scoring = {}   # uploaded run ID -> when it was uploaded, until its public score appears
    log(f'watching {pending}')
    while pending or scoring:
        for run_id, since in list(scoring.items()):
            try:
                score = public_score(MESSAGES[run_id])
            except subprocess.CalledProcessError:
                score = None
            if score is not None:
                evaluate(run_id, score); del scoring[run_id]
            elif time.time() - since > 3 * 3600:
                log(f'{run_id} has no public score after 3 hours; stopped checking'); del scoring[run_id]
        try:
            for run_id in list(pending):
                if run_id in json.loads(NOTES.read_text()).get('submitted', []):
                    log(f'{run_id} already uploaded; skipping'); pending.remove(run_id); continue
                local = REPO / 'runs/finetune' / run_id / 'submission.csv'
                ready = verified.get(run_id)
                if ready is None:
                    ready = remote_ready(run_id)
                    if not ready:
                        continue
                    if ready['pages'] != 208 or ready['empty_pages'] != 0 or str(ready['git_version']).endswith('-dirty'):
                        log(f'{run_id} finished but is not uploadable: {ready}; leaving it for Ian'); pending.remove(run_id); continue
                    local.parent.mkdir(parents=True, exist_ok=True)
                    local.write_bytes(ssh(f'cat {REMOTE}/{run_id}/submission.csv', binary=True))
                    problem = check_csv(local, ready['sha256'])
                    if problem:
                        log(f'{run_id} failed the check ({problem}); not uploading'); pending.remove(run_id); continue
                    verified[run_id] = ready
                    log(f'{run_id} copied to the Mac and verified')
                elif check_csv(local, ready['sha256']):
                    log(f'{run_id} Mac copy changed since it was verified; not uploading'); pending.remove(run_id); continue
                used = submitted_today()
                if used >= DAILY_LIMIT:
                    log(f'{run_id} is ready but {used} submissions were already made today (UTC); waiting'); continue
                subprocess.run(['kaggle', 'competitions', 'submit', COMPETITION, '-f', str(local), '-m', MESSAGES[run_id]],
                               capture_output=True, text=True, timeout=600, check=True)
                mark_submitted(run_id)
                log(f'{run_id} uploaded (sha256 {ready["sha256"][:16]}, commit {str(ready["git_version"])[:12]}); waiting for its public score')
                pending.remove(run_id)
                scoring[run_id] = time.time()
            unreachable = 0
        except ConnectionError:
            unreachable += 1
            unfetched = [p for p in pending if p not in verified]
            if unreachable >= 6 and unfetched:  # about 30 minutes: the workspaces have stopped
                log(f'workspaces unreachable; giving up on {unfetched}, which never finished')
                pending = [p for p in pending if p in verified]
        except subprocess.CalledProcessError as error:
            log(f'kaggle command failed with exit {error.returncode}; retrying next round')
        if pending or scoring:
            time.sleep(120 if scoring and not pending else 300)
    log('all targets handled')


if __name__ == '__main__':
    main(sys.argv[1:])
