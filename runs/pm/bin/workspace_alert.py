"""One pop-up when a new submission file is ready, and one when an EFabric workspace stops.

A finished submission (submission.csv and summary.json both written) is first copied to
runs/finetune/<id> on the Mac and checked by hash; a crashed one is never copied.

Run in the background from the repo root:
    nohup python3 runs/pm/bin/workspace_alert.py > /dev/null 2>&1 &
Each alert is sent once, ever: what has been sent is saved in runs/pm/alert_state.json, so a restart
can't repeat it. Nothing restarts this script if it stops. Settings are in runs/pm/alert_config.json:
after renting new workspaces, replace the "workspaces" entries (the file is reread every check).
Texts are off unless "texts": true and "phone" are set. Log: runs/pm/workspace_alert.log.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PM = Path(__file__).resolve().parents[1]
FINETUNE = PM.parent / 'finetune'
CONFIG, STATE, LOG, PIDFILE = PM / 'alert_config.json', PM / 'alert_state.json', PM / 'workspace_alert.log', PM / 'workspace_alert.pid'
CHECK_SECONDS = 180
PENDING = PM / 'pending_runs.json'  # last count of unfinished queue lines, kept for when the workspaces are down
RENEW_UNTIL = datetime(2026, 12, 2, tzinfo=timezone.utc)  # final deadline: never rent after it
QUEUES = {'ian-badgerscribe': 'gpu1', 'ian-badgerscribe2': 'gpu2'}  # workspace name -> its queue file
PENDING_CMD = ("cd ~/badger-scribe-bucherwurm/runs/pm && for q in queue-gpu1.txt queue-gpu2.txt; do grep -v '^#' $q | awk 'NF{print $1}' "
               "| while read i; do grep -q \"^$i \" $q.done || echo $i; done; done | wc -l")


def log(message):
    with LOG.open('a') as handle:
        handle.write(f'{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {message}\n')


def load_state():
    try:
        return set(json.loads(STATE.read_text())['sent'])
    except (OSError, ValueError, KeyError):
        return set()


def send_once(key, message, config, sent):
    """Pop up (and text, if enabled) only if this key has never been sent."""
    if key in sent:
        return
    sent.add(key)
    STATE.write_text(json.dumps({'sent': sorted(sent)}, indent=2) + '\n')  # saved before sending, so a crash can't resend
    # The message goes in as an argument, so quotes in it can't break the AppleScript.
    subprocess.run(['osascript', '-e', 'on run argv', '-e',
                    'display notification (item 1 of argv) with title "Bücherwurm" sound name "Glass"',
                    '-e', 'end run', message], capture_output=True)
    phone = str(config.get('phone') or '')
    if config.get('texts') and re.fullmatch(r'[\d\s().+-]+', phone):
        digits = re.sub(r'[\s().-]', '', phone)
        digits = '+1' + digits if re.fullmatch(r'\d{10}', digits) else digits
        if re.fullmatch(r'\+?\d{10,15}', digits):  # only ever the one number in alert_config.json
            subprocess.run(['osascript', '-e', 'on run argv', '-e', 'tell application "Messages"', '-e',
                            'send (item 1 of argv) to participant (item 2 of argv) of (1st account whose service type = iMessage)',
                            '-e', 'end tell', '-e', 'end run', message, digits], capture_output=True)
    log(f'ALERT {message}')


def ssh(workspace, command):
    """Command output, or None if the workspace doesn't answer."""
    try:
        out = subprocess.run(['ssh', '-n', '-p', str(workspace.get('port', 2200)), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'StrictHostKeyChecking=accept-new',
                              f"{workspace['user']}@{workspace['host']}", command], capture_output=True, text=True, timeout=60)
        return out.stdout if out.returncode == 0 else None
    except subprocess.SubprocessError:
        return None


def finished_submissions(workspace):
    """{run id: sha256 of submission.csv} for runs with both submission.csv and summary.json, i.e. the prediction finished."""
    out = ssh(workspace, 'cd ~/badger-scribe-bucherwurm/runs/finetune 2>/dev/null && for d in submission-*; do '
                         'test -f "$d/submission.csv" && test -f "$d/summary.json" && sha256sum "$d/submission.csv"; done; echo END')
    if out is None:
        return None
    pairs = (line.split() for line in out.splitlines())
    return {p[1].split('/')[0]: p[0] for p in pairs if len(p) == 2 and re.fullmatch(r'submission-[\w-]+/submission\.csv', p[1])}


def local_hash(run_id):
    path = FINETUNE / run_id / 'submission.csv'
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def copy_submission(workspace, run_id, want):
    """Copy the run folder to runs/finetune/<id> on the Mac. True once the local submission.csv has hash `want`."""
    if local_hash(run_id) == want:
        return True
    if (FINETUNE / run_id).exists():
        log(f'{run_id}: a different local copy exists, not replaced')  # never overwrite without asking
        return False
    incoming = FINETUNE / f'.incoming-{run_id}'
    shutil.rmtree(incoming, ignore_errors=True)
    incoming.mkdir(parents=True)
    source = subprocess.Popen(['ssh', '-n', '-p', str(workspace.get('port', 2200)), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                               f"{workspace['user']}@{workspace['host']}", f'tar cf - -C ~/badger-scribe-bucherwurm/runs/finetune {run_id}'],
                              stdout=subprocess.PIPE)
    try:
        subprocess.run(['tar', 'xf', '-', '-C', str(incoming)], stdin=source.stdout, capture_output=True, timeout=300)
        source.wait(timeout=60)
    except subprocess.SubprocessError:
        source.kill()
    copied = incoming / run_id / 'submission.csv'
    ok = copied.exists() and hashlib.sha256(copied.read_bytes()).hexdigest() == want
    if ok:
        (incoming / run_id).rename(FINETUNE / run_id)
    shutil.rmtree(incoming, ignore_errors=True)
    log(f'{run_id}: copied to the Mac, hash matches' if ok else f'{run_id}: copy failed, retrying next check')
    return ok


def renew_workspaces(key, config, sent):
    """Rent two new workspaces with efabric.py (no AI), point every tool at them and restart both queues."""
    try:
        pending = json.loads(PENDING.read_text())['pending']
    except (OSError, ValueError, KeyError):
        pending = 0
    if pending == 0 or datetime.now(timezone.utc) >= RENEW_UNTIL:
        log(f'not renewing: {pending} unfinished queue lines, deadline {RENEW_UNTIL:%F}')
        sent.add(key)
        STATE.write_text(json.dumps({'sent': sorted(sent)}, indent=2) + '\n')
        return
    log(f'renewing: {pending} unfinished queue lines')
    out = subprocess.run([str(PM / '.venv/bin/python'), str(PM / 'bin/efabric.py'), 'renew'], capture_output=True, text=True, timeout=3600)
    lines = out.stdout.strip().splitlines()
    if out.returncode != 0 or not lines:
        reason = lines[-1] if lines else (out.stderr.strip().splitlines() or ['no output'])[-1]
        send_once(key, f'Workspace renewal needs you: {reason[:300]}', config, sent)
        return
    result = json.loads(lines[-1])  # {name: [user@host, group, evicted_in]}
    workspaces = []
    for index, name in enumerate(QUEUES):
        user, host = result[name][0].split('@')
        workspaces.append({'name': f'GPU {index + 1}', 'user': user, 'host': host, 'port': 2200})
    config['workspaces'] = workspaces
    CONFIG.write_text(json.dumps(config, indent=1) + '\n')  # the dashboard and this script read hosts from here
    started = []
    for workspace, name in zip(workspaces, QUEUES):
        hours, minutes = (int(x) if x else 0 for x in re.fullmatch(r'(?:(\d+)h)? ?(?:(\d+)m)?', result[name][2] or '24h').groups())
        cutoff = datetime.now(timezone.utc) + timedelta(hours=hours, minutes=minutes)
        queue = QUEUES[name]
        command = (f"tmux has-session -t q-{queue} 2>/dev/null || tmux new-session -d -s q-{queue} "
                   f"\"CUTOFF='{cutoff:%Y-%m-%d %H:%M}' bash ~/badger-scribe-bucherwurm/runs/pm/bin/queue.sh ~/badger-scribe-bucherwurm/runs/pm/queue-{queue}.txt\"; tmux ls")
        for _ in range(20):  # a new pod can take a few minutes to accept SSH
            if ssh(workspace, command) is not None:
                started.append(f"{workspace['name']} {result[name][1]} queue to {cutoff:%H:%M} UTC")
                break
            time.sleep(30)
        else:
            started.append(f"{workspace['name']} up but its queue did not start")
    send_once(key, 'Renewed EFabric workspaces: ' + '; '.join(started), config, sent)


def main():
    if PIDFILE.exists():
        try:
            os.kill(int(PIDFILE.read_text()), 0)
            sys.exit('workspace_alert.py is already running')
        except (OSError, ValueError):
            pass
    PIDFILE.write_text(str(os.getpid()))
    sent = load_state()
    starts, misses = {}, {}  # workspace user -> container start seen; consecutive failed checks
    log('started')
    last = time.time()
    while True:
        # time.sleep doesn't count Mac sleep, so a long wall-clock gap means the Mac slept: failed checks
        # from its short wakes (network not up yet) say nothing about the workspaces. False alarm 11-10.
        if time.time() - last > 2 * CHECK_SECONDS:
            misses.clear()
        last = time.time()
        config = json.loads(CONFIG.read_text())
        for workspace in config['workspaces']:
            user = workspace['user']
            start = ssh(workspace, 'TZ=UTC ps -o lstart= -p 1')
            if not start:
                misses[user] = misses.get(user, 0) + 1
                continue
            starts[user], misses[user] = start.strip(), 0
            count = ssh(workspace, PENDING_CMD)
            if count and count.strip().isdigit():  # the workspaces share one home, so either gives the same count
                PENDING.write_text(json.dumps({'pending': int(count), 'at': f'{datetime.now(timezone.utc):%FT%TZ}'}) + '\n')
            ready = finished_submissions(workspace)
            if ready is None:
                continue
            if not any(k.startswith('submission:') for k in sent):
                # First check ever: files that already exist count as seen, so only new ones pop up.
                sent.update({'submission:baseline'} | {f'submission:{r}' for r in ready})
                STATE.write_text(json.dumps({'sent': sorted(sent)}, indent=2) + '\n')
            for run_id, digest in sorted(ready.items()):  # the workspaces share one home, so the key stops a second pop-up
                if copy_submission(workspace, run_id, digest):  # retried every check until it lands
                    send_once(f'submission:{run_id}', f'{run_id}.csv is ready in runs/finetune/{run_id}.', config, sent)
                elif f'submission:{run_id}' not in sent:
                    send_once(f'copyfail:{run_id}', f'{run_id}.csv is ready on EFabric, but the copy to the Mac failed; retrying.', config, sent)
        users = [w['user'] for w in config['workspaces']]
        # One pop-up once every workspace has missed two checks in a row (about 6 minutes) after being seen up.
        if users and all(u in starts and misses.get(u, 0) >= 2 for u in users):
            key = 'stopped:' + '|'.join(f'{u}@{starts[u]}' for u in sorted(users))
            send_once(key, 'Your EFabric workspaces have stopped and are no longer rented.', config, sent)
            if 'renew:' + key not in sent:
                renew_workspaces('renew:' + key, config, sent)
        time.sleep(CHECK_SECONDS)


if __name__ == '__main__':
    main()
