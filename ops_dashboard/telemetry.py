"""Token-free task-event store. Workers report transitions; timestamps expire activity."""
import argparse
import json
import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager

DB = Path(__file__).with_name('telemetry.sqlite3')
STATES = ('waiting', 'active', 'completed', 'blocked', 'cancelled')
MODELS = ('operations', 'churro', 'olmocr', 'lighton', 'weight-space')
STAGES = ('development', 'validation', 'fine-tuning', 'evaluation', 'release')
ACTIVE_TTL = 300


@contextmanager
def connect(path=DB):
    db = sqlite3.connect(str(path), timeout=10)
    db.row_factory = sqlite3.Row
    try:
        with db:
            db.execute('''CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY, task TEXT NOT NULL, worker TEXT NOT NULL,
                title TEXT NOT NULL, state TEXT NOT NULL, model TEXT NOT NULL,
                stage TEXT NOT NULL, note TEXT NOT NULL, source TEXT NOT NULL,
                reported_at REAL NOT NULL)''')
            yield db
    finally:
        db.close()


def report(task, worker, title, state, model='operations', stage='development',
           note='', source='worker report', reported_at=None, path=DB):
    if state not in STATES or model not in MODELS or stage not in STAGES:
        raise ValueError('Unsupported task state, model or stage')
    if not task or not worker or not title:
        raise ValueError('Task, worker and title are required')
    with connect(path) as db:
        db.execute('INSERT INTO events(task,worker,title,state,model,stage,note,source,reported_at) VALUES (?,?,?,?,?,?,?,?,?)',
                   (task, worker, title, state, model, stage, note, source,
                    time.time() if reported_at is None else reported_at))


def snapshot(path=DB, now=None):
    now = time.time() if now is None else now
    with connect(path) as db:
        rows = [dict(row) for row in db.execute('''SELECT * FROM events WHERE seq IN
            (SELECT MAX(seq) FROM events GROUP BY task) ORDER BY worker, seq''')]
    for row in rows:
        row['age_seconds'] = max(0, now - row['reported_at'])
        row['display_state'] = 'unknown' if row['state'] == 'active' and row['age_seconds'] > ACTIVE_TTL else row['state']
    return {'tasks': rows, 'active_ttl_seconds': ACTIVE_TTL,
            'source': 'Task transition events; activity expires after five minutes without a worker report.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--state', choices=STATES, required=True)
    parser.add_argument('--model', choices=MODELS, default='operations')
    parser.add_argument('--stage', choices=STAGES, default='development')
    parser.add_argument('--note', default='')
    args = vars(parser.parse_args())
    report(**args)
    print(json.dumps({'recorded': args['task'], 'state': args['state']}))


if __name__ == '__main__':
    main()
