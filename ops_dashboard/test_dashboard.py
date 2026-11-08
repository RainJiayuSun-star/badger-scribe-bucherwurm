import importlib.util
import json
import tempfile
import unittest
import unittest.mock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from telemetry import report, snapshot
from live_state import enrich, run_state

spec = importlib.util.spec_from_file_location('collector', Path(__file__).with_name('collect_remote.py'))
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CollectorTest(unittest.TestCase):
    def test_only_queue_metadata_is_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pm').mkdir()
            (root / 'pm/queue-gpu1.txt').write_text('# comment\ntraining-one 10 command\nqueued-two 10 command\n')
            (root / 'pm/queue-gpu1.txt.done').write_text('training-one rc=120 date\n')
            run = root / 'finetune/training-one'
            run.mkdir(parents=True)
            (run / 'status.json').write_text(json.dumps({'phase': 'failed', 'epoch': 2, 'text': 'DO NOT EXPORT'}))
            (run / 'events.jsonl').write_text(json.dumps({'event': 'evaluated', 'epoch': 1, 'cer': .2, 'text': 'DO NOT EXPORT'}) + '\n{partial')
            (run / 'predictions.csv').write_text('DO NOT EXPORT')
            rows = collector.collect(root, 'gpu1')
            self.assertEqual(rows[0]['rc'], 120)
            self.assertEqual(rows[0]['scores'], [{'epoch': 1, 'cer': .2}])
            self.assertEqual(rows[1]['status'], {})
            self.assertIsNone(rows[1]['rc'])
            self.assertNotIn('DO NOT EXPORT', json.dumps(rows))


class TaskEventsTest(unittest.TestCase):
    def test_real_transitions_and_expired_activity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.db'
            report('one', 'worker', 'Review', 'active', reported_at=100, path=path)
            self.assertEqual(snapshot(path, 200)['tasks'][0]['display_state'], 'active')
            self.assertEqual(snapshot(path, 401)['tasks'][0]['display_state'], 'unknown')
            report('one', 'worker', 'Review', 'completed', reported_at=402, path=path)
            rows = snapshot(path, 1000)['tasks']
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['display_state'], 'completed')

    def test_parallel_worker_reports_are_not_lost(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.db'
            report('root', 'root', 'Assign', 'completed', path=path)
            def emit(i):
                report(str(i), str(i), 'Check', 'active', path=path)
                report(str(i), str(i), 'Check', 'completed', path=path)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(emit, range(8)))
            tasks = snapshot(path)['tasks']
            self.assertEqual(len(tasks), 9)
            self.assertTrue(all(t['display_state'] == 'completed' for t in tasks))


class DerivedStateTest(unittest.TestCase):
    def test_offline_run_is_unknown_not_running(self):
        self.assertEqual(run_state({'rc': None, 'status': {'phase': 'training'}}, True), 'unknown')
        self.assertEqual(run_state({'rc': None, 'status': {'phase': 'training'}, 'process_alive': False}), 'unknown')
        self.assertEqual(run_state({'rc': 0, 'status': {'phase': 'training'}}, True), 'complete')
        self.assertEqual(run_state({'rc': 120, 'status': {'phase': 'training'}}), 'failed')

    def test_model_stage_comes_from_run_phase(self):
        gpus = [{'queue': 'gpu1', 'checked': '2026-11-06T10:00:00Z', 'runs': [
            {'id': 'ft10-kade', 'rc': None, 'status': {'phase': 'evaluating', 'updated': '2026-11-06T10:00:00Z'}}]}]
        from datetime import datetime
        now = datetime.fromisoformat('2026-11-06T10:00:00+00:00').timestamp()
        models = enrich(gpus, {'tasks': []}, now)
        self.assertEqual(models[0]['stage'], 'evaluation')
        self.assertEqual(models[0]['state'], 'active')
        self.assertEqual(models[-1]['stage'], 'unknown')

    def test_waiting_work_is_not_active_agent_work(self):
        models = enrich([], {'tasks': [{'model': 'weight-space', 'display_state': 'waiting',
                                       'stage': 'validation', 'title': 'Review pilot'}]}, 100)
        self.assertEqual(models[-1]['state'], 'waiting')
        self.assertEqual(models[-1]['note'], 'Review pilot')


def import_server():
    """server.py reads config.json at import; tests use a stand-in so they run without one."""
    import settings
    with unittest.mock.patch.object(settings, 'load', return_value={'workspaces': [], 'remote_runs': '~/runs', 'local_runs': 'runs'}):
        import server
    return server


class KaggleTest(unittest.TestCase):
    def test_kaggle_uploads_mark_submissions_and_bring_their_scores(self):
        server = import_server()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'ops_dashboard'
            root.mkdir()
            (root / 'history_notes.json').write_text(json.dumps({'public_scores': {'submission-01': 0.143}}))
            for sid in ('submission-01', 'submission-02', 'submission-03'):
                (Path(tmp) / 'runs' / 'finetune' / sid).mkdir(parents=True)
                (Path(tmp) / 'runs' / 'finetune' / sid / 'submission.csv').write_text('page_id,text\n')
            kaggle = [{'date': '2026-11-03', 'status': 'complete', 'publicscore': '0.143'},
                      {'date': '2026-11-05', 'status': 'complete', 'publicscore': '0.150'},
                      {'date': '2026-11-06', 'status': 'pending', 'publicscore': ''}]
            with unittest.mock.patch.object(server, 'ROOT', root):
                rows = {s['id']: s for s in server.submission_status([], kaggle)}
                self.assertTrue(all(s['uploaded'] for s in rows.values()))
                self.assertEqual(rows['submission-02']['public_score'], 0.150)
                self.assertIn('+0.007, worse', rows['submission-02']['evaluation'])
                self.assertIsNone(rows['submission-03']['public_score'])
                self.assertEqual(rows['submission-03']['kaggle_status'], 'pending')
                # Without Kaggle, only the notes count as uploaded.
                rows = {s['id']: s for s in server.submission_status([], None)}
                self.assertEqual([sid for sid, s in rows.items() if s['uploaded']], ['submission-01'])

    def test_training_payload_keeps_curves_and_submissions_only(self):
        server = import_server()
        runs = [{'id': 'ft13-all', 'config': {'model_id': 'allenai/olmOCR-2-7B-1025', 'epochs': 3}, 'summary': {}, 'fold': 'all_train',
                 'loss': [[1, 1, 0.5], [135, 3, 0.1]], 'epochs': [], 'adapters': None, 'started': '2026-11-06T22:01'},
                {'id': 'ft6-g1', 'config': {'model_id': 'stanford-oval/churro-3B'}, 'summary': {}, 'fold': 'dominy_group1', 'loss': [],
                 'epochs': [{'epoch': 1, 'val_macro_cer': 0.33}, {'epoch': 2, 'val_macro_cer': None}]},
                {'id': 'dec-x', 'config': {}, 'summary': {'cer': 0.2}, 'epochs': []},
                {'id': 'submission-04', 'config': {}, 'summary': {}, 'epochs': [], 'adapters': {'dominy_accounts': 'ft13-all'}}]
        with tempfile.TemporaryDirectory() as tmp:
            cache, root = Path(tmp) / 'runs.json', Path(tmp)
            cache.write_text(json.dumps(runs))
            (root / 'history_notes.json').write_text(json.dumps({'public_scores': {'submission-04': 0.129}}))
            with unittest.mock.patch.object(server.history, 'CACHE', cache), unittest.mock.patch.object(server, 'ROOT', root):
                payload = server.training_payload()
        rows = {r['id']: r for r in payload['runs']}
        self.assertEqual(set(rows), {'ft13-all', 'ft6-g1', 'submission-04'})
        self.assertEqual(rows['ft13-all']['model'], 'olmOCR-2-7B')
        self.assertEqual(rows['ft13-all']['category'], None, 'an all-data run holds out nothing')
        self.assertEqual(rows['ft6-g1']['category'], 'dominy_accounts')
        self.assertEqual(rows['ft6-g1']['val'], [[1, 0.33]], 'an epoch with no score is dropped')
        self.assertEqual(payload['public'], {'submission-04': 0.129})


if __name__ == '__main__':
    unittest.main()
