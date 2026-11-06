import importlib.util
import json
import tempfile
import unittest
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


if __name__ == '__main__':
    unittest.main()
