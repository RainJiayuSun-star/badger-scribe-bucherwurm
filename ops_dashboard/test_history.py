"""python3 ops_dashboard/test_history.py"""
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from collect_history import collect  # noqa: E402
from history import build, describe  # noqa: E402


class HistoryTests(unittest.TestCase):
    def run_folder(self, root, name, config, metrics=(), events=(), extra=None):
        folder = root / name
        folder.mkdir()
        (folder / 'config.json').write_text(json.dumps(config))
        (folder / 'metrics.jsonl').write_text(''.join(json.dumps(m) + '\n' for m in metrics))
        (folder / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
        for path, text in (extra or {}).items():
            (folder / path).write_text(text)

    def test_no_text_leaves_and_best_epoch_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.run_folder(root, 'ft10-lr2e4-kade103',
                            {'model_id': 'stanford-oval/churro-3B', 'learning_rate': 2e-4, 'epochs': 3, 'lora_rank': 16,
                             'fold': 'runs/folds/kade_103.json', 'prompt': 'Transcribe this page exactly'},
                            metrics=[{'epoch': 1, 'val_macro_cer': 0.09, 'val_cer_kade_letters': 0.09},
                                     {'epoch': 2, 'val_macro_cer': 0.07, 'val_cer_kade_letters': 0.07},
                                     {'epoch': 3, 'val_macro_cer': 0.08, 'val_cer_kade_letters': 0.08}],
                            events=[{'time': '2026-11-05T22:00:00+00:00', 'event': 'training'},
                                    {'time': '2026-11-05T23:00:00+00:00', 'event': 'finished'}],
                            extra={'predictions.csv': 'page_id,text\nkade_103_p001,Liebe Mutter SECRET\n'})
            runs = collect(root)
            self.assertNotIn('SECRET', json.dumps(runs))
            self.assertNotIn('Transcribe', json.dumps(runs))
            record = describe(runs[0], {})
            self.assertEqual(record['best'], 0.07)
            self.assertIn('learning rate 0.0002', record['approach'])
            self.assertEqual(record['metrics'][0][1], {'kade_letters': 0.07})
            self.assertEqual(record['data'], 'German letters')
            self.assertIn('ft10-lr2e4-kade103', build(runs, {}))

    def test_crashed_run_falls_back_to_logged_evals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.run_folder(root, 'ft10-base-kade128', {'fold': 'kade_128.json'},
                            events=[{'time': '2026-11-05T23:00:00+00:00', 'event': 'evaluated', 'epoch': 1, 'cer': 0.08},
                                    {'time': '2026-11-05T23:59:00+00:00', 'event': 'failed', 'error': 'OSError: [Errno 9] Bad file descriptor'}])
            record = describe(collect(root)[0], {})
            self.assertEqual((record['outcome'], record['error'], record['best']), ('failed', 'OSError', 0.08))


if __name__ == '__main__':
    unittest.main()
