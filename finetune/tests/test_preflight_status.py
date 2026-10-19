from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from finetune.preflight import check_data, check_disk, check_gpu, check_run_dir, run_checks
from finetune.status import summarize, verdict

GIB = 1024**3


def _data_root(tmp: Path) -> Path:
    root = tmp / "data"
    (root / "images").mkdir(parents=True)
    (root / "train.csv").write_text("page_id\n")
    (root / "metric.py").write_text("")
    return root


class PreflightTests(unittest.TestCase):
    def test_missing_data_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            problems = check_data(Path(tmp))
        self.assertEqual(len(problems), 3)
        self.assertTrue(any("train.csv" in p for p in problems))

    def test_disk_check_handles_a_directory_that_does_not_exist_yet(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(check_disk({"run": Path(tmp) / "a" / "b"}, 0.001), [])
            self.assertEqual(len(check_disk({"run": Path(tmp)}, 10**9)), 1)

    def test_busy_gpu_is_refused(self):
        self.assertEqual(len(check_gpu(12, lambda: (4 * GIB, 24 * GIB))), 1)
        self.assertEqual(check_gpu(12, lambda: (20 * GIB, 24 * GIB)), [])
        self.assertEqual(check_gpu(0, lambda: (0, 24 * GIB)), [])

    def test_existing_checkpoints_need_resume_or_a_new_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "checkpoint-5").mkdir()
            (run_dir / "checkpoint-5" / "state.json").write_text("{}")
            self.assertEqual(len(check_run_dir(run_dir, run_dir, resume=False, eval_only=False)), 1)
            self.assertEqual(check_run_dir(run_dir, run_dir, resume=True, eval_only=False), [])

    def test_a_live_run_with_the_same_id_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            status = {"phase": "training", "host": socket.gethostname(), "pid": os.getpid()}
            (run_dir / "status.json").write_text(json.dumps(status))
            self.assertEqual(len(check_run_dir(run_dir, run_dir, resume=True, eval_only=False)), 1)
            status["phase"] = "finished"
            (run_dir / "status.json").write_text(json.dumps(status))
            self.assertEqual(check_run_dir(run_dir, run_dir, resume=True, eval_only=False), [])

    def test_all_problems_are_reported_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"data_root": str(Path(tmp) / "nowhere"), "min_free_disk_gib": 10**9, "min_free_gpu_gib": 12}
            with self.assertRaises(SystemExit) as stopped:
                run_checks(config, Path(tmp) / "run", Path(tmp) / "run", False, False, lambda: (1 * GIB, 24 * GIB))
        message = str(stopped.exception.code)
        self.assertIn("nothing was started", message)
        self.assertIn("train.csv", message)
        self.assertIn("GPU memory", message)
        self.assertIn("GiB free", message)

    def test_good_setup_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"data_root": str(_data_root(Path(tmp))), "min_free_disk_gib": 0.001, "min_free_gpu_gib": 12}
            run_checks(config, Path(tmp) / "run", Path(tmp) / "run", False, False, lambda: (20 * GIB, 24 * GIB))


class StatusTests(unittest.TestCase):
    now = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)

    def _status(self, phase, minutes_ago, pid=999999999):
        return {"phase": phase, "host": "elsewhere", "pid": pid,
                "updated": (self.now - timedelta(minutes=minutes_ago)).isoformat()}

    def test_verdicts(self):
        self.assertEqual(verdict({"phase": "finished"}, [], self.now), "finished")
        self.assertIn("Python error", verdict({"phase": "failed"}, [{"event": "failed"}], self.now))
        self.assertIn("SIGTERM", verdict(self._status("stopping", 1), [{"event": "signal", "signal": "SIGTERM"}], self.now))
        self.assertEqual(verdict(self._status("training", 1), [{"event": "training"}], self.now), "running: training")
        self.assertIn("no progress", verdict(self._status("evaluating", 60), [{"event": "evaluating"}], self.now))

    def test_dead_process_on_this_machine_is_reported(self):
        status = {"phase": "training", "host": socket.gethostname(), "pid": 999999999,
                  "updated": datetime.now(timezone.utc).isoformat()}
        self.assertIn("died without warning", verdict(status, [{"event": "training"}]))

    def test_summary_collects_scores_and_skips_a_cut_off_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run-a"
            run_dir.mkdir()
            (run_dir / "status.json").write_text(json.dumps({"phase": "finished", "epoch": 2}))
            (run_dir / "events.jsonl").write_text(
                '{"time": "t", "event": "evaluated", "epoch": 1, "cer": 0.2}\n'
                '{"time": "t", "event": "evaluated", "epoch": 2, "cer": 0.1}\n'
                '{"time": "t", "event": "fini')
            summary = summarize(run_dir)
        self.assertEqual(summary["scores"], [(1, 0.2), (2, 0.1)])
        self.assertEqual(summary["verdict"], "finished")


if __name__ == "__main__":
    unittest.main()
