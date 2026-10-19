from __future__ import annotations

import importlib.util
import json
import signal
import tempfile
import unittest
from pathlib import Path

from finetune.checkpoints import checkpoint_dir, checkpoints_by_step, latest_checkpoint, prune_checkpoints
from finetune.runstate import Heartbeat, RunState

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None


def _fake_checkpoint(root: Path, step: int) -> None:
    (root / f"checkpoint-{step}").mkdir()
    (root / f"checkpoint-{step}" / "state.json").write_text("{}")


class FifoTests(unittest.TestCase):
    def test_prune_keeps_the_newest_by_step_not_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in (9, 22, 100):
                _fake_checkpoint(root, step)
            (root / "best_adapter").mkdir()
            removed = prune_checkpoints(root, keep=2)
            self.assertEqual([p.name for p in removed], ["checkpoint-9"])
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["best_adapter", "checkpoint-100", "checkpoint-22"])
            self.assertEqual(latest_checkpoint(root).name, "checkpoint-100")
            self.assertEqual(prune_checkpoints(root, keep=0), [])

    def test_partial_and_unfinished_checkpoints_are_never_counted_or_pruned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fake_checkpoint(root, 10)
            (root / "partial-checkpoint-20").mkdir()       # killed mid-save
            (root / "checkpoint-30").mkdir()               # no state.json: incomplete
            self.assertEqual([p.name for p in checkpoints_by_step(root)], ["checkpoint-10"])
            self.assertEqual(prune_checkpoints(root, keep=1), [])
            self.assertEqual(latest_checkpoint(root).name, "checkpoint-10")

    def test_checkpoint_root_is_separate_from_the_run_directory(self):
        run_dir = Path("/home/runs/abc")
        self.assertEqual(checkpoint_dir({"run_id": "abc"}, run_dir), run_dir)
        self.assertEqual(checkpoint_dir({"run_id": "abc", "checkpoint_root": "/tmp/ck"}, run_dir), Path("/tmp/ck/abc"))


@unittest.skipUnless(HAS_TORCH, "torch and safetensors are installed by the cuda extra")
class RoundTripTests(unittest.TestCase):
    def _setup(self):
        import torch

        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
        model[0].weight.requires_grad_(False)       # a frozen "base" weight
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return model, optimizer, scheduler

    def _step(self, model, optimizer, scheduler):
        import torch

        loss = model(torch.ones(3, 4)).pow(2).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    def test_resume_continues_exactly_where_training_stopped(self):
        import torch

        from finetune.checkpoints import load_checkpoint, save_checkpoint

        torch.manual_seed(0)
        model, optimizer, scheduler = self._setup()
        self._step(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as tmp:
            path, size = save_checkpoint(Path(tmp), 1, model, optimizer, scheduler, {"global_step": 1, "best_cer": 0.2})
            self.assertEqual(path.name, "checkpoint-1")
            self.assertFalse((Path(tmp) / "partial-checkpoint-1").exists())
            self.assertGreater(size, 0)
            self._step(model, optimizer, scheduler)
            expected = [p.detach().clone() for p in model.parameters()]

            torch.manual_seed(123)
            fresh, fresh_optimizer, fresh_scheduler = self._setup()
            with torch.no_grad():
                fresh[0].weight.copy_(model[0].weight)      # the frozen base comes from the pretrained model
            saved = load_checkpoint(path, fresh, fresh_optimizer, fresh_scheduler)
            self.assertEqual(saved["best_cer"], 0.2)
            self._step(fresh, fresh_optimizer, fresh_scheduler)
        for got, want in zip(fresh.parameters(), expected):
            self.assertTrue(torch.allclose(got, want))

    def test_only_trainable_tensors_are_saved(self):
        from safetensors.torch import load_file

        from finetune.checkpoints import save_checkpoint

        model, optimizer, scheduler = self._setup()
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = save_checkpoint(Path(tmp), 5, model, optimizer, scheduler, {})
            self.assertEqual(sorted(load_file(str(path / "trainable.safetensors"))), ["0.bias", "1.bias", "1.weight"])


class RunStateTests(unittest.TestCase):
    def test_events_and_status_record_where_the_run_was(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp))
            state.phase("training", epoch=2, step=40)
            Heartbeat(state, seconds=0)(step=41)
            try:
                raise RuntimeError("CUDA out of memory")
            except RuntimeError as error:
                state.failed(error)
            status = json.loads((Path(tmp) / "status.json").read_text())
            events = [json.loads(line) for line in (Path(tmp) / "events.jsonl").read_text().splitlines()]
        self.assertEqual(status["phase"], "failed")
        self.assertEqual(status["step"], 41)
        self.assertEqual(status["epoch"], 2)
        self.assertIn("disk_free_gib_run_dir", status["resources"])
        self.assertEqual([e["event"] for e in events], ["training", "failed"])
        self.assertIn("CUDA out of memory", events[-1]["traceback"])

    def test_sigterm_is_logged_before_the_run_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp))
            previous = signal.getsignal(signal.SIGTERM)
            try:
                state.catch_signals()
                with self.assertRaises(SystemExit) as stopped:
                    signal.raise_signal(signal.SIGTERM)
            finally:
                signal.signal(signal.SIGTERM, previous)
                signal.signal(signal.SIGHUP, signal.SIG_DFL)
                signal.signal(signal.SIGINT, signal.default_int_handler)
            events = [json.loads(line) for line in (Path(tmp) / "events.jsonl").read_text().splitlines()]
        self.assertEqual(stopped.exception.code, 128 + signal.SIGTERM)
        self.assertEqual(events[-1]["event"], "signal")
        self.assertEqual(events[-1]["signal"], "SIGTERM")

    def test_disabled_state_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "run", enabled=False)
            state.phase("training")
            self.assertFalse((Path(tmp) / "run").exists())


if __name__ == "__main__":
    unittest.main()
