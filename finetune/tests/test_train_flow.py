"""The whole training loop, wired together, on a tiny model and CPU.

Unit tests call each piece alone, so they can't catch a piece wired in wrong.
These run finetune.train.main with only the model, batches, decoding and metric
faked: phases, scoring, best_adapter, checkpoints, preflight, shutdown and resume
are all real.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import signal
import sys
import tempfile
import unittest
from itertools import chain, repeat
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HAS_STACK = all(importlib.util.find_spec(name) for name in ("torch", "accelerate", "transformers", "safetensors", "yaml", "PIL"))


@unittest.skipUnless(HAS_STACK, "needs the fine-tuning dependencies")
class TrainingFlowTests(unittest.TestCase):
    def setUp(self):
        # Import before torch.cuda.is_available is patched below.
        import accelerate  # noqa: F401
        import torch
        import transformers  # noqa: F401
        from PIL import Image

        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        data = root / "data"
        (data / "images").mkdir(parents=True)
        (data / "metric.py").write_text("")
        pages = [(f"kade_00{doc}_p00{n}", f"kade_00{doc}") for doc in (1, 2) for n in (1, 2, 3)]
        with (data / "train.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, ["page_id", "doc_id", "text", "category", "label_source"])
            writer.writeheader()
            for page_id, doc_id in pages:
                writer.writerow({"page_id": page_id, "doc_id": doc_id, "text": "Liebe Mutter", "category": "kade_letters", "label_source": "human"})
                Image.new("RGB", (8, 8)).save(data / "images" / f"{page_id}.jpg")
        self.fold = root / "fold.json"
        self.fold.write_text(json.dumps({"name": "fold", "eval_ids": [p for p, d in pages if d == "kade_001"], "train_ids": [p for p, d in pages if d == "kade_002"]}))
        self.config = root / "config.yaml"
        self.config.write_text(f"model_id: fake\nquantization: none\nepochs: 2\neffective_batch_size: 2\nhuman_repeats: 2\n"
                               f"data_root: {data}\nruns_root: {root / 'runs'}\nmin_free_disk_gib: 0\nmin_free_gpu_gib: 0\n")
        self.run_dir = root / "runs" / "flow"
        self.cers = [0.2, 0.3]     # epoch 2 scores worse, so epoch 1 stays the best
        self.evals = self.saves = self.batches = 0
        self.signal_at_batch = None
        self.addCleanup(signal.signal, signal.SIGTERM, signal.getsignal(signal.SIGTERM))
        self.addCleanup(signal.signal, signal.SIGHUP, signal.getsignal(signal.SIGHUP))
        self.addCleanup(signal.signal, signal.SIGINT, signal.getsignal(signal.SIGINT))

    def _model(self, *_):
        torch, test = self.torch, self

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 1)

            def forward(self, x):
                return SimpleNamespace(loss=self.layer(x).pow(2).mean())

            def save_pretrained(self, path):
                test.saves += 1
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "adapter.txt").write_text(str(test.saves))

        return Tiny(), SimpleNamespace(save_pretrained=lambda path: Path(path).mkdir(parents=True, exist_ok=True))

    def _collate(self, rows, processor, device, epoch, augmenter, config):
        self.batches += 1
        if self.batches == self.signal_at_batch:
            signal.raise_signal(signal.SIGTERM)
        return {"x": self.torch.ones(1, 2, device=device)}

    def _generate(self, model, processor, rows, device, max_new_tokens, config, on_page):
        self.evals += 1
        for done in range(1, len(rows) + 1):
            on_page(done, len(rows))
        return ["text"] * len(rows), [{"loop_cut": 0, "retry_used": 0}] * len(rows)

    def _train(self, *extra):
        from accelerate.state import AcceleratorState

        AcceleratorState._reset_state(True)
        metric = SimpleNamespace(page_cer=lambda pred, ref: self.cers[min(self.evals, len(self.cers)) - 1], page_wer=lambda pred, ref: 0.0)
        argv = ["train", "--config", str(self.config), "--fold", str(self.fold), "--run-id", "flow", "--no-wandb", *extra]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {"ACCELERATE_USE_CPU": "true"}), \
                mock.patch("torch.cuda.is_available", side_effect=chain([True], repeat(False))), \
                mock.patch("finetune.train.load_model_and_processor", self._model), \
                mock.patch("finetune.train.collate_train", self._collate), \
                mock.patch("finetune.train.generate_texts", self._generate), \
                mock.patch("finetune.train.load_metric", return_value=metric), \
                mock.patch("finetune.train.version_info", return_value={}), \
                mock.patch("finetune.train.MetricWriter"):
            from finetune.train import main

            return main()

    def _events(self):
        return [json.loads(line) for line in (self.run_dir / "events.jsonl").read_text().splitlines()]

    def _kinds(self):
        return [event["event"] for event in self._events()]

    def test_full_run_keeps_the_best_epoch_and_later_runs_cannot_overwrite_it(self):
        self.assertEqual(self._train(), 0)
        self.assertEqual(self._kinds()[-1], "finished")
        self.assertEqual((self.run_dir / "best_adapter" / "adapter.txt").read_text(), "1", "epoch 2 scored worse")
        self.assertAlmostEqual(json.loads((self.run_dir / "status.json").read_text())["best_cer"], 0.2)
        evaluated = [event for event in self._events() if event["event"] == "evaluated"]
        self.assertAlmostEqual(evaluated[0]["human_cer"], 0.2)
        self.assertAlmostEqual(evaluated[1]["human_cer"], 0.3)
        self.assertEqual(sorted(p.name for p in self.run_dir.glob("checkpoint-*")), ["checkpoint-3", "checkpoint-6"])

        with self.assertRaisesRegex(SystemExit, "already has checkpoints"):
            self._train()
        for path in self.run_dir.glob("checkpoint-*"):
            shutil.rmtree(path)
        for extra in ((), ("--resume",)):
            with self.assertRaisesRegex(SystemExit, "best_adapter"):
                self._train(*extra)
        self.assertEqual((self.run_dir / "best_adapter" / "adapter.txt").read_text(), "1")

    def test_a_shutdown_is_not_a_crash_and_resume_finishes_the_run(self):
        from finetune.status import verdict

        self.signal_at_batch = 8    # 3 human pages x 2 repeats = 6 batches an epoch, so this is partway into epoch 2
        with self.assertRaises(SystemExit) as stopped:
            self._train()
        self.assertEqual(stopped.exception.code, 128 + signal.SIGTERM)
        self.assertEqual(self._kinds()[-1], "signal")
        self.assertNotIn("failed", self._kinds())
        status = json.loads((self.run_dir / "status.json").read_text())
        self.assertIn("stopped from outside", verdict(status, self._events()))

        self.signal_at_batch = None
        self.assertEqual(self._train("--resume"), 0)
        self.assertEqual(self._kinds()[-1], "finished")
        self.assertIn("resumed", self._kinds())
        self.assertEqual(self.evals, 2, "epoch 1 was not scored again")
        self.assertEqual((self.run_dir / "best_adapter" / "adapter.txt").read_text(), "1", "worse epoch 2 did not replace it")

    def test_write_errors_skip_the_save_and_training_carries_on(self):
        with mock.patch("finetune.train.save_checkpoint", side_effect=OSError("Disk quota exceeded")), \
                mock.patch("finetune.train.save_best_adapter", side_effect=OSError("Disk quota exceeded")):
            self.assertEqual(self._train(), 0)
        kinds = self._kinds()
        self.assertEqual(kinds[-1], "finished")
        self.assertIn("best_adapter_skipped", kinds)
        self.assertEqual([e["reason"] for e in self._events() if e["event"] == "checkpoint_skipped"], ["write_error", "write_error"])


if __name__ == "__main__":
    unittest.main()
