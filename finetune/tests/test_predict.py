"""finetune.predict end to end, with the model and decoding faked and no real images."""

from __future__ import annotations

import csv
import importlib.util
import json
import signal
import sys
import tempfile
import unittest
from itertools import chain, repeat
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HAS_STACK = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "yaml", "PIL"))

# (page_id, category, label_source), in sample_submission order; test.csv lists them reversed.
PAGES = [("kade_009_p001", "kade_letters", "human"), ("survey_005_p001", "survey_notes", "silver_claude"),
         ("kade_009_p002", "kade_letters", "human"), ("dominy_003_p002", "dominy_accounts", "silver_claude")]
PAGE_KEYS = {"page_id", "doc_id", "category", "image_path"}


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@unittest.skipUnless(HAS_STACK, "needs the fine-tuning dependencies")
class PredictTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = root = Path(self.tmp.name)
        data = root / "data"
        (data / "images").mkdir(parents=True)
        (data / "metric.py").write_text("")
        write_csv(data / "sample_submission.csv", ["page_id", "text"], [{"page_id": p, "text": ""} for p, _, _ in PAGES])
        write_csv(data / "test.csv", ["page_id", "doc_id", "category", "label_source"],
                  [{"page_id": p, "doc_id": p[:-5], "category": c, "label_source": s} for p, c, s in reversed(PAGES)])
        train = [(f"kade_00{doc}_p00{n}", f"kade_00{doc}") for doc in (1, 2) for n in (1, 2)]
        write_csv(data / "train.csv", ["page_id", "doc_id", "text", "category", "label_source"],
                  [{"page_id": p, "doc_id": d, "text": "Liebe Mutter", "category": "kade_letters", "label_source": "human"} for p, d in train])
        for page_id in [p for p, _, _ in PAGES] + [p for p, _ in train]:
            (data / "images" / f"{page_id}.jpg").touch()
        self.fold = root / "fold.json"
        self.fold.write_text(json.dumps({"name": "fold", "eval_ids": [p for p, d in train if d == "kade_001"], "train_ids": [p for p, d in train if d == "kade_002"]}))
        self.config = root / "config.yaml"
        self.config.write_text(f"model_id: fake\nquantization: none\neval_max_new_tokens: 4096\n"
                               f"data_root: {data}\nruns_root: {root / 'runs'}\nmin_free_disk_gib: 0\nmin_free_gpu_gib: 0\n")
        self.run_dir = root / "runs" / "sub"
        self.german, self.english = root / "german", root / "english"
        for adapter in (self.german, self.english):
            adapter.mkdir()
            (adapter / "adapter_config.json").touch()
            (adapter / "adapter_model.safetensors").touch()
        self.all_adapters = [f"kade_letters={self.german}", f"survey_notes={self.english}", f"dominy_accounts={self.english}"]
        self.model = None
        self.model_loads = 0
        self.decoded = []
        self.crash_on_decode = None
        self.texts = {}
        self.missing_keys = []
        for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            self.addCleanup(signal.signal, number, signal.getsignal(number))

    def _load(self, config, local_rank):
        test = self
        self.model_loads += 1

        class FakeModel:
            training = True
            active = "default"
            loaded = {}

            def load_adapter(self, path, adapter_name, is_trainable):
                test.assertFalse(is_trainable)
                self.loaded = {**self.loaded, adapter_name: path}
                self.training = True    # a new adapter's dropout starts on, as in PEFT
                return SimpleNamespace(missing_keys=test.missing_keys)

            def set_adapter(self, name):
                self.active = name

            def eval(self):
                self.training = False

        self.model = FakeModel()
        return self.model, SimpleNamespace()

    def _generate(self, model, processor, rows, device, max_new_tokens, config):
        self.assertEqual(len(rows), 1)
        self.assertEqual(max_new_tokens, 4096)
        if len(self.decoded) == self.crash_on_decode:
            raise RuntimeError("CUDA out of memory")
        page = rows[0]
        self.decoded.append({"page_id": page["page_id"], "adapter": model.loaded[model.active], "active": model.active, "training": model.training, "keys": set(page)})
        return [self.texts.get(page["page_id"], f"text of {page['page_id']}")], [{"loop_cut": 1, "retry_used": 0}]

    def _predict(self, *extra, adapters=None, load=None, generate=None):
        argv = ["predict", "--config", str(self.config), "--run-id", "sub"]
        for adapter in self.all_adapters if adapters is None else adapters:
            argv += ["--adapter", adapter]
        metric = SimpleNamespace(page_cer=lambda pred, ref: 0.25)
        with mock.patch.object(sys, "argv", [*argv, *extra]), \
                mock.patch("torch.cuda.is_available", side_effect=chain([True], repeat(False))), \
                mock.patch("torch.cuda.max_memory_reserved", return_value=0), \
                mock.patch("finetune.predict.load_model_and_processor", load or self._load), \
                mock.patch("finetune.predict.generate_texts", generate or self._generate), \
                mock.patch("finetune.predict.load_metric", return_value=metric):
            from finetune.predict import main

            return main()

    def _events(self):
        return [json.loads(line) for line in (self.run_dir / "events.jsonl").read_text().splitlines()]

    def _summary(self):
        return json.loads((self.run_dir / "summary.json").read_text())

    def test_each_page_uses_its_categorys_adapter_with_dropout_off(self):
        self.assertEqual(self._predict(), 0)
        self.assertEqual([(d["page_id"], d["active"]) for d in self.decoded], [(p, c) for p, c, _ in PAGES])
        expected = {"kade_letters": self.german, "survey_notes": self.english, "dominy_accounts": self.english}
        for decoded in self.decoded:
            self.assertEqual(decoded["adapter"], str(expected[decoded["active"]].resolve()))
            self.assertFalse(decoded["training"], "dropout was still on while decoding")

    def test_submission_follows_sample_order_and_the_summary_fills_the_card(self):
        self._predict()
        rows = read_csv(self.run_dir / "submission.csv")
        self.assertEqual([row["page_id"] for row in rows], [p for p, _, _ in PAGES])
        self.assertEqual(rows[0]["text"], "text of kade_009_p001")
        summary = self._summary()
        self.assertEqual((summary["pages"], summary["empty_pages"], summary["loop_cuts"], summary["model_id"]), (4, [], 4, "fake"))
        self.assertEqual(summary["adapters"]["survey_notes"], str(self.english.resolve()))
        self.assertIn("wall_seconds", summary)
        self.assertIn("peak_gpu_gib", summary)
        self.assertIn("git_version", summary)
        self.assertEqual(len((self.run_dir / "pages.jsonl").read_text().splitlines()), 4)
        self.assertEqual(self._events()[-1]["event"], "finished")

    def test_refuses_a_category_with_no_adapter_before_loading_the_model(self):
        with self.assertRaisesRegex(SystemExit, "No --adapter for dominy_accounts, survey_notes"):
            self._predict(adapters=[f"kade_letters={self.german}"])
        self.assertEqual(self.model_loads, 0)
        self.assertFalse(self.run_dir.exists())

    def test_refuses_bad_adapter_arguments_and_folders(self):
        for bad, message in ((f"kade_letter={self.german}", "must be CATEGORY=PATH"), ("kade_letters=", "must be CATEGORY=PATH"),
                             (f"kade_letters={self.root}", "adapter_config.json")):
            with self.assertRaisesRegex(SystemExit, message):
                self._predict(adapters=[bad])
        with self.assertRaisesRegex(SystemExit, "twice"):
            self._predict(adapters=[f"kade_letters={self.german}", f"kade_letters={self.english}"])
        self.assertEqual(self.model_loads, 0)

    def test_refuses_when_the_test_files_are_missing_or_disagree(self):
        data = self.root / "data"
        write_csv(data / "sample_submission.csv", ["page_id", "text"], [{"page_id": "kade_009_p001", "text": ""}])
        with self.assertRaisesRegex(SystemExit, "don't match"):
            self._predict()
        (data / "test.csv").unlink()
        with self.assertRaisesRegex(SystemExit, "test.csv is missing"):
            self._predict()

    def test_an_adapter_with_missing_tensors_stops_the_run(self):
        self.missing_keys = ["base_model.layer.lora_A.kade_letters.weight"]
        with self.assertRaisesRegex(SystemExit, "missing 1 tensors"):
            self._predict()
        self.assertEqual(self.decoded, [])
        self.assertEqual(self._events()[-1]["event"], "failed")

    def test_the_model_never_sees_label_source(self):
        self._predict()
        self.assertEqual(len(self.decoded), 4)
        self.assertTrue(all(d["keys"] == PAGE_KEYS for d in self.decoded))

    def test_a_crash_keeps_finished_pages_and_a_rerun_skips_them(self):
        self.crash_on_decode = 2
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self._predict()
        self.assertEqual(self._events()[-1]["event"], "failed")
        self.assertEqual([row["page_id"] for row in read_csv(self.run_dir / "predictions.csv")], [p for p, _, _ in PAGES[:2]])
        self.assertFalse((self.run_dir / "submission.csv").exists(), "a partial file must not look submittable")

        self.crash_on_decode = None
        self.decoded.clear()
        self.assertEqual(self._predict(), 0)
        self.assertEqual([d["page_id"] for d in self.decoded], [p for p, _, _ in PAGES[2:]])
        self.assertEqual([row["page_id"] for row in read_csv(self.run_dir / "submission.csv")], [p for p, _, _ in PAGES])
        self.assertEqual(self._summary()["loop_cuts"], 4, "pages from both processes are counted")

    def test_a_rerun_of_a_finished_run_does_not_load_the_model(self):
        self._predict()
        self.assertEqual(self.model_loads, 1)
        self.assertEqual(self._predict(), 0)
        self.assertEqual(self.model_loads, 1)
        self.assertEqual(len(read_csv(self.run_dir / "submission.csv")), 4)

    def test_continuing_with_other_settings_is_refused(self):
        self.crash_on_decode = 1
        with self.assertRaises(RuntimeError):
            self._predict()
        before = (self.run_dir / "predictions.csv").read_text()
        swapped = [f"kade_letters={self.english}", f"survey_notes={self.english}", f"dominy_accounts={self.english}"]
        with self.assertRaisesRegex(SystemExit, "different settings"):
            self._predict(adapters=swapped)
        with self.assertRaisesRegex(SystemExit, "different settings"):
            self._predict("--fold", str(self.fold))
        self.assertEqual((self.run_dir / "predictions.csv").read_text(), before)

    def test_an_empty_page_is_reported_but_not_redone(self):
        self.texts = {"survey_005_p001": "  \n"}
        with mock.patch("builtins.print") as printed:
            self._predict()
        self.assertEqual(self._summary()["empty_pages"], ["survey_005_p001"])
        self.assertIn("survey_005_p001", " ".join(str(call) for call in printed.call_args_list))
        self.decoded.clear()
        self._predict()
        self.assertEqual(self.decoded, [])

    def test_check_predictions_stops_on_anything_but_each_page_once_in_order(self):
        from finetune.predict import check_predictions

        path = self.root / "check.csv"
        order = ["a", "b", "c"]
        for ids, message in ((["a", "b"], "Missing 1"), (["a", "b", "c", "c"], "4 rows"), (["b", "a", "c"], "in order"), (["a", "b", "d"], "Missing 1")):
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, ["page_id", "text"])
                writer.writeheader()
                writer.writerows({"page_id": page_id, "text": "x"} for page_id in ids)
            with self.assertRaisesRegex(SystemExit, message):
                check_predictions(path, order)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, ["page_id", "text"])
            writer.writeheader()
            writer.writerows([{"page_id": "a", "text": "x\ny"}, {"page_id": "b", "text": ""}, {"page_id": "c", "text": " "}])
        self.assertEqual(check_predictions(path, order), ["b", "c"])

    def test_fold_mode_scores_and_writes_no_submission(self):
        self.assertEqual(self._predict("--fold", str(self.fold), adapters=[f"kade_letters={self.german}"]), 0)
        self.assertEqual([d["page_id"] for d in self.decoded], ["kade_001_p001", "kade_001_p002"])
        self.assertTrue(all(d["keys"] == PAGE_KEYS for d in self.decoded), "no label_source or reference text")
        self.assertEqual(self._summary()["cer"], 0.25)
        self.assertFalse((self.run_dir / "submission.csv").exists())

        from finetune.status import summarize

        self.assertEqual(summarize(self.run_dir)["scores"], [("fold", 0.25)])
        self.assertEqual(summarize(self.run_dir)["verdict"], "finished")

    @unittest.skipUnless(importlib.util.find_spec("peft"), "needs peft")
    def test_saved_adapters_really_load_and_switch_in_peft(self):
        """Real PEFT on a tiny model: the path no GPU run has exercised yet."""
        import torch
        from peft import LoraConfig, get_peft_model

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(8, 1)

            def forward(self, x):
                return self.layer(x)

        def wrapped(lora_seed=None):
            torch.manual_seed(0)
            base = Tiny()
            torch.manual_seed(lora_seed or 0)
            # A fresh LoRA starts at zero, as in load_model_and_processor; a seeded one stands in for a trained one.
            return get_peft_model(base, LoraConfig(r=2, lora_dropout=0.5, target_modules=["layer"], init_lora_weights=lora_seed is None))

        def output(model):
            with torch.no_grad():
                return str(model(torch.ones(1, 8)).item())

        expected = {}
        for path, seed in ((self.german, 1), (self.english, 2)):
            trained = wrapped(seed)
            trained.save_pretrained(path)
            expected[path] = output(trained.eval())
        untrained = output(wrapped().eval())
        self.assertEqual(len({*expected.values(), untrained}), 3)

        self._predict(load=lambda config, rank: (wrapped(), SimpleNamespace()),
                      generate=lambda model, processor, rows, *_: ([output(model)], [{"loop_cut": 0, "retry_used": 0}]))
        texts = {row["page_id"]: row["text"] for row in read_csv(self.run_dir / "submission.csv")}
        for page_id, category, _ in PAGES:
            self.assertEqual(texts[page_id], expected[self.german if category == "kade_letters" else self.english], page_id)


if __name__ == "__main__":
    unittest.main()
