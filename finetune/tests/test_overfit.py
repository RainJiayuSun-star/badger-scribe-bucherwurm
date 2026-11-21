from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from finetune import overfit
from finetune.data import train_sample
from finetune.folds import build_folds, check_folds
from finetune.train import stage_rows
from finetune.tests.test_folds import SPEC, rows_for


class AccountDocsFromTests(unittest.TestCase):
    def test_old_groups_are_kept_and_new_books_always_train(self):
        old_rows = rows_for(SPEC)
        new_rows = old_rows + rows_for({"dominy_100": ("dominy_accounts", 40), "dominy_101": ("dominy_accounts", 3)})
        old = {f["name"]: f for f in build_folds(old_rows)}
        counts = Counter(r["doc_id"] for r in old_rows if r["category"] == "dominy_accounts")
        new = {f["name"]: f for f in build_folds(new_rows, counts)}
        for name, fold in old.items():
            self.assertEqual(new[name]["eval_ids"], fold["eval_ids"], name)
        # A book moved to test leaves its group; the other groups keep their exact membership.
        moved = old["dominy_group1"]["eval_docs"][0]
        shrunk = {f["name"]: f for f in build_folds([r for r in new_rows if r["doc_id"] != moved], counts)}
        self.assertNotIn(moved, shrunk["dominy_group1"]["eval_docs"])
        for name in old:
            if name != "dominy_group1":
                self.assertEqual(shrunk[name]["eval_ids"], old[name]["eval_ids"], name)
        always = {"dominy_100", "dominy_101"}
        check_folds(list(new.values()), new_rows, always)
        self.assertTrue(all(any(p.startswith("dominy_100") for p in f["train_ids"]) for f in new.values()))
        with self.assertRaises(ValueError):
            check_folds(list(new.values()), new_rows)  # without the exemption the new books count as never held out
        leaked = dict(new["dominy_group1"], eval_ids=new["dominy_group1"]["eval_ids"] + ["dominy_100_p000"])
        with self.assertRaises(ValueError):
            check_folds([leaked if f["name"] == "dominy_group1" else f for f in new.values()], new_rows, always)


class TrainingRowsTests(unittest.TestCase):
    def test_excluded_label_sources_never_train(self):
        rows = [{"page_id": f"p{i}", "category": "dominy_accounts", "label_source": source}
                for i, source in enumerate(["human", "silver_claude-opus", "silver_claude-fable"])]
        stages = stage_rows({"train": rows}, {"stages": [["dominy_accounts"]]}, 2, ("silver_claude-opus",))
        self.assertEqual(sorted(r["page_id"] for r in stages[0]), ["p0", "p0", "p2"])

    def test_training_sample_is_fixed_unique_and_category_limited(self):
        rows = [{"page_id": f"p{i}", "category": "dominy_accounts" if i % 2 else "survey_notes"} for i in range(20)]
        rows += [dict(rows[1], repeat=1)]
        first = train_sample(rows, 4, 7, {"dominy_accounts"})
        self.assertEqual(first, train_sample(list(reversed(rows)), 4, 7, {"dominy_accounts"}))
        self.assertEqual(len({r["page_id"] for r in first}), 4)
        self.assertTrue(all(r["category"] == "dominy_accounts" and "repeat" not in r for r in first))
        self.assertEqual(len(train_sample(rows, 50, 7, set())), 20, "no held-out category: every category")
        self.assertEqual(train_sample(rows, 0, 7, set()), [])


def write_metrics(run: Path, held: list[float], train: list[float]) -> None:
    run.mkdir(parents=True)
    lines = []
    for epoch, (v, t) in enumerate(zip(held, train), start=1):
        lines += [json.dumps({"epoch": epoch, "val_macro_cer": v}), json.dumps({"epoch": epoch, "train_macro_cer": t}),
                  json.dumps({"epoch": epoch, "train_loss": 0.1})]
    (run / "metrics.jsonl").write_text("\n".join(lines) + "\n{cut off")


class ReportTests(unittest.TestCase):
    def test_verdicts(self):
        curve = lambda held, train: {e: {"val": v, "train": t} for e, (v, t) in enumerate(zip(held, train), start=1)}
        self.assertTrue(overfit.verdict(curve([0.30, 0.25, 0.27, 0.29], [0.20, 0.10, 0.06, 0.03])).startswith("overfitting"))
        self.assertTrue(overfit.verdict(curve([0.30, 0.25, 0.252], [0.2, 0.1, 0.05])).startswith("no overfitting"))
        self.assertIn("unstable", overfit.verdict(curve([0.20, 0.35, 0.30], [0.10, 0.20, 0.15])))

    def test_report_picks_the_fewest_epochs_within_noise_of_the_best_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_metrics(Path(tmp) / "of-a", [0.30, 0.25, 0.248, 0.26], [0.2, 0.1, 0.05, 0.03])
            write_metrics(Path(tmp) / "of-b", [0.20, 0.17, 0.168, 0.18], [0.2, 0.1, 0.05, 0.03])
            text = overfit.report(sorted(Path(tmp).iterdir()))
        self.assertIn("epochs to use for all-data runs: 2", text)
        self.assertIn("best mean at epoch 3", text)

    def test_config_and_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, out = Path(tmp) / "base.yaml", Path(tmp) / "of.yaml"
            base.write_text("model_id: m\nepochs: 3\n")
            config = overfit.write_config(base, out, ["loss_truncation=0.05", "exclude_label_sources=[silver_claude-opus]"])
            self.assertEqual((config["epochs"], config["train_eval_pages"], config["loss_truncation"]), (6, 12, 0.05))
            self.assertEqual(config["exclude_label_sources"], ["silver_claude-opus"])
            self.assertIn("model_id: m", out.read_text())
        line = overfit.queue_lines("of1", "cfg.yaml", "clone", ["dominy_group1"], 300)[0]
        self.assertEqual(line.split()[:3], ["of1-dominy_group1", "300", "clone"])
        self.assertTrue(line.endswith("train_item.sh of1-dominy_group1 cfg.yaml dominy_group1"))
