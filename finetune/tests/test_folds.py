from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from finetune.data import fold_partition
from finetune.folds import build_folds, check_folds, group_documents


def rows_for(spec):
    """spec: {doc_id: (category, pages)}"""
    rows = []
    for doc, (category, pages) in spec.items():
        for page in range(pages):
            rows.append({"page_id": f"{doc}_p{page:03d}", "doc_id": doc, "category": category, "label_source": "human"})
    return rows


SPEC = {
    "kade_001": ("kade_letters", 20), "kade_002": ("kade_letters", 5),
    "survey_001": ("survey_notes", 30), "survey_002": ("survey_notes", 30),
    **{f"dominy_{i:03d}": ("dominy_accounts", 1 + i % 4) for i in range(1, 13)},
}


class GroupDocumentsTests(unittest.TestCase):
    def test_balances_page_totals(self):
        groups = group_documents({"a": 4, "b": 3, "c": 2, "d": 2, "e": 1}, 2)
        self.assertEqual(groups, [["a", "d"], ["b", "c", "e"]])

    def test_is_deterministic(self):
        counts = {f"d{i}": i % 3 + 1 for i in range(18)}
        self.assertEqual(group_documents(counts, 6), group_documents(dict(reversed(list(counts.items()))), 6))


class BuildFoldsTests(unittest.TestCase):
    def setUp(self):
        self.rows = rows_for(SPEC)
        self.folds = build_folds(self.rows)

    def test_one_fold_per_german_and_survey_document_and_six_account_groups(self):
        names = [fold["name"] for fold in self.folds]
        self.assertEqual(names[:4], ["kade_001", "kade_002", "survey_001", "survey_002"])
        self.assertEqual(sum(name.startswith("dominy_group") for name in names), 6)

    def test_no_leakage_and_every_page_held_out_once(self):
        check_folds(self.folds, self.rows)

    def test_training_side_keeps_other_categories(self):
        fold = self.folds[0]
        self.assertTrue(any(page.startswith("survey_") for page in fold["train_ids"]))
        self.assertFalse(any(page.startswith("kade_001") for page in fold["train_ids"]))

    def test_small_held_out_set_is_flagged(self):
        by_name = {fold["name"]: fold for fold in self.folds}
        self.assertTrue(by_name["kade_002"]["noisy"])
        self.assertFalse(by_name["kade_001"]["noisy"])

    def test_check_catches_leakage(self):
        broken = [dict(self.folds[0], train_ids=self.folds[0]["train_ids"] + self.folds[0]["eval_ids"][:1])]
        with self.assertRaises(ValueError):
            check_folds(broken, self.rows)


class FoldPartitionTests(unittest.TestCase):
    def test_held_out_documents_become_the_development_set(self):
        rows = rows_for(SPEC)
        fold = build_folds(rows)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fold.json"
            path.write_text(json.dumps(fold))
            partition = fold_partition(rows, path)
        self.assertEqual({r["doc_id"] for r in partition["development"]}, {"kade_001"})
        self.assertNotIn("kade_001", {r["doc_id"] for r in partition["train"]})
        self.assertEqual(len(partition["train"]) + len(partition["development"]), len(rows))


if __name__ == "__main__":
    unittest.main()
