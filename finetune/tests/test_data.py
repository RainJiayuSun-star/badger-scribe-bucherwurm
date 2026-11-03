from __future__ import annotations

import unittest

from finetune.data import DEV_HUMAN_COUNTS, make_partition


def rows():
    result = []
    counts = {"kade_letters": 218, "dominy_accounts": 2, "survey_notes": 15}
    for category, count in counts.items():
        result.extend(
            {
                "page_id": f"{category}-{index}",
                "category": category,
                "label_source": "human",
                "text": "text",
                "doc_id": "document",
            }
            for index in range(count)
        )
    result.append({"page_id": "held", "category": "kade_letters", "label_source": "human", "text": "held", "doc_id": "held"})
    return result


class PartitionTests(unittest.TestCase):
    def test_fixed_counts_and_no_holdout_leakage(self):
        partition = make_partition(rows(), {"held"}, 7)
        dev = partition["development"]
        self.assertNotIn("held", {row["page_id"] for row in partition["train"]})
        for category, count in DEV_HUMAN_COUNTS.items():
            self.assertEqual(sum(row["category"] == category for row in dev), count)

    def test_partition_is_deterministic(self):
        self.assertEqual(
            [row["page_id"] for row in make_partition(rows(), {"held"}, 7)["development"]],
            [row["page_id"] for row in make_partition(rows(), {"held"}, 7)["development"]],
        )
