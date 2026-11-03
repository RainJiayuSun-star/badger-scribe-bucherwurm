from __future__ import annotations

import unittest

from benchmark.make_solutions import FIELDS, solution_rows


class SolutionRowsTests(unittest.TestCase):
    def setUp(self):
        self.by_id = {
            "a_p1": {"page_id": "a_p1", "text": "one", "category": "c", "label_source": "human", "doc_id": "a", "extra": "x"},
            "b_p1": {"page_id": "b_p1", "text": "two", "category": "c", "label_source": "silver_claude", "doc_id": "b"},
        }

    def test_keeps_requested_order_and_only_the_solution_columns(self):
        rows = solution_rows(["b_p1", "a_p1"], self.by_id)
        self.assertEqual([row["page_id"] for row in rows], ["b_p1", "a_p1"])
        self.assertTrue(all(list(row) == FIELDS for row in rows))

    def test_unknown_page_id_is_an_error(self):
        with self.assertRaises(SystemExit):
            solution_rows(["missing"], self.by_id)


if __name__ == "__main__":
    unittest.main()
