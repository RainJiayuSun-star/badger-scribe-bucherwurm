from __future__ import annotations

import unittest

from finetune.prep import (
    find_page_break,
    illegible_spans,
    prepare_rows,
    split_spread,
    strip_postmark,
)


def row(page_id, doc_id, category, text, source="human"):
    return {"page_id": page_id, "doc_id": doc_id, "category": category, "label_source": source, "text": text}


class PostmarkTests(unittest.TestCase):
    def test_removes_the_indented_postmark_line(self):
        text, removed = strip_postmark("Lieber Sohn!\nWir sind gesund\n\t\t\t\tPostmark January 24, 1939")
        self.assertEqual(text, "Lieber Sohn!\nWir sind gesund")
        self.assertEqual(removed, 1)

    def test_keeps_lines_that_merely_mention_a_postmark(self):
        text, removed = strip_postmark("I saw the postmark on it\nPostmark")
        self.assertEqual(removed, 0)
        self.assertEqual(text, "I saw the postmark on it\nPostmark")


class IllegibleTests(unittest.TestCase):
    def test_reports_each_run_of_hash_marks(self):
        self.assertEqual(illegible_spans("as it is #### put in #\nok"), [(9, 13), (21, 22)])

    def test_no_marks_no_spans(self):
        self.assertEqual(illegible_spans("clean text"), [])


SPREAD = "\n".join(
    ["135", "Township 39 North"] + [f"left line {i}" for i in range(18)]
    + ["136", "Range 6 East"] + [f"right line {i}" for i in range(18)]
)


class SpreadTests(unittest.TestCase):
    def test_finds_the_right_page_number(self):
        self.assertEqual(find_page_break(SPREAD), 20)

    def test_splits_into_left_then_right(self):
        left, right = split_spread(SPREAD)
        self.assertTrue(left.startswith("135\n") and left.endswith("left line 17"))
        self.assertTrue(right.startswith("136\n") and right.endswith("right line 17"))

    def test_no_page_numbers_means_no_split(self):
        self.assertIsNone(split_spread("\n".join(f"line {i}" for i in range(40))))

    def test_a_number_pair_that_does_not_start_near_the_top_is_not_a_page_pair(self):
        text = "\n".join(["Sections"] * 5 + ["12"] + ["x"] * 14 + ["13"] + ["y"] * 20)
        self.assertIsNone(find_page_break(text))

    def test_a_break_far_from_the_middle_is_not_trusted(self):
        text = "\n".join(["135", "a", "136"] + ["x"] * 50)
        self.assertIsNone(find_page_break(text))


class PrepareRowsTests(unittest.TestCase):
    def test_applies_every_rule_and_keeps_the_input_order(self):
        rows = [
            row("kade_001_p001", "kade_001", "kade_letters", "Liebe Mutter\nPostmark March 3, 1938"),
            row("dominy_001_p002", "dominy_001", "dominy_accounts", "as soon as it is #### put in order", "silver_claude"),
            row("survey_001_p0001", "survey_001", "survey_notes", SPREAD, "silver_claude"),
        ]
        records = prepare_rows(rows)
        self.assertEqual([r["page_id"] for r in records], [r["page_id"] for r in rows])
        self.assertEqual(records[0]["text"], "Liebe Mutter")
        self.assertEqual(records[0]["notes"], ["postmark_removed"])
        self.assertEqual(records[1]["illegible_spans"], [(17, 21)])
        self.assertEqual([h["side"] for h in records[2]["halves"]], ["left", "right"])

    def test_unsplittable_spread_is_flagged_not_dropped(self):
        records = prepare_rows([row("survey_001_p0002", "survey_001", "survey_notes", "no numbers here\nat all")])
        self.assertIsNone(records[0]["halves"])
        self.assertIn("spread_not_split", records[0]["notes"])


if __name__ == "__main__":
    unittest.main()
