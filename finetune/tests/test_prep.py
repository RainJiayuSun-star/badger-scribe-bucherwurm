from __future__ import annotations

import unittest

from finetune.prep import illegible_spans, prepare_rows, strip_postmark


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


class PrepareRowsTests(unittest.TestCase):
    def test_applies_every_rule_and_keeps_the_input_order(self):
        rows = [
            row("kade_001_p001", "kade_001", "kade_letters", "Liebe Mutter\nPostmark March 3, 1938"),
            row("dominy_001_p002", "dominy_001", "dominy_accounts", "as soon as it is #### put in order", "silver_claude"),
        ]
        records = prepare_rows(rows)
        self.assertEqual([r["page_id"] for r in records], [r["page_id"] for r in rows])
        self.assertEqual(records[0]["text"], "Liebe Mutter")
        self.assertEqual(records[0]["notes"], ["postmark_removed"])
        self.assertEqual(records[1]["illegible_spans"], [(17, 21)])


if __name__ == "__main__":
    unittest.main()
