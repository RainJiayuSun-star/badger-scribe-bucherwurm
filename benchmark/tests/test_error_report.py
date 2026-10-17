from __future__ import annotations

import unittest

from benchmark.error_report import diff_html, ref_tokens


class DiffTests(unittest.TestCase):
    def test_identical_text_has_no_marks_and_keeps_line_breaks(self):
        out = diff_html("Liebe Mutter\nIch bin", "Liebe  Mutter Ich bin")
        self.assertNotIn("<del>", out)
        self.assertNotIn("<ins>", out)
        self.assertEqual(out, "Liebe Mutter<br>Ich bin")

    def test_one_letter_misread_is_marked_inside_the_word(self):
        out = diff_html("Nun mus Ich", "Nun muss Ich")
        self.assertIn("<ins>s</ins>", out)
        self.assertNotIn("<del>", out)

    def test_missed_line_is_struck_through(self):
        out = diff_html("In Liebe\nPostmark July 18, 1940", "In Liebe")
        self.assertIn("<del>Postmark July 18, 1940</del>", out)

    def test_unrelated_text_is_shown_whole(self):
        out = diff_html("Eastham", "[Stylized bird logo]")
        self.assertIn("<del>Eastham</del>", out)
        self.assertIn("<ins>[Stylized bird logo]</ins>", out)

    def test_illegible_reference_is_grey_not_red(self):
        out = diff_html("den 1# Mai", "den 12 Mai")
        self.assertIn('class="ill"', out)
        self.assertNotIn("<del>", out)

    def test_html_in_text_is_escaped(self):
        self.assertIn("&amp;lt;", diff_html("a", "&lt;"))

    def test_tokens_mark_the_last_word_of_each_line(self):
        self.assertEqual(ref_tokens("a b\nc"), [("a", False), ("b", True), ("c", True)])


if __name__ == "__main__":
    unittest.main()
