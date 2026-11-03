from pathlib import Path
import tempfile
import unittest

from pipeline.dataset import image_path, read_page_ids


class DatasetTests(unittest.TestCase):
    def test_reads_unique_page_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pages.csv"
            path.write_text("page_id,text\na,one\nb,two\n", encoding="utf-8")
            self.assertEqual(read_page_ids(path), ["a", "b"])

    def test_reads_page_ids_from_bom_prefixed_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pages.csv"
            path.write_text("page_id\na\n", encoding="utf-8-sig")
            self.assertEqual(read_page_ids(path), ["a"])

    def test_resolves_nested_kaggle_image_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "images" / "page.jpg"
            expected.parent.mkdir()
            expected.touch()
            self.assertEqual(image_path(root, "page"), expected)
