from pathlib import Path
import tempfile
import unittest

from pipeline.outputs import read_completed, write_predictions


class OutputTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_order_and_newlines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            write_predictions(path, {"b": "second\nline", "a": "first"}, ["a", "b", "c"])
            self.assertEqual(read_completed(path), {"a": "first", "b": "second\nline"})
