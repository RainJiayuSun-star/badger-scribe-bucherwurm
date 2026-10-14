from pathlib import Path
import tempfile
import unittest

from pipeline.outputs import read_completed
from pipeline.prompts import VERBATIM_V1
from pipeline.runner import run_pages


class FakeBackend:
    def transcribe(self, image: Path):
        return f"text for {image.stem}", {"elapsed_seconds": 0.1}


class RunnerTests(unittest.TestCase):
    def test_checkpoint_keeps_prior_predictions_when_resuming(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "new.jpg").touch()
            output = root / "predictions.csv"
            run_pages(
                backend=FakeBackend(),
                page_ids=["new"],
                all_page_ids=["old", "new"],
                predictions={"old": "saved text"},
                image_dir=root,
                output_csv=output,
                log_jsonl=root / "requests.jsonl",
                prompt=VERBATIM_V1,
                model="test-model",
                base_url="https://example.test/v1",
            )
            self.assertEqual(read_completed(output), {"old": "saved text", "new": "text for new"})
