from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = PROJECT_ROOT / "pipeline" / "transcribe_gateway.py"


class CliTests(unittest.TestCase):
    def test_continue_and_no_resume_are_mutually_exclusive(self):
        result = subprocess.run(
            [sys.executable, str(ENTRYPOINT), "--continue", "--no-resume"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("not allowed with argument", result.stderr)

    def test_rejects_unknown_selected_page_id_before_calling_gateway(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_csv = root / "pages.csv"
            input_csv.write_text("page_id\nknown\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ENTRYPOINT), "--input-csv", str(input_csv),
                 "--image-dir", str(root), "--output-csv", str(root / "out.csv"),
                 "--page-id", "missing"],
                env={"OPENAI_API_KEY": "placeholder"},
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("page ID not found", result.stderr)
