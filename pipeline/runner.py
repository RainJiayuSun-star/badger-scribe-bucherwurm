"""Page-by-page orchestration shared by remote and future local backends."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

from pipeline.dataset import image_path
from pipeline.outputs import append_log, write_predictions
from pipeline.prompts import Prompt


class TranscriptionBackend(Protocol):
    def transcribe(self, image: Path) -> tuple[str, dict]: ...


def run_pages(*, backend: TranscriptionBackend, page_ids: list[str], all_page_ids: list[str], predictions: dict[str, str], image_dir: Path, output_csv: Path, log_jsonl: Path, prompt: Prompt, model: str, base_url: str) -> None:
    """Process remaining pages, checkpointing successes and logging failures."""
    for number, page_id in enumerate(page_ids, start=1):
        try:
            text, metadata = backend.transcribe(image_path(image_dir, page_id))
            predictions[page_id] = text
            write_predictions(output_csv, predictions, all_page_ids)
            append_log(log_jsonl, {"page_id": page_id, "status": "ok", "model": model, "base_url": base_url.rstrip("/"), "prompt_name": prompt.name, "prompt_sha256": prompt.sha256, **metadata})
            print(f"[{number}/{len(page_ids)}] {page_id}: ok ({metadata['elapsed_seconds']:.1f}s)")
        except Exception as exc:
            append_log(log_jsonl, {"page_id": page_id, "status": "error", "model": model, "base_url": base_url.rstrip("/"), "prompt_name": prompt.name, "prompt_sha256": prompt.sha256, "error": str(exc)})
            print(f"[{number}/{len(page_ids)}] {page_id}: ERROR {exc}", file=sys.stderr)
