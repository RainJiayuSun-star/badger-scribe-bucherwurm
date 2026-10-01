"""Resumable predictions and append-only run metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_completed(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and set(("page_id", "text")) - set(rows[0]):
        raise ValueError(f"existing output {path} must contain page_id,text")
    return {row["page_id"]: row.get("text", "") for row in rows if row.get("page_id")}


def write_predictions(path: Path, predictions: dict[str, str], ordered_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["page_id", "text"])
        writer.writeheader()
        for page_id in ordered_ids:
            if page_id in predictions:
                writer.writerow({"page_id": page_id, "text": predictions[page_id]})
    temporary.replace(path)


def append_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
