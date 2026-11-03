"""Regenerate the local reference-transcription CSVs for the saved splits.

The *_solution.csv files contain competition label text, so they are not
committed. This rebuilds them from train.csv and the page IDs already saved in
the tracked split JSON files, without rewriting those JSON files.

    python benchmark/make_solutions.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_train_rows, splits_dir  # noqa: E402

FIELDS = ["page_id", "text", "category", "label_source", "doc_id"]

# (split directory, ids file, key inside it, output name)
TARGETS = (
    ("", "holdout.json", "val", "holdout_solution.csv"),
    ("", "smoke.json", "pages", "smoke_solution.csv"),
    ("release_09", "holdout.json", "val", "holdout_solution.csv"),
    ("release_09", "smoke.json", "pages", "smoke_solution.csv"),
)


def solution_rows(page_ids: list[str], by_id: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    missing = [page_id for page_id in page_ids if page_id not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} page IDs are not in train.csv, e.g. {missing[:3]}")
    return [{field: by_id[page_id][field] for field in FIELDS} for page_id in page_ids]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    by_id = {row["page_id"]: row for row in read_train_rows()}
    for subdir, ids_file, key, output in TARGETS:
        folder = splits_dir() / subdir if subdir else splits_dir()
        ids = json.loads((folder / ids_file).read_text(encoding="utf-8"))[key]
        rows = solution_rows(list(ids), by_id)
        write_csv(folder / output, rows)
        print(f"wrote {folder / output} ({len(rows)} pages)")


if __name__ == "__main__":
    main()
