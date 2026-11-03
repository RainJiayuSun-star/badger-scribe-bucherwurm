"""Clean training labels (offline, no GPU).

The answer keys are used as fine-tuning targets, and a few of their quirks would
teach a model the wrong thing. This module builds cleaned records without
touching train.csv:

* Postmark lines (copied from an envelope, not visible on the page) are removed.
* Runs of `#` (illegible in the reference) are counted for the summary. The
  trainer masks them by token (vlm.mask_illegible): a `#` in a prediction is
  scored as an ordinary character, so a model should never be taught to emit one.

    python -m finetune.prep            # writes runs/prep/records.jsonl and prints a summary
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .config import ROOT, default_data_root

POSTMARK = re.compile(r"^[ \t]*postmark\b[^\n]*\d", re.IGNORECASE)
ILLEGIBLE = re.compile(r"#+")


def strip_postmark(text: str) -> tuple[str, int]:
    """Drop envelope Postmark lines. Returns the cleaned text and how many lines went."""
    lines = text.split("\n")
    kept = [line for line in lines if not POSTMARK.match(line)]
    return "\n".join(kept).rstrip(), len(lines) - len(kept)


def illegible_spans(text: str) -> list[tuple[int, int]]:
    """Character spans (start, end) of each run of `#`."""
    return [(match.start(), match.end()) for match in ILLEGIBLE.finditer(text)]


def prepare_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Return one cleaned record per training row, in input order."""
    records = []
    for row in rows:
        text, notes = row["text"], []
        text, removed = strip_postmark(text)
        if removed:
            notes.append("postmark_removed")
        records.append({
            "page_id": row["page_id"],
            "doc_id": row["doc_id"],
            "category": row["category"],
            "label_source": row["label_source"],
            "text": text,
            "illegible_spans": illegible_spans(text),
            "notes": notes,
        })
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pages": len(records),
        "postmark_pages": sum("postmark_removed" in r["notes"] for r in records),
        "illegible_pages": sum(bool(r["illegible_spans"]) for r in records),
        "illegible_chars": sum(end - start for r in records for start, end in r["illegible_spans"]),
        "notes": dict(Counter(note for r in records for note in r["notes"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "prep" / "records.jsonl")
    args = parser.parse_args()
    with (args.data_root / "train.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    records = prepare_rows(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    print(json.dumps(summarize(records), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
