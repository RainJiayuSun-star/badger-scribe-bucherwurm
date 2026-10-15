"""Clean training labels and split two-page survey spreads (offline, no GPU).

The answer keys are used as fine-tuning targets, and a few of their quirks would
teach a model the wrong thing. This module builds cleaned records without
touching train.csv:

* Postmark lines (copied from an envelope, not visible on the page) are removed.
* Runs of `#` (illegible in the reference) are reported as character spans, so
  the trainer can mask the loss there. A `#` in a prediction is scored as an
  ordinary character, so a model should never be taught to emit one.
* A survey key covers a two-page spread (left page, then right page). Where the
  key contains both printed page numbers, it is split into the two halves.

    python -m finetune.prep            # writes runs/prep/records.jsonl and prints a summary
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import ROOT, default_data_root

POSTMARK = re.compile(r"^[ \t]*postmark\b[^\n]*\d", re.IGNORECASE)
ILLEGIBLE = re.compile(r"#+")
PAGE_NUMBER = re.compile(r"^\d{1,3}$")

# The right page's number sits about halfway through a key (observed mean 0.50,
# standard deviation 0.08). Anything far outside that is not trusted.
BREAK_MIN, BREAK_MAX = 0.20, 0.80
LEFT_NUMBER_MAX_LINE = 3


def strip_postmark(text: str) -> tuple[str, int]:
    """Drop envelope Postmark lines. Returns the cleaned text and how many lines went."""
    lines = text.split("\n")
    kept = [line for line in lines if not POSTMARK.match(line)]
    return "\n".join(kept).rstrip(), len(lines) - len(kept)


def illegible_spans(text: str) -> list[tuple[int, int]]:
    """Character spans (start, end) of each run of `#`."""
    return [(match.start(), match.end()) for match in ILLEGIBLE.finditer(text)]


def find_page_break(text: str) -> int | None:
    """Line index where the right page starts, or None if it cannot be told.

    A spread key reads the whole left page, then the whole right page, and each
    page starts with its printed number: a standalone N near the top, then a
    standalone N+1 about halfway through.
    """
    lines = text.split("\n")
    count = len(lines)
    numbers = [(index, int(line.strip())) for index, line in enumerate(lines) if PAGE_NUMBER.match(line.strip())]
    for left_index, left_number in numbers:
        if left_index > LEFT_NUMBER_MAX_LINE:
            break
        for right_index, right_number in numbers:
            if right_number == left_number + 1 and right_index > left_index:
                if BREAK_MIN * count <= right_index <= BREAK_MAX * count:
                    return right_index
    return None


def split_spread(text: str) -> tuple[str, str] | None:
    """Split a two-page key into (left, right), or None when no break is found."""
    index = find_page_break(text)
    if index is None:
        return None
    lines = text.split("\n")
    return "\n".join(lines[:index]).rstrip(), "\n".join(lines[index:]).rstrip()


def prepare_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Return one cleaned record per training row, in input order."""
    records = []
    for row in rows:
        text, notes = row["text"], []
        text, removed = strip_postmark(text)
        if removed:
            notes.append("postmark_removed")
        record: dict[str, Any] = {
            "page_id": row["page_id"],
            "doc_id": row["doc_id"],
            "category": row["category"],
            "label_source": row["label_source"],
            "text": text,
            "illegible_spans": illegible_spans(text),
            "notes": notes,
            "halves": None,
        }
        if row["category"] == "survey_notes":
            halves = split_spread(text)
            if halves is None:
                record["notes"].append("spread_not_split")
            else:
                record["halves"] = [
                    {"side": side, "text": half, "illegible_spans": illegible_spans(half)}
                    for side, half in zip(("left", "right"), halves)
                ]
        records.append(record)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    survey = [r for r in records if r["category"] == "survey_notes"]
    by_doc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in survey:
        by_doc[record["doc_id"]][1] += 1
        by_doc[record["doc_id"]][0] += record["halves"] is not None
    return {
        "pages": len(records),
        "postmark_pages": sum("postmark_removed" in r["notes"] for r in records),
        "illegible_pages": sum(bool(r["illegible_spans"]) for r in records),
        "illegible_chars": sum(end - start for r in records for start, end in r["illegible_spans"]),
        "survey_split_by_doc": {doc: f"{done}/{total}" for doc, (done, total) in sorted(by_doc.items())},
        "survey_split_total": f"{sum(v[0] for v in by_doc.values())}/{sum(v[1] for v in by_doc.values())}",
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
