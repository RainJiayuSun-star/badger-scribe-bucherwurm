"""Create the Release 09 extended holdout without changing the Wave 1 split.

The original frozen holdout stays in ``splits/``. This script writes an
independent release-specific split under ``splits/release_09/``:

* every page from the original 95-page validation holdout, unchanged;
* three deterministic pages from each survey document newly present in the
  September release (24 additional pages).

The challenge permits a page-level survey split because the survey collection
is one bound volume. Kade and Dominy retain their original document-level
holdout, preserving all Wave 1 comparisons.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = ROOT.parent / "badger-scribe-data-09"
ORIGINAL_HOLDOUT = ROOT / "splits" / "holdout.json"
ORIGINAL_SMOKE = ROOT / "splits" / "smoke.json"
NEW_SURVEY_DOCS = (
    "survey_002", "survey_003", "survey_004", "survey_006",
    "survey_007", "survey_009", "survey_010", "survey_011",
)
PAGES_PER_NEW_SURVEY_DOC = 3


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def stable_sample(page_ids: list[str], count: int) -> list[str]:
    ranked = sorted(
        page_ids,
        key=lambda page_id: hashlib.sha256(
            f"badger-scribe-release-09-v1:{page_id}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ranked) < count:
        raise ValueError(f"need {count} pages, found only {len(ranked)}")
    return ranked[:count]


def write_solution(path: Path, page_ids: list[str], by_id: dict[str, dict[str, str]]) -> None:
    fields = ["page_id", "text", "category", "label_source", "doc_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for page_id in page_ids:
            writer.writerow({field: by_id[page_id][field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "splits" / "release_09")
    args = parser.parse_args()

    rows = read_rows(args.data_root / "train.csv")
    by_id = {row["page_id"]: row for row in rows}
    original = json.loads(ORIGINAL_HOLDOUT.read_text(encoding="utf-8"))
    original_val = list(original["val"])
    missing_original = [page_id for page_id in original_val if page_id not in by_id]
    if missing_original:
        raise SystemExit(f"Release 09 is missing original validation pages: {missing_original}")

    extra_by_doc: dict[str, list[str]] = {}
    for doc_id in NEW_SURVEY_DOCS:
        candidates = [
            row["page_id"] for row in rows
            if row["doc_id"] == doc_id and row["category"] == "survey_notes"
        ]
        extra_by_doc[doc_id] = stable_sample(candidates, PAGES_PER_NEW_SURVEY_DOC)
    extra_survey = [page_id for doc_id in NEW_SURVEY_DOCS for page_id in extra_by_doc[doc_id]]
    validation = original_val + extra_survey
    validation_set = set(validation)
    training = [row["page_id"] for row in rows if row["page_id"] not in validation_set]

    original_smoke = json.loads(ORIGINAL_SMOKE.read_text(encoding="utf-8"))["pages"]
    kade_dominy_smoke = [page_id for page_id in original_smoke if by_id[page_id]["category"] != "survey_notes"]
    smoke = kade_dominy_smoke + extra_survey[:4]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "description": "Release 09 extended holdout. Preserves the Wave 1 validation pages and adds survey pages from the September release.",
        "data_root": str(args.data_root.resolve()),
        "selection": {
            "original_holdout": str(ORIGINAL_HOLDOUT.relative_to(ROOT)),
            "original_validation_pages": len(original_val),
            "additional_survey_docs": list(NEW_SURVEY_DOCS),
            "pages_per_new_survey_doc": PAGES_PER_NEW_SURVEY_DOC,
            "selection_method": "SHA-256 rank of badger-scribe-release-09-v1:<page_id>",
        },
        "counts": {
            "train": len(training),
            "val": len(validation),
            "smoke": len(smoke),
            "val_by_category": dict(sorted(Counter(by_id[page_id]["category"] for page_id in validation).items())),
            "val_by_label_source": dict(sorted(Counter(by_id[page_id]["label_source"] for page_id in validation).items())),
        },
        "train": training,
        "val": validation,
        "smoke": smoke,
        "extra_survey_by_doc": extra_by_doc,
    }
    (args.output_dir / "holdout.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "smoke.json").write_text(json.dumps({"description": "12-page Release 09 smoke subset: four pages per category.", "pages": smoke}, indent=2) + "\n", encoding="utf-8")
    write_solution(args.output_dir / "holdout_solution.csv", validation, by_id)
    write_solution(args.output_dir / "smoke_solution.csv", smoke, by_id)
    full_train = {
        "description": "All labeled Release 09 training pages, in train.csv order. Use only for diagnostic fit benchmarking; it is not an independent validation set.",
        "data_root": str(args.data_root.resolve()),
        "n_pages": len(rows),
        "counts": {
            "by_category": dict(sorted(Counter(row["category"] for row in rows).items())),
            "by_label_source": dict(sorted(Counter(row["label_source"] for row in rows).items())),
        },
        "pages": rows,
    }
    (args.output_dir / "train_full.json").write_text(
        json.dumps(full_train, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"wrote {args.output_dir}")
    print(f"train={len(training)} val={len(validation)} smoke={len(smoke)} full_train={len(rows)}")
    print("validation by category:", metadata["counts"]["val_by_category"])


if __name__ == "__main__":
    main()
