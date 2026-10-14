"""Freeze the local train/val holdout. Deterministic; do not reshape after scoring starts."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import data_root, read_train_rows, splits_dir

KADE_VAL_DOC = "kade_007"
DOMINY_VAL_DOCS = {
    "dominy_001",
    "dominy_007",
    "dominy_008",
    "dominy_010",
    "dominy_011",
    "dominy_012",
}
SURVEY_VAL_PAGES = {
    "survey_001_p0025",
    "survey_001_p0027",
    "survey_001_p0028",
    "survey_001_p0029",
    "survey_001_p0030",
    "survey_001_p0032",
    "survey_001_p0035",
    "survey_001_p0036",
}
# 12-page smoke set drawn only from val (4 per category).
SMOKE_PAGES = [
    "kade_007_p005",
    "kade_007_p022",
    "kade_007_p040",
    "kade_007_p062",
    "dominy_001_p002",
    "dominy_001_p003",
    "dominy_008_p002",
    "dominy_010_p002",
    "survey_001_p0025",
    "survey_001_p0028",
    "survey_001_p0032",
    "survey_001_p0036",
]


def assign_split(row: dict) -> str:
    cat = row["category"]
    doc = row["doc_id"]
    pid = row["page_id"]
    if cat == "kade_letters":
        return "val" if doc == KADE_VAL_DOC else "train"
    if cat == "dominy_accounts":
        return "val" if doc in DOMINY_VAL_DOCS else "train"
    if cat == "survey_notes":
        return "val" if pid in SURVEY_VAL_PAGES else "train"
    raise ValueError(f"unknown category {cat}")


def main() -> None:
    rows = read_train_rows()
    by_split: dict[str, list[str]] = {"train": [], "val": []}
    meta = defaultdict(lambda: defaultdict(int))
    for row in rows:
        split = assign_split(row)
        by_split[split].append(row["page_id"])
        meta[split][row["category"]] += 1
        meta[f"{split}_source"][row["label_source"]] += 1

    missing_smoke = [p for p in SMOKE_PAGES if p not in by_split["val"]]
    if missing_smoke:
        raise SystemExit(f"smoke pages not in val: {missing_smoke}")

    out_dir = splits_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Frozen local holdout of released train.csv. Never use Kaggle test for selection.",
        "kade_val_doc": KADE_VAL_DOC,
        "dominy_val_docs": sorted(DOMINY_VAL_DOCS),
        "survey_val_pages": sorted(SURVEY_VAL_PAGES),
        "counts": {k: dict(v) for k, v in meta.items()},
        "n_train": len(by_split["train"]),
        "n_val": len(by_split["val"]),
        "n_smoke": len(SMOKE_PAGES),
        "train": by_split["train"],
        "val": by_split["val"],
        "smoke": SMOKE_PAGES,
    }
    holdout_path = out_dir / "holdout.json"
    holdout_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    smoke_path = out_dir / "smoke.json"
    smoke_path.write_text(
        json.dumps(
            {
                "description": "12-page smoke subset of val; spans all three categories.",
                "pages": SMOKE_PAGES,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    train_index = {r["page_id"]: r for r in rows}
    sol_path = out_dir / "holdout_solution.csv"
    with sol_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["page_id", "text", "category", "label_source", "doc_id"])
        w.writeheader()
        for pid in by_split["val"]:
            r = train_index[pid]
            w.writerow({
                "page_id": r["page_id"],
                "text": r["text"],
                "category": r["category"],
                "label_source": r["label_source"],
                "doc_id": r["doc_id"],
            })

    smoke_sol = out_dir / "smoke_solution.csv"
    with smoke_sol.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["page_id", "text", "category", "label_source", "doc_id"])
        w.writeheader()
        for pid in SMOKE_PAGES:
            r = train_index[pid]
            w.writerow({
                "page_id": r["page_id"],
                "text": r["text"],
                "category": r["category"],
                "label_source": r["label_source"],
                "doc_id": r["doc_id"],
            })

    print(f"wrote {holdout_path}")
    print(f"train={payload['n_train']} val={payload['n_val']} smoke={payload['n_smoke']}")
    print("val by category:", dict(meta["val"]))
    print("official metric.py expects solution columns page_id,text,category")
    print(f"  {sol_path}")
    print(f"  {smoke_sol}")
    print("data root:", data_root())


if __name__ == "__main__":
    main()
