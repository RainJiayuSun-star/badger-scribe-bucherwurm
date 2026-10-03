"""Release 09 manifest and data-loading helpers.

All split decisions live here so a run cannot silently train on a holdout page.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CATEGORIES = ("kade_letters", "dominy_accounts", "survey_notes")
DEV_HUMAN_COUNTS = {"kade_letters": 55, "dominy_accounts": 2, "survey_notes": 4}


def read_rows(data_root: Path) -> list[dict[str, str]]:
    with (data_root / "train.csv").open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def held_ids(holdout: Path) -> set[str]:
    return set(json.loads(holdout.read_text(encoding="utf-8"))["val"])


def rank(seed: int, page_id: str) -> str:
    return hashlib.sha256(f"badger-finetune-dev-v1:{seed}:{page_id}".encode()).hexdigest()


def make_partition(rows: list[dict[str, str]], holdout_ids: set[str], seed: int) -> dict[str, Any]:
    """Create the fixed development set from eligible human-labeled pages."""
    eligible = [row for row in rows if row["page_id"] not in holdout_ids]
    dev_ids: set[str] = set()
    for category, count in DEV_HUMAN_COUNTS.items():
        human = [r for r in eligible if r["category"] == category and r["label_source"] == "human"]
        if len(human) < count:
            raise ValueError(f"need {count} eligible human {category} pages, found {len(human)}")
        human.sort(key=lambda row: rank(seed, row["page_id"]))
        dev_ids.update(row["page_id"] for row in human[:count])
    train = [row for row in eligible if row["page_id"] not in dev_ids]
    dev = [row for row in eligible if row["page_id"] in dev_ids]
    return {
        "seed": seed,
        "holdout_ids": sorted(holdout_ids),
        "development_ids": sorted(dev_ids),
        "train": train,
        "development": dev,
    }


def selected_rows(partition: dict[str, Any], categories: Iterable[str], split: str) -> list[dict[str, str]]:
    allowed = set(categories)
    rows = partition[split]
    selected = [row for row in rows if row["category"] in allowed]
    if not selected:
        raise ValueError(f"no {split} rows for {sorted(allowed)}")
    return selected


def manifest_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = {}
    for category in CATEGORIES:
        selected = [row for row in rows if row["category"] == category]
        by_category[category] = dict(Counter(row["label_source"] for row in selected))
    return {"n_pages": len(rows), "by_category_label_source": by_category}


def image_path(data_root: Path, page_id: str) -> Path:
    for base in (data_root / "images", data_root / "images" / "images"):
        path = base / f"{page_id}.jpg"
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing image for {page_id} under {data_root / 'images'}")


def write_manifest(path: Path, partition: dict[str, Any], curriculum: dict[str, Any], stages: list[list[dict[str, str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "partition_seed": partition["seed"],
        "holdout_ids": partition["holdout_ids"],
        "development_ids": partition["development_ids"],
        "curriculum": curriculum,
        "development": manifest_summary(partition["development"]),
        "stages": [
            {"summary": manifest_summary(rows), "page_ids": [row["page_id"] for row in rows]}
            for rows in stages
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
