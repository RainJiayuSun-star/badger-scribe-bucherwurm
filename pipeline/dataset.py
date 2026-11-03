"""Input CSV and image-layout handling for Badger Scribe pages."""

from __future__ import annotations

import csv
from pathlib import Path


def read_page_ids(path: Path) -> list[str]:
    # Kaggle CSV exports may carry a UTF-8 BOM before the first header.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "page_id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a page_id column")
        rows = list(reader)
    page_ids = [row["page_id"].strip() for row in rows if row.get("page_id", "").strip()]
    if len(page_ids) != len(set(page_ids)):
        raise ValueError(f"{path} contains duplicate page_id values")
    return page_ids


def image_path(image_dir: Path, page_id: str) -> Path:
    """Accept an image folder or the outer directory from the Kaggle archive."""
    direct = image_dir / f"{page_id}.jpg"
    nested = image_dir / "images" / f"{page_id}.jpg"
    if direct.exists() or not nested.exists():
        return direct
    return nested
