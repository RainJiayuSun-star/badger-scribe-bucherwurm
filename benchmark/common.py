"""Shared paths and CSV loading for Badger Scribe benchmarks."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))  # scripts here run by path, so the repo root isn't importable otherwise
from pipeline.outputs import write_predictions  # noqa: E402,F401  (atomic; re-exported for the scripts)


def data_root() -> Path:
    env = os.environ.get("BADGER_SCRIBE_DATA")
    if env:
        return Path(env).expanduser().resolve()
    return ROOT / "badger-scribe-data"


def load_metric(root: Path | None = None):
    """Import the official metric.py from the data directory."""
    path = (root or data_root()) / "metric.py"
    spec = importlib.util.spec_from_file_location("badger_metric", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS requested but not available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def image_dir() -> Path:
    nested = data_root() / "images" / "images"
    if nested.is_dir():
        return nested
    return data_root() / "images"


def image_path(page_id: str) -> Path:
    return image_dir() / f"{page_id}.jpg"


def splits_dir() -> Path:
    return ROOT / "splits"


def runs_dir() -> Path:
    return ROOT / "runs"


def results_dir() -> Path:
    return ROOT / "results"


def read_train_rows() -> list[dict]:
    path = data_root() / "train.csv"
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_metadata() -> dict[str, dict]:
    path = data_root() / "metadata.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["doc_id"]: row for row in csv.DictReader(f)}


def load_split(name: str = "val") -> dict:
    path = splits_dir() / "holdout.json"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if name == "holdout":
        name = "val"
    if name not in data:
        raise KeyError(f"split {name!r} not in {path}")
    return data


def page_ids(split: str) -> list[str]:
    data = load_split(split)
    if split == "smoke":
        return list(data["smoke"])
    return list(data[split])
