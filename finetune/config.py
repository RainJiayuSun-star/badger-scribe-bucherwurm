"""Configuration loading and experiment-name resolution."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    """Same rule as benchmark/common.py: BADGER_SCRIBE_DATA, else <repo>/badger-scribe-data."""
    env = os.environ.get("BADGER_SCRIBE_DATA")
    return Path(env).expanduser().resolve() if env else ROOT / "badger-scribe-data"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit("Install the fine-tuning extra: uv sync --extra finetune") from exc
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_config(path: Path) -> dict[str, Any]:
    config = _load_yaml(path)
    config["_config_path"] = str(path.resolve())
    config.setdefault("data_root", str(default_data_root()))
    config.setdefault("holdout", str(ROOT / "splits" / "release_09" / "holdout.json"))
    config.setdefault("runs_root", str(ROOT / "runs" / "finetune"))
    config.setdefault("seed", 20261003)
    config.setdefault("epochs", 3)
    config.setdefault("max_seq_length", 8192)
    config.setdefault("per_device_batch_size", 1)
    config.setdefault("effective_batch_size", 8)
    config.setdefault("learning_rate", 1.0e-4)
    config.setdefault("vision_learning_rate", 2.0e-5)
    config.setdefault("warmup_ratio", 0.05)
    config.setdefault("logging_steps", 5)
    config.setdefault("save_every_epochs", 1)
    config.setdefault("augmentation_profile", str(ROOT / "finetune" / "augmentation" / "default.yaml"))
    config.setdefault("adapter_policy", "language_projector")
    config.setdefault("wandb_project", "badger-scribe-finetune")
    # Image budget per page. One image token covers (patch * 2)^2 pixels, so at
    # patch 14 the default of 1280 tokens is about 1.0 MP, matching the benchmark.
    config.setdefault("image_patch_size", 14)
    config.setdefault("max_pixels", 1280 * 28 * 28)
    config.setdefault("max_pixels_by_category", {})
    # "4bit" (QLoRA, NF4) or "none" (bf16 LoRA; a 3B-4B model fits a 24 GB A5000).
    config.setdefault("quantization", "4bit")
    # Train on finetune.prep's cleaned labels (Postmark lines removed); scoring
    # always uses the original keys.
    config.setdefault("use_cleaned_labels", True)
    # Exclude `#` (illegible in the key) from the loss so the model never learns to emit it.
    config.setdefault("mask_illegible", True)
    # Stop decoding at a repetition loop and cut it back to its first copy (finetune.loopguard).
    config.setdefault("loop_guard", True)
    # Off by default: re-decode looping or near-empty pages with sampling at these temperatures.
    config.setdefault("loop_retry_temperatures", [])
    config.setdefault("retry_min_chars", 40)
    # Each human-labeled training page appears this many times per epoch. English
    # train pages are mostly unreviewed drafts, so this tilts training toward the
    # reviewed ones. Training only; inference never sees label_source.
    config.setdefault("human_repeats", 1)
    # Resume checkpoints hold the full model state (about 7.4 GB for Churro-3B).
    # checkpoint_root puts them on a scratch disk while metrics, predictions and
    # best_adapter stay in runs_root; keep_checkpoints deletes the oldest beyond
    # this many (0 keeps all).
    config.setdefault("checkpoint_root", None)
    config.setdefault("keep_checkpoints", 0)
    if "model_id" not in config:
        raise ValueError(f"{path} is missing model_id")
    return config


def load_curricula(path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = path or ROOT / "finetune" / "configs" / "curricula.yaml"
    data = _load_yaml(path)
    curricula = data.get("curricula")
    if not isinstance(curricula, dict):
        raise ValueError(f"{path} must contain curricula")
    return curricula


def resolve_curriculum(config: dict[str, Any], name: str | None) -> dict[str, Any]:
    curricula = load_curricula()
    selected = name or config.get("curriculum", "kade")
    if selected not in curricula:
        raise ValueError(f"unknown curriculum {selected!r}; choices: {', '.join(sorted(curricula))}")
    value = copy.deepcopy(curricula[selected])
    value["name"] = selected
    return value


def resolved_config_json(config: dict[str, Any]) -> str:
    return json.dumps(config, indent=2, sort_keys=True, default=str) + "\n"
