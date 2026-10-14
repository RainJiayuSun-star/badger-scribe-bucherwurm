"""Configuration loading and experiment-name resolution."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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
    config.setdefault("data_root", str(ROOT.parent / "badger-scribe-data-09"))
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
