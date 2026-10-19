"""Small resume checkpoints: only what training changes.

`accelerator.save_state` writes the whole model, frozen base weights included
(about 7.4 GB for Churro-3B), which filled the EFabric home quota and, written
to /tmp, got workspaces evicted. A LoRA run only changes its trainable
parameters, so a checkpoint here holds those, the optimizer and scheduler
state and the random-number state: about 0.4 GB for Churro-3B at rank 16.

A checkpoint is written under a temporary name and renamed when complete, so a
run killed mid-save never leaves a half checkpoint that looks finished. Old
checkpoints are removed oldest first (FIFO) only after the new one is in place,
so the last complete epoch is always on disk.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np

PREFIX = "checkpoint-"
PARTIAL_PREFIX = "partial-"
STATE_FILE = "state.json"
# Head room on top of the estimated size before a save is attempted.
DISK_MARGIN_BYTES = 512 * 1024**2


def checkpoint_dir(config: dict[str, Any], run_dir: Path) -> Path:
    """Where resume checkpoints go. Everything else stays in run_dir."""
    root = config.get("checkpoint_root")
    return Path(root) / config["run_id"] if root else run_dir


def checkpoints_by_step(directory: Path) -> list[Path]:
    """Complete checkpoints, oldest first. Partial and malformed ones are ignored."""
    found = []
    for path in directory.glob(f"{PREFIX}*"):
        step = path.name[len(PREFIX):]
        if step.isdigit() and (path / STATE_FILE).is_file():
            found.append((int(step), path))
    return [path for _, path in sorted(found)]


def latest_checkpoint(directory: Path) -> Path | None:
    checkpoints = checkpoints_by_step(directory)
    return checkpoints[-1] if checkpoints else None


def prune_checkpoints(directory: Path, keep: int) -> list[Path]:
    """Delete the oldest complete checkpoints so at most `keep` remain (0 keeps all)."""
    if keep <= 0:
        return []
    removed = checkpoints_by_step(directory)[:-keep]
    for path in removed:
        shutil.rmtree(path)
    return removed


def trainable_parameters(model) -> dict[str, Any]:
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def estimate_bytes(model) -> int:
    """Trainable weights plus two AdamW moments of the same size."""
    return sum(p.numel() * p.element_size() for p in trainable_parameters(model).values()) * 3


def disk_free(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def _directory_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def save_checkpoint(directory: Path, step: int, model, optimizer, scheduler, state: dict[str, Any]) -> tuple[Path, int]:
    """Write checkpoint-<step> atomically. Returns its path and size in bytes."""
    import torch
    from safetensors.torch import save_file

    final = directory / f"{PREFIX}{step}"
    partial = directory / f"{PARTIAL_PREFIX}{PREFIX}{step}"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    weights = {name: p.detach().to("cpu").contiguous() for name, p in trainable_parameters(model).items()}
    save_file(weights, str(partial / "trainable.safetensors"))
    torch.save(optimizer.state_dict(), partial / "optimizer.pt")
    torch.save(scheduler.state_dict(), partial / "scheduler.pt")
    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    with (partial / "rng.pkl").open("wb") as handle:
        pickle.dump(rng, handle)
    (partial / STATE_FILE).write_text(json.dumps({**state, "step": step}, indent=2) + "\n", encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    os.replace(partial, final)
    return final, _directory_bytes(final)


def load_checkpoint(path: Path, model, optimizer, scheduler) -> dict[str, Any]:
    """Restore a checkpoint written by save_checkpoint. Returns its state.json."""
    import torch
    from safetensors.torch import load_file

    weights = load_file(str(path / "trainable.safetensors"))
    parameters = trainable_parameters(model)
    missing = sorted(set(parameters) - set(weights))
    if missing:
        raise SystemExit(f"{path} does not match this model: {len(missing)} trainable tensors missing, e.g. {missing[0]}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(weights[name].to(parameter.device, parameter.dtype))
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False))
    scheduler.load_state_dict(torch.load(path / "scheduler.pt", map_location="cpu", weights_only=False))
    with (path / "rng.pkl").open("rb") as handle:
        rng = pickle.load(handle)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return json.loads((path / STATE_FILE).read_text(encoding="utf-8"))
