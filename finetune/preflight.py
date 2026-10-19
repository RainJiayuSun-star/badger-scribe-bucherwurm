"""Checks run before a training or evaluation run loads the model.

Each failure stops the run with a message saying what is wrong and what to do,
instead of failing an hour in (a full disk at the first checkpoint, a GPU
already busy with another job) or quietly mixing two runs in one directory.
Thresholds live in the config: `min_free_disk_gib` and `min_free_gpu_gib`.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any, Callable

from .checkpoints import checkpoints_by_step

ACTIVE_PHASES = {"loading_model", "training", "evaluating", "saving_checkpoint", "stopping"}


def _gib(value: int) -> float:
    return value / 1024**3


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def check_data(data_root: Path) -> list[str]:
    problems = []
    for name in ("train.csv", "metric.py"):
        if not (data_root / name).is_file():
            problems.append(f"{data_root / name} is missing. Put the competition data in {data_root} or set BADGER_SCRIBE_DATA.")
    if not (data_root / "images").is_dir():
        problems.append(f"{data_root / 'images'} is missing. Unpack the page images there.")
    return problems


def check_disk(paths: dict[str, Path], min_free_gib: float) -> list[str]:
    problems = []
    for label, path in paths.items():
        existing = path
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        free = _gib(shutil.disk_usage(existing).free)
        if free < min_free_gib:
            problems.append(f"Only {free:.1f} GiB free for the {label} ({path}); this run needs at least {min_free_gib:g} GiB. "
                            f"Free space (old checkpoint-* folders are the usual culprit) or lower min_free_disk_gib.")
    return problems


def check_gpu(min_free_gib: float, mem_get_info: Callable[[], tuple[int, int]] | None = None) -> list[str]:
    if min_free_gib <= 0:
        return []
    if mem_get_info is None:
        import torch

        mem_get_info = torch.cuda.mem_get_info
    free, total = mem_get_info()
    if _gib(free) < min_free_gib:
        return [f"Only {_gib(free):.1f} of {_gib(total):.1f} GiB GPU memory is free; this run needs about {min_free_gib:g} GiB. "
                f"Another job is probably using the GPU (check nvidia-smi), or lower min_free_gpu_gib."]
    return []


def check_run_dir(run_dir: Path, checkpoints_root: Path, resume: bool, eval_only: bool) -> list[str]:
    problems = []
    status_path = run_dir / "status.json"
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = {}
        if (status.get("phase") in ACTIVE_PHASES and status.get("host") == socket.gethostname()
                and isinstance(status.get("pid"), int) and _pid_alive(status["pid"])):
            problems.append(f"Run {run_dir.name} is already running (pid {status['pid']}, phase {status['phase']}). "
                            f"Wait for it, stop it, or use a different --run-id.")
    if not resume and not eval_only and checkpoints_by_step(checkpoints_root):
        problems.append(f"{checkpoints_root} already has checkpoints from an earlier run. "
                        f"Add --resume to continue it, or use a different --run-id to start fresh.")
    return problems


def run_checks(config: dict[str, Any], run_dir: Path, checkpoints_root: Path, resume: bool, eval_only: bool,
               mem_get_info: Callable[[], tuple[int, int]] | None = None) -> None:
    """Raise SystemExit listing every problem found, or return if there are none."""
    problems = check_data(Path(config["data_root"]))
    problems += check_disk({"run directory": run_dir, "checkpoints": checkpoints_root}, float(config["min_free_disk_gib"]))
    problems += check_gpu(float(config["min_free_gpu_gib"]), mem_get_info)
    problems += check_run_dir(run_dir, checkpoints_root, resume, eval_only)
    if problems:
        raise SystemExit("Preflight checks failed; nothing was started:\n" + "\n".join(f"  - {p}" for p in problems))
