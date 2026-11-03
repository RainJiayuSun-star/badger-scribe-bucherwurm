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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .checkpoints import checkpoints_by_step
from .runstate import gib

ACTIVE_PHASES = {"loading_model", "training", "evaluating", "saving_checkpoint", "stopping"}
# status.json is rewritten every 30 s while a run works, so older than this means stuck or gone.
STALE_SECONDS = 15 * 60


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def age_seconds(stamp: str | None, now: datetime | None = None) -> float | None:
    if not stamp:
        return None
    try:
        return ((now or datetime.now(timezone.utc)) - datetime.fromisoformat(stamp)).total_seconds()
    except (TypeError, ValueError):
        return None


def run_is_live(status: dict[str, Any], now: datetime | None = None) -> bool:
    """Whether a status.json describes a run still working on this machine.

    Process IDs are reused after a restart, so a live PID alone proves nothing:
    it can be an unrelated process, or this very launch.
    """
    pid = status.get("pid")
    if status.get("phase") not in ACTIVE_PHASES or status.get("host") != socket.gethostname():
        return False
    if not isinstance(pid, int) or pid == os.getpid() or not pid_alive(pid):
        return False
    age = age_seconds(status.get("updated"), now)
    return age is not None and age <= STALE_SECONDS


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
        free = gib(shutil.disk_usage(existing).free)
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
    if gib(free) < min_free_gib:
        return [f"Only {gib(free):.1f} of {gib(total):.1f} GiB GPU memory is free; this run needs about {min_free_gib:g} GiB. "
                f"Another job is probably using the GPU (check nvidia-smi), or lower min_free_gpu_gib."]
    return []


def check_run_dir(run_dir: Path, resume: bool, eval_only: bool) -> list[str]:
    problems = []
    status_path = run_dir / "status.json"
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = {}
        if run_is_live(status):
            problems.append(f"Run {run_dir.name} is already running (pid {status['pid']}, phase {status['phase']}). "
                            f"Wait for it, stop it, or use a different --run-id.")
    if eval_only:
        return problems
    has_checkpoint = bool(checkpoints_by_step(run_dir))
    if has_checkpoint and not resume:
        problems.append(f"{run_dir} already has checkpoints from an earlier run. "
                        f"Add --resume to continue it, or use a different --run-id to start fresh.")
    elif (run_dir / "best_adapter").exists() and not has_checkpoint:
        # The first epoch of a new run would replace this model, even if it scored worse.
        problems.append(f"{run_dir} already has a trained best_adapter and no checkpoint to continue from. "
                        f"Use a different --run-id so it isn't overwritten.")
    return problems


def run_checks(config: dict[str, Any], run_dir: Path, resume: bool, eval_only: bool,
               mem_get_info: Callable[[], tuple[int, int]] | None = None) -> None:
    """Raise SystemExit listing every problem found, or return if there are none."""
    problems = check_data(Path(config["data_root"]))
    problems += check_disk({"run directory": run_dir}, float(config["min_free_disk_gib"]))
    problems += check_gpu(float(config["min_free_gpu_gib"]), mem_get_info)
    problems += check_run_dir(run_dir, resume, eval_only)
    if problems:
        raise SystemExit("Preflight checks failed; nothing was started:\n" + "\n".join(f"  - {p}" for p in problems))
