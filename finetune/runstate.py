"""Record what a run is doing, so a run that dies can be diagnosed afterwards.

Two files in the run directory (which stays on persistent storage):

* `status.json`, rewritten in place: the current phase (loading, training,
  evaluating, saving, finished, failed), step, epoch, page within an
  evaluation, and a resource snapshot. After a crash it shows where the run was.
* `events.jsonl`, append-only: start, each epoch's evaluation and checkpoint,
  low-disk skips, signals received, the traceback of an exception, and the end.

Kubernetes sends SIGTERM before it stops a pod (an eviction, a workspace
shutdown), so a `signal` event as the last line means the run was stopped from
outside; no final event at all means it was killed without warning (SIGKILL,
for example out of memory or a node failure).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_int(path: str) -> int | None:
    try:
        text = Path(path).read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def _gib(value: int | None) -> float | None:
    return None if value is None else round(value / 1024**3, 2)


def resources(paths: dict[str, Path]) -> dict[str, Any]:
    """Memory and disk at this moment. Every field is best effort."""
    snapshot: dict[str, Any] = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                snapshot["process_ram_gib"] = round(int(line.split()[1]) / 1024**2, 2)
    except OSError:
        pass
    snapshot["container_ram_gib"] = _gib(_read_int("/sys/fs/cgroup/memory.current"))
    snapshot["container_ram_limit_gib"] = _gib(_read_int("/sys/fs/cgroup/memory.max"))
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            snapshot["gpu_allocated_gib"] = _gib(torch.cuda.memory_allocated())
            snapshot["gpu_peak_gib"] = _gib(torch.cuda.max_memory_allocated())
            snapshot["gpu_free_gib"] = _gib(free)
            snapshot["gpu_total_gib"] = _gib(total)
    except Exception:  # noqa: BLE001 - diagnostics must never stop a run
        pass
    for name, path in paths.items():
        try:
            usage = shutil.disk_usage(path)
            snapshot[f"disk_free_gib_{name}"] = _gib(usage.free)
        except OSError:
            pass
    return snapshot


class RunState:
    def __init__(self, run_dir: Path, watch: dict[str, Path] | None = None, enabled: bool = True):
        self.run_dir = run_dir
        self.watch = watch or {"run_dir": run_dir}
        self.enabled = enabled
        self.status: dict[str, Any] = {"host": socket.gethostname(), "pid": os.getpid(), "started": _now()}
        if enabled:
            run_dir.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, **fields: Any) -> None:
        if not self.enabled:
            return
        record = {"time": _now(), "event": kind, **fields}
        with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def update(self, with_resources: bool = False, **fields: Any) -> None:
        if not self.enabled:
            return
        self.status.update(fields, updated=_now())
        if with_resources:
            self.status["resources"] = resources(self.watch)
        temporary = self.run_dir / "status.json.tmp"
        temporary.write_text(json.dumps(self.status, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, self.run_dir / "status.json")

    def phase(self, name: str, **fields: Any) -> None:
        """Enter a phase: logged as an event and as the current status, with resources."""
        self.update(with_resources=True, phase=name, phase_started=_now(), **fields)
        self.event(name, **fields, resources=self.status.get("resources"))

    def failed(self, error: BaseException) -> None:
        self.update(with_resources=True, phase="failed", error=f"{type(error).__name__}: {error}")
        self.event("failed", error=f"{type(error).__name__}: {error}",
                   traceback="".join(traceback.format_exception(error))[-8000:],
                   resources=self.status.get("resources"))

    def catch_signals(self) -> None:
        """Log SIGTERM/SIGHUP/SIGINT, then stop the run through a normal exception."""
        def handler(number, _frame):
            name = signal.Signals(number).name
            self.update(with_resources=True, phase="stopping", signal=name)
            self.event("signal", signal=name, resources=self.status.get("resources"))
            raise SystemExit(128 + number)

        for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            try:
                signal.signal(number, handler)
            except (ValueError, OSError):
                pass


class Heartbeat:
    """Rate-limits status rewrites during long loops."""

    def __init__(self, state: RunState, seconds: float = 30.0):
        self.state, self.seconds, self.last = state, seconds, 0.0

    def __call__(self, **fields: Any) -> None:
        now = time.monotonic()
        if now - self.last >= self.seconds:
            self.last = now
            self.state.update(with_resources=True, **fields)
