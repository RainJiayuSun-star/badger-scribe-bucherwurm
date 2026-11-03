"""Summarize fine-tuning runs from their status.json and events.jsonl.

    python -m finetune.status                # one line per run, newest first
    python -m finetune.status <run-id>       # one run: where it is, scores, how it ended
"""

from __future__ import annotations

import argparse
import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoints import latest_checkpoint
from .config import ROOT
from .preflight import ACTIVE_PHASES, STALE_SECONDS, age_seconds, pid_alive


def read_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "events.jsonl"
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue    # a line cut off by a crash
    return events


def _alive(status: dict[str, Any]) -> bool | None:
    """Whether the run's process is alive, or None when it ran on another machine."""
    if status.get("host") != socket.gethostname() or not isinstance(status.get("pid"), int):
        return None
    return pid_alive(status["pid"])


def verdict(status: dict[str, Any], events: list[dict[str, Any]], now: datetime | None = None) -> str:
    """One sentence on the state of a run."""
    now = now or datetime.now(timezone.utc)
    phase = status.get("phase")
    last = events[-1]["event"] if events else None
    if phase == "finished":
        return "finished"
    if phase == "failed" or last == "failed":
        return "crashed with a Python error (see the traceback below)"
    if last == "signal":
        return f"stopped from outside ({events[-1].get('signal')}), e.g. the workspace was shut down or evicted"
    if phase in ACTIVE_PHASES:
        alive = _alive(status)
        age = age_seconds(status.get("updated"), now)
        minutes = f"{int(age // 60)} min ago" if age is not None else "at an unknown time"
        if alive is False:
            return f"died without warning while {phase} (out of memory, node failure, or workspace restart); last update {minutes}"
        if age is not None and age > STALE_SECONDS:
            return f"no progress while {phase}; last update {minutes} (stuck, or died on another machine)"
        return f"running: {phase}"
    return f"unknown (phase {phase!r})"


def _where(status: dict[str, Any]) -> str:
    parts = []
    if status.get("epoch"):
        parts.append(f"epoch {status['epoch']}")
    if status.get("step") is not None and status.get("total_steps"):
        parts.append(f"step {status['step']}/{status['total_steps']}")
    if status.get("phase") == "evaluating" and status.get("eval_pages"):
        parts.append(f"page {status.get('eval_page', 0)}/{status['eval_pages']}")
    return ", ".join(parts)


def summarize(run_dir: Path) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    events = read_events(run_dir)
    scores = [(e.get("epoch") or e.get("fold"), e["cer"]) for e in events if e["event"] == "evaluated"]
    checkpoint = latest_checkpoint(run_dir)
    return {"run_id": run_dir.name, "status": status, "events": events, "scores": scores,
            "verdict": verdict(status, events), "where": _where(status),
            "checkpoint": checkpoint.name if checkpoint else None,
            "updated": status.get("updated", "")}


def print_one(summary: dict[str, Any], tail: int = 8) -> None:
    status = summary["status"]
    print(f"{summary['run_id']}: {summary['verdict']}")
    if summary["where"]:
        print(f"  at: {summary['where']}")
    for label, cer in summary["scores"]:
        print(f"  CER after {label}: {cer:.4f}")
    print(f"  resume checkpoint: {summary['checkpoint'] or 'none'}")
    resources = status.get("resources") or {}
    if resources:
        print("  last resources: " + ", ".join(f"{k}={v}" for k, v in resources.items()))
    failed = [e for e in summary["events"] if e["event"] == "failed"]
    if failed:
        print("  traceback (end):")
        for line in failed[-1].get("traceback", "").rstrip().splitlines()[-12:]:
            print(f"    {line}")
    print(f"  last {min(tail, len(summary['events']))} events:")
    for event in summary["events"][-tail:]:
        details = {k: v for k, v in event.items() if k not in ("time", "event", "resources", "traceback")}
        print(f"    {event['time']}  {event['event']}  {json.dumps(details, default=str) if details else ''}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", nargs="?", help="Show one run in detail")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs" / "finetune")
    args = parser.parse_args()
    if args.run_id:
        run_dir = args.runs_root / args.run_id
        if not run_dir.is_dir():
            raise SystemExit(f"no run directory {run_dir}")
        print_one(summarize(run_dir))
        return 0
    runs = [summarize(path) for path in args.runs_root.iterdir() if (path / "status.json").is_file()] if args.runs_root.is_dir() else []
    if not runs:
        print(f"no runs with status.json under {args.runs_root}")
        return 0
    for summary in sorted(runs, key=lambda s: s["updated"], reverse=True):
        best = min((cer for _, cer in summary["scores"]), default=None)
        best_text = f"best CER {best:.4f}" if best is not None else "no score yet"
        where = f" [{summary['where']}]" if summary["where"] and not summary["verdict"].startswith("finished") else ""
        print(f"{summary['updated'][:16]:<17} {summary['run_id']:<40} {best_text:<18} {summary['verdict']}{where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
