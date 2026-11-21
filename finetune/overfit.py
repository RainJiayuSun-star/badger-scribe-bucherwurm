"""Routine overfitting test: train past the usual 3 epochs and, after every epoch, score a
fixed sample of training pages next to the held-out pages (config key train_eval_pages).

Training CER that keeps falling while held-out CER rises is overfitting. Averaging the
held-out curve over several folds gives one fixed epoch count for all-data runs, which
avoids choosing each fold's best epoch on its own held-out pages.

    # 1) a test config: a base config plus the test settings
    python -m finetune.overfit config finetune/configs/olmocr2-7b-bf16-english.yaml runs/pm/configs/of1-olmocr-trunc.yaml \\
        --set loss_truncation=0.05 --set data_root=/home/igloege/badger-scribe-data-1118
    # 2) queue lines for queue.sh, one per fold
    python -m finetune.overfit queue of1-olmocr-trunc runs/pm/configs/of1-olmocr-trunc.yaml badger-scribe-run-o \\
        dominy_group1 dominy_group2 dominy_group3 --minutes 480
    # 3) after the runs: per-epoch table, a verdict per fold and the epoch count to use
    python -m finetune.overfit report runs/finetune/of1-olmocr-trunc-*
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Repeating identical settings moved CER by about 0.005 (EXPERIMENT_LOG.md), so smaller changes mean little.
NOISE = 0.005
DEFAULTS = {"epochs": 6, "train_eval_pages": 12}


def write_config(base: Path, out: Path, sets: list[str]) -> dict:
    import yaml

    config = yaml.safe_load(base.read_text(encoding="utf-8"))
    config.update(DEFAULTS)
    for item in sets:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set takes KEY=VALUE, got {item!r}")
        config[key] = yaml.safe_load(value)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Overfitting test from {base.name}; written by finetune.overfit\n" + yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def queue_lines(name: str, config: str, clone: str, folds: list[str], minutes: int) -> list[str]:
    return [f"{name}-{fold} {minutes} {clone} bash ~/badger-scribe-bucherwurm/runs/pm/bin/train_item.sh {name}-{fold} {config} {fold}"
            for fold in folds]


def curves(run_dir: Path) -> dict[int, dict[str, float]]:
    """{epoch: {'val': held-out CER, 'train': training-sample CER}} from metrics.jsonl; numbers only."""
    by_epoch: dict[int, dict[str, float]] = {}
    path = run_dir / "metrics.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a line cut off mid-write
        for key, name in (("val_macro_cer", "val"), ("train_macro_cer", "train")):
            if isinstance(row.get(key), (int, float)) and isinstance(row.get("epoch"), int):
                by_epoch.setdefault(row["epoch"], {})[name] = float(row[key])
    return by_epoch


def verdict(curve: dict[int, dict[str, float]]) -> str:
    held = {e: c["val"] for e, c in curve.items() if "val" in c}
    if len(held) < 2:
        return "too few epochs"
    best, last = min(held, key=held.get), max(held)
    if held[last] - held[best] <= NOISE:
        return f"no overfitting seen: best epoch {best}, final epoch within {NOISE} of it"
    train = {e: c["train"] for e, c in curve.items() if "train" in c}
    if best in train and last in train and train[last] < train[best]:
        return (f"overfitting: held-out CER rose {held[last] - held[best]:.4f} after epoch {best} "
                f"while training CER fell {train[best] - train[last]:.4f}")
    return f"held-out CER rose {held[last] - held[best]:.4f} after epoch {best}, but training CER did not fall: unstable, not overfitting"


def report(run_dirs: list[Path]) -> str:
    lines, runs = [], {}
    for run_dir in run_dirs:
        curve = curves(run_dir)
        if not curve:
            continue
        runs[run_dir.name] = curve
        lines.append(f"{run_dir.name}")
        lines.append("  epoch  train CER  held-out CER  gap")
        for epoch in sorted(curve):
            train, val = curve[epoch].get("train"), curve[epoch].get("val")
            gap = f"{val - train:+.4f}" if train is not None and val is not None else "-"
            show = lambda x: "-" if x is None else f"{x:.4f}"
            lines.append(f"  {epoch:>5}  {show(train):>9}  {show(val):>12}  {gap}")
        lines.append(f"  {verdict(curve)}")
    if not runs:
        return "no metrics found"
    shared = sorted(set.intersection(*[{e for e, c in curve.items() if "val" in c} for curve in runs.values()]))
    if len(runs) > 1 and shared:
        means = {e: sum(curve[e]["val"] for curve in runs.values()) / len(runs) for e in shared}
        lines.append(f"mean held-out CER over {len(runs)} runs: " + ", ".join(f"epoch {e} {means[e]:.4f}" for e in shared))
        best = min(means, key=means.get)
        # Fewest epochs within the noise of the best mean: cheaper, and as good as we can measure.
        pick = min(e for e in shared if means[e] - means[best] <= NOISE)
        lines.append(f"epochs to use for all-data runs: {pick} (best mean at epoch {best}, {means[best]:.4f})")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("config", help="write a test config")
    c.add_argument("base", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a setting; repeat as needed (epochs and train_eval_pages too)")
    q = commands.add_parser("queue", help="print queue.sh lines, one per fold")
    q.add_argument("name")
    q.add_argument("config", help="config path inside the clone")
    q.add_argument("clone", help="clone folder under ~ on EFabric")
    q.add_argument("folds", nargs="+")
    q.add_argument("--minutes", type=int, default=480, help="expected minutes per run, for queue.sh's cutoff check")
    r = commands.add_parser("report", help="read finished or running test runs")
    r.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "config":
        config = write_config(args.base, args.out, args.set)
        print(f"wrote {args.out}: epochs {config['epochs']}, train_eval_pages {config['train_eval_pages']}")
    elif args.command == "queue":
        print("\n".join(queue_lines(args.name, args.config, args.clone, args.folds, args.minutes)))
    else:
        print(report(args.runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
