"""Render standard training curves from a fine-tuning run's local CSV log."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics_path = args.run_dir / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no metrics in {metrics_path}")
    import matplotlib.pyplot as plt

    def values(key: str):
        result = []
        for row in rows:
            try:
                result.append((float(row["step"]), float(row[key])))
            except (KeyError, TypeError, ValueError):
                pass
        return zip(*result) if result else ([], [])

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, key, title in [
        (axes[0, 0], "train_loss", "Training loss"),
        (axes[0, 1], "learning_rate", "Learning rate"),
        (axes[1, 0], "val_macro_cer", "Human development macro CER"),
        (axes[1, 1], "peak_gpu_memory_mb", "Peak CUDA memory (MB)"),
    ]:
        x, y = values(key)
        if x:
            axis.plot(x, y, marker="o", markersize=3)
        axis.set(title=title, xlabel="Optimizer step", ylabel=key)
        axis.grid(alpha=0.25)
    output = args.output or args.run_dir / "training-curves.png"
    figure.savefig(output, dpi=160)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
