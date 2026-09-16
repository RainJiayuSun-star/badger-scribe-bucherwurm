"""Sequential Wave 1 + pipeline driver. Continues on per-run failure."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)

INFER = ROOT / "benchmark" / "infer_vlm.py"
EVAL = ROOT / "benchmark" / "eval.py"
PIPE = ROOT / "benchmark" / "infer_pipeline.py"


def run(cmd: list[str]) -> int:
    print("\n>>>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    print(f"exit {proc.returncode}: {' '.join(cmd)}", flush=True)
    return proc.returncode


def infer_and_eval(run_id: str, split: str, infer_args: list[str], model: str, model_id: str) -> None:
    code = run([str(PY), str(INFER), *infer_args, "--run-id", run_id, "--split", split])
    preds = ROOT / "runs" / run_id / "preds.csv"
    if code == 0 and preds.exists():
        run([str(PY), str(EVAL), "--run-id", run_id, "--split", split, "--append-results"])
        return
    reason = "load_failed" if code == 2 else "infer_failed"
    run([
        str(PY), str(EVAL), "--run-id", run_id, "--split", split,
        "--record-skip", reason, "--model", model, "--model-id", model_id,
        "--device", "mps", "--dtype", "fp16",
    ])


def main() -> None:
    jobs = [
        # Baseline metadata ablation (smoke), then full holdout ± metadata.
        dict(run_id="smoke-churro-meta", split="smoke", model="churro",
             model_id="stanford-oval/churro-3B",
             infer=["--model", "churro", "--with-metadata"]),
        dict(run_id="churro3b-zeroshot", split="val", model="churro",
             model_id="stanford-oval/churro-3B",
             infer=["--model", "churro"]),
        dict(run_id="churro3b-zeroshot-meta", split="val", model="churro",
             model_id="stanford-oval/churro-3B",
             infer=["--model", "churro", "--with-metadata"]),
        # Wave 1 VLMs: smoke then val for 3B; smoke-only first for 7B/8B.
        dict(run_id="smoke-qwen25vl-3b", split="smoke", model="qwen2.5-vl-3b",
             model_id="Qwen/Qwen2.5-VL-3B-Instruct",
             infer=["--model", "qwen2.5-vl-3b"]),
        dict(run_id="qwen25vl-3b-zeroshot", split="val", model="qwen2.5-vl-3b",
             model_id="Qwen/Qwen2.5-VL-3B-Instruct",
             infer=["--model", "qwen2.5-vl-3b"]),
        dict(run_id="smoke-qwen25vl-7b", split="smoke", model="qwen2.5-vl-7b",
             model_id="Qwen/Qwen2.5-VL-7B-Instruct",
             infer=["--model", "qwen2.5-vl-7b"]),
        dict(run_id="smoke-qwen3vl-8b", split="smoke", model="qwen3-vl-8b",
             model_id="Qwen/Qwen3-VL-8B-Instruct",
             infer=["--model", "qwen3-vl-8b"]),
        dict(run_id="smoke-olmocr-2-7b", split="smoke", model="olmocr-2-7b",
             model_id="allenai/olmOCR-2-7B-1025",
             infer=["--model", "olmocr-2-7b"]),
        dict(run_id="smoke-deepseek-ocr", split="smoke", model="deepseek-ocr",
             model_id="deepseek-ai/DeepSeek-OCR",
             infer=["--model", "deepseek-ocr"]),
    ]
    for job in jobs:
        infer_and_eval(job["run_id"], job["split"], job["infer"], job["model"], job["model_id"])

    # If a 7B/8B smoke succeeded, run the full holdout too.
    for run_id, model, model_id in [
        ("qwen25vl-7b-zeroshot", "qwen2.5-vl-7b", "Qwen/Qwen2.5-VL-7B-Instruct"),
        ("qwen3vl-8b-zeroshot", "qwen3-vl-8b", "Qwen/Qwen3-VL-8B-Instruct"),
        ("olmocr-2-7b-zeroshot", "olmocr-2-7b", "allenai/olmOCR-2-7B-1025"),
    ]:
        smoke_id = {
            "qwen25vl-7b-zeroshot": "smoke-qwen25vl-7b",
            "qwen3vl-8b-zeroshot": "smoke-qwen3vl-8b",
            "olmocr-2-7b-zeroshot": "smoke-olmocr-2-7b",
        }[run_id]
        preds = ROOT / "runs" / smoke_id / "preds.csv"
        if preds.exists() and preds.stat().st_size > 20:
            infer_and_eval(run_id, "val", ["--model", model], model, model_id)
        else:
            run([
                str(PY), str(EVAL), "--run-id", run_id, "--split", "val",
                "--record-skip", "skipped_after_smoke_fail", "--model", model,
                "--model-id", model_id, "--device", "mps", "--dtype", "fp16",
            ])

    # Pipeline on smoke then val.
    for split, run_id in [("smoke", "smoke-pipeline-trocr"), ("val", "pipeline-kraken-trocr")]:
        code = run([str(PY), str(PIPE), "--split", split, "--run-id", run_id])
        preds = ROOT / "runs" / run_id / "preds.csv"
        if code == 0 and preds.exists():
            run([str(PY), str(EVAL), "--run-id", run_id, "--split", split, "--append-results"])
        else:
            run([
                str(PY), str(EVAL), "--run-id", run_id, "--split", split,
                "--record-skip", "pipeline_failed", "--model", "pipeline-kraken-trocr",
                "--model-id", "dh-unibe/trocr-kurrent+microsoft/trocr-base-handwritten",
                "--device", "mps", "--dtype", "fp16",
            ])

    run([str(PY), str(EVAL), "--run-id", "churro3b-zeroshot", "--split", "val", "--write-table"])
    print("wave1 driver done", flush=True)


if __name__ == "__main__":
    main()
