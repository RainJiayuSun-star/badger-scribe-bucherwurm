"""Transcribe pages with a fine-tuned adapter per category.

    python -m finetune.predict --config finetune/configs/churro-3b-bf16.yaml --run-id submission-01 \\
      --adapter kade_letters=<german adapter> \\
      --adapter survey_notes=<english adapter> \\
      --adapter dominy_accounts=<english adapter>

Add --base CATEGORY=CONFIG to decode a category with another config's base model, prompt and
image settings (e.g. olmOCR for the account books). Each base is loaded in turn and unloaded
before the next, so one GPU only ever holds one model.

Writes to runs/finetune/<run-id>/. Every page is saved as it finishes, so rerunning
the same command after a crash continues where it stopped. submission.csv appears
only once every page is there. Add --fold to score a held-out training document
instead of predicting the test set, which proves the adapter loaded.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from benchmark.common import load_metric
from pipeline.dataset import read_page_ids
from pipeline.outputs import append_log, read_completed, write_predictions

from .config import ROOT, load_config, resolved_config_json
from .data import CATEGORIES, fold_partition, image_path, read_rows
from .preflight import run_checks
from .runstate import Heartbeat, RunState, gib
from .status import read_events
from .vlm import generate_texts, load_model_and_processor

# All that inference may see of a page. The rules forbid behaving differently on label_source.
PAGE_FIELDS = ("page_id", "doc_id", "category")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="Rerun with the same ID to continue after a crash")
    parser.add_argument("--adapter", action="append", required=True, metavar="CATEGORY=PATH", help="best_adapter folder for a category; repeat for each category")
    parser.add_argument("--base", action="append", default=[], metavar="CATEGORY=CONFIG", help="decode a category with this config's base model instead of --config's; repeat for each category")
    parser.add_argument("--fold", type=Path, help="Fold file from finetune.folds: score its held-out training pages instead of predicting the test set")
    return parser.parse_args()


def parse_adapters(items: list[str], flag: str = "--adapter") -> dict[str, Path]:
    adapters: dict[str, Path] = {}
    for item in items:
        category, _, path = item.partition("=")
        if category not in CATEGORIES or not path:
            raise SystemExit(f"{flag} {item!r} must be CATEGORY=PATH, with CATEGORY one of {', '.join(CATEGORIES)}")
        if category in adapters:
            raise SystemExit(f"{flag} names {category} twice")
        adapters[category] = Path(path).expanduser().resolve()
    return adapters


def check_inputs(data_root: Path, adapters: dict[str, Path], fold: Path | None) -> None:
    """Preflight checks specific to prediction; finetune.preflight covers the rest."""
    problems = []
    if fold:
        if not fold.is_file():
            problems.append(f"{fold} is missing. Build the fold files with python -m finetune.folds.")
    else:
        for name in ("test.csv", "sample_submission.csv"):
            if not (data_root / name).is_file():
                problems.append(f"{data_root / name} is missing. Put the competition data in {data_root} or set BADGER_SCRIBE_DATA.")
    for category, path in adapters.items():
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (path / name).is_file():
                problems.append(f"The {category} adapter has no {name} in {path}. Point --adapter at a best_adapter folder.")
    if problems:
        raise SystemExit("Preflight checks failed; nothing was started:\n" + "\n".join(f"  - {p}" for p in problems))


def test_pages(data_root: Path) -> list[dict[str, str]]:
    """Test pages in sample_submission order, without label_source."""
    order = read_page_ids(data_root / "sample_submission.csv")
    with (data_root / "test.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["page_id"]: row for row in csv.DictReader(handle)}
    if set(rows) != set(order):
        raise SystemExit(f"test.csv lists {len(rows)} pages and sample_submission.csv {len(order)}, and they don't match")
    return [{key: rows[page_id][key] for key in PAGE_FIELDS} for page_id in order]


def check_predictions(path: Path, order: list[str]) -> list[str]:
    """Stop unless the file has every page once, in order. Returns the pages with no text.

    Kaggle scores a missing page as CER 1.0 without complaint, so nothing later would catch one.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    found = [row["page_id"] for row in rows]
    if found != order:
        missing = [page_id for page_id in order if page_id not in set(found)]
        raise SystemExit(f"{path} has {len(found)} rows, not {len(order)} pages once each in order. Missing {len(missing)}, starting {missing[:5]}")
    return [row["page_id"] for row in rows if not row["text"].strip()]


def read_page_log(path: Path) -> list[dict[str, Any]]:
    records = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue    # a line cut off by a crash
    return records


def git_version() -> str | None:
    """The commit, with -dirty if tracked files differ from it."""
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=40"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    args = parse_args()
    holder: dict[str, RunState] = {}
    try:
        return run(args, holder)
    except BaseException as error:
        state = holder.get("state")
        # A shutdown signal already logged itself; that is not a crash.
        stopped = state and state.status.get("phase") == "stopping"
        if state and not stopped and not (isinstance(error, SystemExit) and error.code in (0, None)):
            state.failed(error)
        raise


def run(args: argparse.Namespace, holder: dict[str, RunState]) -> int:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Run this command on the A5000 host.")
    config = load_config(args.config)
    data_root = Path(config["data_root"])
    run_dir = Path(config["runs_root"]) / args.run_id
    adapters = parse_adapters(args.adapter)
    bases = {}
    for category, path in parse_adapters(args.base, "--base").items():
        if not path.is_file():
            raise SystemExit(f"--base {category}: {path} is missing; nothing was started.")
        # Only the model side comes from the base config; data and run folders stay --config's.
        bases[category] = {**load_config(path), "data_root": config["data_root"], "runs_root": config["runs_root"]}
    # Every base must fit when it loads, so check the largest free-GPU need up front.
    run_checks({**config, "min_free_gpu_gib": max(float(c["min_free_gpu_gib"]) for c in [config, *bases.values()])},
               run_dir, resume=True, eval_only=False)
    check_inputs(data_root, adapters, args.fold)

    references: dict[str, str] = {}
    if args.fold:
        development = fold_partition(read_rows(data_root), args.fold)["development"]
        references = {row["page_id"]: row["text"] for row in development}
        pages = [{key: row[key] for key in PAGE_FIELDS} for row in development]
    else:
        pages = test_pages(data_root)
    categories = {page["category"] for page in pages}
    if categories - set(adapters):
        raise SystemExit(f"No --adapter for {', '.join(sorted(categories - set(adapters)))}; nothing was started.")
    adapters = {category: adapters[category] for category in sorted(categories)}
    model_configs = {category: bases.get(category, config) for category in adapters}
    # One group per base model, --config's first; each loads once.
    groups: dict[str, list[str]] = {}
    for category in sorted(adapters, key=lambda c: c in bases):
        groups.setdefault(model_configs[category]["_config_path"], []).append(category)
    for page in pages:
        page["image_path"] = str(image_path(data_root, page["page_id"]))
    order = [page["page_id"] for page in pages]

    # Continuing with other settings would mix two models in one submission.
    settings = resolved_config_json({**config, "adapters": adapters, "fold": args.fold.resolve() if args.fold else None,
                                     **({"bases": {category: bases[category] for category in adapters if category in bases}} if bases else {})})
    settings_path = run_dir / "config.json"
    if settings_path.is_file() and settings_path.read_text(encoding="utf-8") != settings:
        raise SystemExit(f"{run_dir} was started with different settings (see its config.json). Use a different --run-id.")
    run_dir.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(settings, encoding="utf-8")

    state = RunState(run_dir)
    holder["state"] = state
    state.catch_signals()
    progress_path = run_dir / "predictions.csv"
    page_log_path = run_dir / "pages.jsonl"
    predictions = read_completed(progress_path)
    pending = [page for page in pages if page["page_id"] not in predictions]
    heartbeat = Heartbeat(state) if pending else None
    for group in groups.values():
        todo = [page for page in pending if page["category"] in group]
        if not todo:
            continue
        model_config = model_configs[group[0]]
        started = time.monotonic()
        state.phase("loading_model", run_id=args.run_id, fold=args.fold.stem if args.fold else None, pending_pages=len(pending), model_id=model_config["model_id"])
        model, processor = load_model_and_processor(model_config, 0)
        # shortcut: English loads once per category (about 0.1 GB extra); key by path if memory ever matters.
        for category in group:
            path = adapters[category]
            result = model.load_adapter(str(path), adapter_name=category, is_trainable=False)
            # PEFT only warns here, and the page would then decode with untrained weights.
            if result.missing_keys:
                raise SystemExit(f"{path} is missing {len(result.missing_keys)} tensors this model needs, e.g. {result.missing_keys[0]}")
        # After the loads: each new adapter's dropout starts on, and generate_texts doesn't turn it off.
        model.eval()
        state.event("model_loaded", seconds=round(time.monotonic() - started, 1), model_id=model_config["model_id"])
        state.phase("evaluating", eval_page=len(predictions), eval_pages=len(pages))
        for page in todo:
            model.set_adapter(page["category"])
            started = time.monotonic()
            texts, infos = generate_texts(model, processor, [page], "cuda:0", int(model_config.get("eval_max_new_tokens", 2048)), model_config)
            predictions[page["page_id"]] = texts[0]
            write_predictions(progress_path, predictions, order)
            append_log(page_log_path, {"page_id": page["page_id"], "category": page["category"], "seconds": round(time.monotonic() - started, 2),
                                       "chars": len(texts[0]), **infos[0],
                                       # Reserved, not allocated: the submission card asks what the run actually used.
                                       "peak_gpu_gib": gib(torch.cuda.max_memory_reserved())})
            heartbeat(eval_page=len(predictions), eval_pages=len(pages))
        # Free this base before the next loads (the rules allow several models, one at a time).
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    empty = check_predictions(progress_path, order)
    if empty:
        print(f"WARNING: {len(empty)} pages have no text and will score CER 1.0: {empty}", flush=True)
    records = read_page_log(page_log_path)
    loop_cuts = sum(record["loop_cut"] for record in records)
    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "fold": args.fold.stem if args.fold else None,
        "pages": len(pages),
        "empty_pages": empty,
        "loop_cuts": loop_cuts,
        # Model loading plus decoding, added up across restarts, for eval_wall_clock on the card.
        "wall_seconds": round(sum(record["seconds"] for record in records)
                              + sum(event["seconds"] for event in read_events(run_dir) if event["event"] == "model_loaded"), 1),
        "peak_gpu_gib": max((record["peak_gpu_gib"] for record in records), default=None),
        "git_version": git_version(),
        "model_id": config["model_id"],
        "model_ids": {category: model_configs[category]["model_id"] for category in adapters},
        "adapters": {category: str(path) for category, path in adapters.items()},
    }
    if args.fold:
        metric = load_metric(data_root)
        by_category: dict[str, list[float]] = defaultdict(list)
        for page in pages:
            by_category[page["category"]].append(float(metric.page_cer(predictions[page["page_id"]], references[page["page_id"]])))
        summary["cer_by_category"] = {category: mean(values) for category, values in by_category.items()}
        summary["cer"] = mean(summary["cer_by_category"].values())
        state.event("evaluated", fold=args.fold.stem, cer=summary["cer"], loop_cuts=loop_cuts)
        print(f"{args.fold.stem}: CER {summary['cer']:.4f} over {len(pages)} pages", flush=True)
    else:
        write_predictions(run_dir / "submission.csv", predictions, order)
        print(f"{len(pages)} pages written to {run_dir / 'submission.csv'}", flush=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state.phase("finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
