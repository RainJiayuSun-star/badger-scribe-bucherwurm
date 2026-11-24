"""Zero-shot screen of a gateway-served model on held-out training pages. Test pages are refused.

    OPENAI_API_KEY=... python -m finetune.screen_gateway --run-id screen-qwen38-gateway-zs --runs /path/to/runs

Pages get the same prompt, image budget and loop guard as finetune.train --eval-only, and the same
score files as finetune.screen_base, so runs/analysis/paired_compare.py can pair them with earlier
screens. Every page is saved as it finishes, so rerunning the same command continues after a stop.
"""

from __future__ import annotations

import argparse
import base64
import csv
import http.client
import io
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image
from qwen_vl_utils import smart_resize

from benchmark.common import load_metric
from pipeline.config import DEFAULT_BASE_URL
from pipeline.gateway import IncompleteResponse, parse_stream
from pipeline.outputs import append_log, read_completed, write_predictions

from .config import ROOT, default_data_root, load_config
from .data import fold_partition, image_path, read_rows
from .loopguard import cut_loops
from .vlm import PROMPT, max_pixels_for

FOLDS = [f"dominy_group{n}" for n in range(1, 7)] + ["survey_011", "kade_112"]
RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


def refuse_test_pages(page_ids: set[str], data_root: Path) -> None:
    with (data_root / "test.csv").open(newline="", encoding="utf-8-sig") as handle:
        overlap = page_ids & {row["page_id"] for row in csv.DictReader(handle)}
    if overlap:
        raise SystemExit(f"refusing to send {len(overlap)} test pages to the gateway")


def page_url(path: Path, max_pixels: int, factor: int = 32) -> tuple[str, list[int]]:
    """The page at the harness's image budget, so the server does not choose the resolution."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    height, width = smart_resize(image.height, image.width, factor=factor, max_pixels=max_pixels)
    buffer = io.BytesIO()
    image.resize((width, height), Image.BICUBIC).save(buffer, "JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"), [width, height]


def decode(url: str, model: str, key: str, max_tokens: int, retries: int = 8) -> tuple[str, dict[str, Any]]:
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens, "stream": True,
                       "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                                 {"type": "text", "text": PROMPT}]}]}).encode("utf-8")
    for attempt in range(retries + 1):
        wait = min(2 ** attempt, 60)
        request = urllib.request.Request(DEFAULT_BASE_URL + "/chat/completions", data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "text/event-stream"})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                text, usage, served, finish = parse_stream(response)
            return text, {"usage": usage, "served_model": served, "finish_reason": finish, "attempt": attempt + 1}
        except urllib.error.HTTPError as error:
            if error.code not in RETRYABLE or attempt == retries:
                raise RuntimeError(f"gateway HTTP {error.code}: {error.read()[:300].decode('utf-8', 'replace')}") from error
            wait = max(wait, int(error.headers.get("Retry-After") or 0))  # a shared pilot: back off when asked
        except (IncompleteResponse, OSError, http.client.HTTPException) as error:
            if attempt == retries:
                raise RuntimeError(f"gateway request failed: {error}") from error
        time.sleep(wait)
    raise AssertionError("unreachable")


def score_fold(rows: list[dict[str, str]], raw: dict[str, str], metric, out: Path, fold: str) -> dict[str, Any]:
    order = [row["page_id"] for row in rows]
    cut = {page_id: cut_loops(raw[page_id]) for page_id in order}
    write_predictions(out / f"{fold}-preds.csv", {page_id: cut[page_id][0] for page_id in order}, order)
    scores = [{"page_id": row["page_id"], "doc_id": row["doc_id"], "category": row["category"], "label_source": row["label_source"],
               "cer": round(float(metric.page_cer(cut[row["page_id"]][0], row["text"])), 4),
               "length_ratio": round(len(cut[row["page_id"]][0]) / max(1, len(row["text"])), 3),
               "loop_cut": int(cut[row["page_id"]][1]), "retry_used": 0} for row in rows]
    with (out / f"{fold}-preds-scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    return {"pages": len(rows), "category": rows[0]["category"], "loop_cuts": sum(s["loop_cut"] for s in scores),
            "cer": statistics.fmean(float(metric.page_cer(cut[row["page_id"]][0], row["text"])) for row in rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--config", type=Path, default=ROOT / "finetune" / "configs" / "qwen38-27b-4bit-english.yaml")
    parser.add_argument("--fold", action="append", help="Fold name; default: the 8 screen folds")
    parser.add_argument("--workers", type=int, default=3, help="Parallel requests; the gateway is a shared pilot")
    parser.add_argument("--limit", type=int, help="Decode only this many pages per fold (smoke test; no scores)")
    args = parser.parse_args()
    key = os.environ.get("OPENAI_API_KEY") or parser.error("OPENAI_API_KEY is not set")
    data_root, config = default_data_root(), load_config(args.config)
    rows = read_rows(data_root)
    folds = {name: [dict(row) for row in fold_partition(rows, args.runs / "folds" / f"{name}.json")["development"]] for name in args.fold or FOLDS}
    refuse_test_pages({row["page_id"] for fold_rows in folds.values() for row in fold_rows}, data_root)

    run_dir = args.runs / "finetune" / args.run_id
    out = run_dir / "zero_shot"
    out.mkdir(parents=True, exist_ok=True)
    settings = json.dumps({"model": args.model, "gateway": DEFAULT_BASE_URL, "prompt": PROMPT, "temperature": 0, "enable_thinking": False,
                           "max_tokens": int(config["eval_max_new_tokens"]), "max_pixels": config["max_pixels"],
                           "max_pixels_by_category": config["max_pixels_by_category"], "folds": list(folds)}, indent=2, sort_keys=True) + "\n"
    if (run_dir / "config.json").is_file() and (run_dir / "config.json").read_text(encoding="utf-8") != settings:
        raise SystemExit(f"{run_dir} was started with different settings. Use a different --run-id.")
    (run_dir / "config.json").write_text(settings, encoding="utf-8")

    raw = {name: read_completed(out / f"{name}-raw.csv") for name in folds}
    lock, started = threading.Lock(), time.monotonic()
    pending = [(name, row) for name, fold_rows in folds.items() for row in fold_rows[: args.limit] if row["page_id"] not in raw[name]]
    total, done = len(pending), [0]

    def run_page(job: tuple[str, dict[str, str]]) -> None:
        name, row = job
        begun = time.monotonic()
        url, size = page_url(image_path(data_root, row["page_id"]), max_pixels_for(config, row["category"]))
        text, info = decode(url, args.model, key, int(config["eval_max_new_tokens"]))
        with lock:
            append_log(run_dir / "pages.jsonl", {"fold": name, "page_id": row["page_id"], "seconds": round(time.monotonic() - begun, 2),
                                                 "image": size, "raw_chars": len(text), **info})
            raw[name][row["page_id"]] = text
            write_predictions(out / f"{name}-raw.csv", raw[name], [r["page_id"] for r in folds[name] if r["page_id"] in raw[name]])
            done[0] += 1
            print(f"{done[0]}/{total} pages, {time.monotonic() - started:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(run_page, pending):
            pass
    if args.limit:
        return 0

    metric = load_metric(data_root)
    summary = {name: score_fold(fold_rows, raw[name], metric, out, name) for name, fold_rows in folds.items()}
    by_category: dict[str, list[float]] = {}
    for result in summary.values():
        by_category.setdefault(result["category"], []).extend([result["cer"]] * result["pages"])
    report = {"run_id": args.run_id, "model": args.model, "folds": summary, "pages": sum(r["pages"] for r in summary.values()),
              "cer_by_category": {category: statistics.fmean(values) for category, values in by_category.items()},
              "loop_cuts": sum(r["loop_cuts"] for r in summary.values()), "wall_seconds": round(time.monotonic() - started, 1)}
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key_: report[key_] for key_ in ("pages", "cer_by_category", "loop_cuts", "wall_seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
