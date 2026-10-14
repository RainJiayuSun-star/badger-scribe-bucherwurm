"""Command-line parsing for the remote gateway baseline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pipeline.config import DEFAULT_BASE_URL, DEFAULT_MODEL, GatewaySettings
from pipeline.dataset import read_page_ids
from pipeline.gateway import GatewayBackend
from pipeline.outputs import read_completed, write_predictions
from pipeline.prompts import VERBATIM_V1, prompt_from_file
from pipeline.runner import run_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Badger Scribe pages through remote Churro-3B.")
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV containing a page_id column")
    parser.add_argument("--image-dir", type=Path, required=True, help="Page images or outer Kaggle images directory")
    parser.add_argument("--output-csv", type=Path, required=True, help="Kaggle-format output CSV; existing rows are resumed")
    parser.add_argument("--log-jsonl", type=Path, help="Append-only request log (default: next to output CSV)")
    parser.add_argument("--base-url", default=os.environ.get("BADGERCHAT_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", help="Optional text file replacing the versioned default prompt")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=360.0, help="Per-request timeout; cold starts can take a while")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Process only the first N unfinished pages")
    parser.add_argument("--page-id", action="append", dest="page_ids",
                        help="Process this page ID only; repeat to select multiple pages")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--continue", dest="continue_run", action="store_true",
        help="Resume unfinished pages from the existing output CSV (the default behavior)",
    )
    resume_group.add_argument(
        "--no-resume", action="store_true",
        help="Ignore rows already present in output CSV and start a fresh run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set. Load it in this shell, then rerun.", file=sys.stderr)
        return 2
    prompt = prompt_from_file(args.prompt_file) if args.prompt_file else VERBATIM_V1
    settings = GatewaySettings(args.base_url, args.model, args.max_tokens, args.timeout, args.retries)
    page_ids = read_page_ids(args.input_csv)
    predictions = {} if args.no_resume else read_completed(args.output_csv)
    pending = [page_id for page_id in page_ids if page_id not in predictions]
    if args.page_ids:
        unknown = [page_id for page_id in args.page_ids if page_id not in page_ids]
        if unknown:
            print(f"page ID not found in {args.input_csv}: {', '.join(unknown)}", file=sys.stderr)
            return 2
        wanted = set(args.page_ids)
        pending = [page_id for page_id in pending if page_id in wanted]
    elif args.limit is not None:
        pending = pending[:args.limit]
    log_path = args.log_jsonl or args.output_csv.with_suffix(".requests.jsonl")
    print(f"{len(page_ids)} pages; {len(predictions)} already saved; {len(pending)} to request; prompt={prompt.name}")
    run_pages(backend=GatewayBackend(settings, api_key, prompt), page_ids=pending, all_page_ids=page_ids, predictions=predictions, image_dir=args.image_dir, output_csv=args.output_csv, log_jsonl=log_path, prompt=prompt, model=settings.model, base_url=settings.base_url)
    write_predictions(args.output_csv, predictions, page_ids)
    missing = [page_id for page_id in page_ids if page_id not in predictions]
    print(f"Wrote {len(predictions)}/{len(page_ids)} predictions to {args.output_csv}")
    return 1 if missing else 0
