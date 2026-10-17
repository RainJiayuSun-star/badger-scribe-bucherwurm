"""Build a local HTML report for reviewing transcription errors page by page.

Each page is shown next to an inline diff of prediction against reference,
laid out on the reference's own line breaks so it can be read against the
image. Text the model missed is struck through in red; text it added is
green; `#` in the reference (unscored) is grey. The page list is sorted worst
first and can be filtered by category, label source and CER.

    python benchmark/error_report.py --preds runs/<run>/development/epoch-3-preds.csv
    python benchmark/error_report.py --preds 'runs/<run>/zero_shot/*-preds.csv' \\
        --baseline 'runs/<other>/zero_shot/*-preds.csv' --out runs/report.html

The report contains label text and points at local images, so write it under
runs/ (git-ignored) and do not commit or publish it.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import glob
import html
import importlib.util
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import data_root, image_path, read_train_rows, runs_dir  # noqa: E402

csv.field_size_limit(10**8)
_TAG_RE = re.compile(r"</?(?:u|del|ins)>")
# Below this similarity a replaced span is shown as one deletion and one insertion.
MIN_CHAR_SIMILARITY = 0.5


def load_metric():
    path = data_root() / "metric.py"
    spec = importlib.util.spec_from_file_location("badger_metric", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean(text: str) -> str:
    """The metric's normalization except that line breaks are kept for display."""
    text = _TAG_RE.sub("", text.replace("\\n", "\n"))
    text = unicodedata.normalize("NFC", text)
    lines = (" ".join(line.split()) for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def ref_tokens(text: str) -> list[tuple[str, bool]]:
    """Reference words, each with whether a line break follows it."""
    tokens = []
    for line in clean(text).split("\n"):
        words = line.split()
        tokens += [(word, index == len(words) - 1) for index, word in enumerate(words)]
    return tokens


def _ref_html(text: str) -> str:
    out = []
    for char in text:
        if char == "\n":
            out.append("<br>")
        elif char == "#":
            out.append('<span class="ill">#</span>')
        else:
            out.append(html.escape(char))
    return "".join(out)


def _join_ref(tokens: list[tuple[str, bool]]) -> str:
    return "".join(word + ("\n" if newline else " ") for word, newline in tokens)


def _char_diff(ref: str, pred: str) -> str:
    """Character-level diff of one replaced span. `ref` may contain line breaks."""
    flat = ref.replace("\n", " ")
    out = []
    matcher = difflib.SequenceMatcher(None, flat, pred, autojunk=False)
    if matcher.ratio() < MIN_CHAR_SIMILARITY:
        # Mostly different text: interleaved fragments are unreadable, so show both whole.
        trailing = "<br>" if ref.endswith("\n") else " "
        return f"<del>{_ref_html(ref.rstrip())}</del> <ins>{html.escape(pred.strip())}</ins>{trailing}"
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            out.append(_ref_html(ref[a0:a1]))
            continue
        if a1 > a0:
            segment = ref[a0:a1]
            if set(segment.strip()) <= {"#"}:
                out.append(_ref_html(segment))
            else:
                trailing = "<br>" if segment.endswith("\n") else ""
                out.append(f"<del>{_ref_html(segment.rstrip(chr(10)))}</del>{trailing}")
        if b1 > b0:
            out.append(f"<ins>{html.escape(pred[b0:b1])}</ins>")
    return "".join(out)


def diff_html(reference: str, prediction: str) -> str:
    """Word-aligned diff, refined to characters inside each replaced span."""
    ref = ref_tokens(reference)
    pred = clean(prediction).split()
    matcher = difflib.SequenceMatcher(None, [word for word, _ in ref], pred, autojunk=False)
    out = []
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        ref_text = _join_ref(ref[a0:a1])
        pred_text = " ".join(pred[b0:b1])
        if op == "equal":
            out.append(_ref_html(ref_text))
        elif op == "delete":
            trailing = "<br>" if ref_text.endswith("\n") else " "
            out.append(f"<del>{_ref_html(ref_text.rstrip())}</del>{trailing}")
        elif op == "insert":
            out.append(f"<ins>{html.escape(pred_text)}</ins> ")
        else:
            out.append(_char_diff(ref_text, pred_text + (" " if not ref_text.endswith("\n") else "")))
            if ref_text.endswith("\n") and not out[-1].endswith("<br>"):
                out.append("<br>")
    return re.sub(r"(\s|<br>)+$", "", "".join(out))


def read_preds(patterns: list[str]) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Predictions by page ID, plus loop-guard flags from any sibling *-scores.csv."""
    preds: dict[str, str] = {}
    flags: dict[str, dict[str, str]] = {}
    files = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not files:
        raise SystemExit(f"no prediction files match {patterns}")
    for path in files:
        if path.endswith("-scores.csv"):
            continue
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                preds[row["page_id"]] = row.get("text") or ""
        scores = Path(path).with_name(Path(path).stem + "-scores.csv")
        if scores.is_file():
            with scores.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    flags[row["page_id"]] = row
    return preds, flags


def build_pages(preds, flags, truth, metric, baseline=None, out_dir: Path | None = None) -> list[dict]:
    pages = []
    for page_id, prediction in sorted(preds.items()):
        row = truth.get(page_id)
        if row is None:
            continue    # test pages have no reference
        cer = metric.page_cer(prediction, row["text"])
        page = {
            "id": page_id,
            "doc": row["doc_id"],
            "cat": row["category"],
            "src": row["label_source"],
            "cer": round(cer, 4),
            "loop": flags.get(page_id, {}).get("loop_cut") in ("1", "True", "true"),
            "retry": flags.get(page_id, {}).get("retry_used") in ("1", "True", "true"),
            "ref_len": len(metric.normalize_text(row["text"])),
            "pred_len": len(metric.normalize_text(prediction)),
            "diff": diff_html(row["text"], prediction),
            "img": os.path.relpath(image_path(page_id), out_dir) if out_dir else str(image_path(page_id)),
        }
        if baseline is not None and page_id in baseline:
            page["base"] = round(metric.page_cer(baseline[page_id], row["text"]), 4)
        pages.append(page)
    pages.sort(key=lambda page: -page["cer"])
    return pages


TEMPLATE = Path(__file__).resolve().parent / "error_report.html"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds", action="append", required=True, help="Prediction CSV (page_id,text); globs and repeats allowed")
    parser.add_argument("--baseline", action="append", help="Optional second run to compare against, same format")
    parser.add_argument("--title", default="Error review")
    parser.add_argument("--out", type=Path, default=runs_dir() / "error_report.html")
    args = parser.parse_args()

    truth = {row["page_id"]: row for row in read_train_rows()}
    preds, flags = read_preds(args.preds)
    baseline = read_preds(args.baseline)[0] if args.baseline else None
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pages = build_pages(preds, flags, truth, load_metric(), baseline, out.parent)
    if not pages:
        raise SystemExit("none of the predicted pages are in train.csv")
    payload = json.dumps({"title": args.title, "pages": pages}, ensure_ascii=False).replace("</", "<\\/")
    page = TEMPLATE.read_text(encoding="utf-8").replace("__TITLE__", html.escape(args.title)).replace("__DATA__", payload)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {len(pages)} pages to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
