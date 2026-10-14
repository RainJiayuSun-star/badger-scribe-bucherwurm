"""Line-level HTR pipeline: segment, recognize, assemble.

Tries Kraken for segmentation when installed; otherwise a projection-profile
fallback. Recognizers: dh-unibe/trocr-kurrent on German (kade_letters),
microsoft/trocr-base-handwritten on English.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import image_path, page_ids, read_train_rows, runs_dir, splits_dir

GERMAN_RECOGNIZER = "dh-unibe/trocr-kurrent"
ENGLISH_RECOGNIZER = "microsoft/trocr-base-handwritten"
GERMAN_FALLBACKS = [
    "fhswf/TrOCR-kurrent",
    "microsoft/trocr-base-handwritten",
]


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    thresh = 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > max_var:
            max_var = var
            thresh = t
    return int(thresh)


def binarize(img: Image.Image) -> np.ndarray:
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    arr = np.array(gray, dtype=np.uint8)
    t = otsu_threshold(arr)
    ink_dark = arr.mean() > t
    if ink_dark:
        binary = arr < t
    else:
        binary = arr > t
    return binary.astype(np.uint8)


def projection_lines(binary: np.ndarray, min_height: int = 8) -> list[tuple[int, int]]:
    row_ink = binary.sum(axis=1)
    thresh = max(2, int(0.015 * binary.shape[1]))
    mask = row_ink > thresh
    spans = []
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= min_height:
                spans.append((start, i))
            start = None
    if start is not None and binary.shape[0] - start >= min_height:
        spans.append((start, binary.shape[0]))
    # merge nearby fragments
    merged = []
    for s, e in spans:
        if merged and s - merged[-1][1] < 6:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def crop_line(img: Image.Image, binary: np.ndarray, y0: int, y1: int, pad: int = 4) -> Image.Image | None:
    h, w = binary.shape
    y0p = max(0, y0 - pad)
    y1p = min(h, y1 + pad)
    band = binary[y0p:y1p]
    if band.sum() < 20:
        return None
    cols = band.sum(axis=0)
    xs = np.where(cols > 0)[0]
    if len(xs) == 0:
        return None
    x0 = max(0, int(xs[0]) - pad)
    x1 = min(w, int(xs[-1]) + pad)
    crop = img.crop((x0, y0p, x1, y1p)).convert("RGB")
    if crop.size[0] < 8 or crop.size[1] < 8:
        return None
    return crop


def kraken_lines(img_file: Path) -> list[Image.Image] | None:
    try:
        from kraken import binarization, pageseg
        from kraken.lib.util import pil2array
    except Exception:
        return None
    im = Image.open(img_file)
    try:
        bw = binarization.nlbin(im)
        seg = pageseg.segment(bw)
    except Exception:
        return None
    boxes = []
    for line in getattr(seg, "lines", []) or []:
        box = getattr(line, "bbox", None) or getattr(line, "boundary", None)
        if box is None:
            continue
        if hasattr(box, "xmin"):
            boxes.append((box.xmin, box.ymin, box.xmax, box.ymax))
        elif isinstance(box, (list, tuple)) and len(box) == 4:
            boxes.append(tuple(box))
    if not boxes:
        return None
    rgb = im.convert("RGB")
    crops = []
    for x0, y0, x1, y1 in sorted(boxes, key=lambda b: (b[1], b[0])):
        crop = rgb.crop((int(x0), int(y0), int(x1), int(y1)))
        if crop.size[0] >= 8 and crop.size[1] >= 8:
            crops.append(crop)
    return crops or None


def segment_page(img_file: Path) -> tuple[list[Image.Image], str]:
    kr = kraken_lines(img_file)
    if kr:
        return kr, "kraken"
    img = Image.open(img_file).convert("RGB")
    binary = binarize(img)
    spans = projection_lines(binary)
    crops = []
    for y0, y1 in spans:
        crop = crop_line(img, binary, y0, y1)
        if crop is not None:
            crops.append(crop)
    if not crops:
        crops = [img]
    return crops, "projection"


def load_processor(model_id: str):
    """Repos without a tokenizer.json break AutoTokenizer on transformers 5.x,
    which dropped slow-tokenizer conversion; the fast class still reads the
    vocab.json/merges.txt pair directly."""
    from transformers import AutoImageProcessor, RobertaTokenizerFast, TrOCRProcessor

    try:
        return TrOCRProcessor.from_pretrained(model_id)
    except Exception:
        return TrOCRProcessor(
            image_processor=AutoImageProcessor.from_pretrained(model_id),
            tokenizer=RobertaTokenizerFast.from_pretrained(model_id),
        )


def load_recognizer(model_id: str, device: str, fallbacks: list[str] | None = None):
    from transformers import VisionEncoderDecoderModel

    candidates = [model_id] + (fallbacks or [])
    last_err = None
    for mid in candidates:
        try:
            processor = load_processor(mid)
            model = VisionEncoderDecoderModel.from_pretrained(mid)
            model.to(device)
            model.eval()
            return processor, model, mid
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"could not load recognizer {model_id}: {last_err}")


def recognize_line(processor, model, crop: Image.Image, device: str) -> str:
    import torch

    pixel_values = processor(images=crop, return_tensors="pt").pixel_values.to(device)
    with torch.inference_mode():
        ids = model.generate(pixel_values, max_new_tokens=128)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def category_of(pid: str, train_index: dict[str, dict]) -> str:
    row = train_index.get(pid)
    if row:
        return row["category"]
    if pid.startswith("kade_"):
        return "kade_letters"
    if pid.startswith("dominy_"):
        return "dominy_accounts"
    return "survey_notes"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="smoke", choices=["smoke", "val", "train"])
    p.add_argument("--run-id", default="pipeline-kraken-trocr")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    ids = page_ids(args.split)
    if args.limit:
        ids = ids[: args.limit]
    train_index = {r["page_id"]: r for r in read_train_rows()}

    run_dir = runs_dir() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "preds.csv"
    log_path = run_dir / "pages.jsonl"
    config = {
        "run_id": args.run_id,
        "model": "pipeline-kraken-trocr",
        "model_id": f"{GERMAN_RECOGNIZER}+{ENGLISH_RECOGNIZER}",
        "split": args.split,
        "device": device,
        "dtype": "fp32" if device == "cpu" else "fp16",
        "with_metadata": False,
        "n_pages": len(ids),
        "family": "pipeline",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2))

    t_load = time.perf_counter()
    de_proc, de_model, de_id = load_recognizer(GERMAN_RECOGNIZER, device, GERMAN_FALLBACKS)
    en_proc, en_model, en_id = load_recognizer(ENGLISH_RECOGNIZER, device)
    load_s = time.perf_counter() - t_load
    config["german_recognizer"] = de_id
    config["english_recognizer"] = en_id
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"recognizers de={de_id} en={en_id} loaded in {load_s:.1f}s")

    preds = {}
    if args.resume and pred_path.exists():
        with pred_path.open(newline="", encoding="utf-8") as f:
            preds = {r["page_id"]: r["text"] for r in csv.DictReader(f)}

    def write_preds():
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["page_id", "text"])
            w.writeheader()
            for pid in ids:
                if pid in preds:
                    w.writerow({"page_id": pid, "text": preds[pid]})

    page_times = []
    seg_stats = {"kraken": 0, "projection": 0}
    t0 = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as logf:
        for i, pid in enumerate(ids, start=1):
            if pid in preds:
                print(f"[{i}/{len(ids)}] {pid} skip (resume)")
                continue
            img = image_path(pid)
            t_page = time.perf_counter()
            crops, seg = segment_page(img)
            seg_stats[seg] = seg_stats.get(seg, 0) + 1
            cat = category_of(pid, train_index)
            if cat == "kade_letters":
                proc, model = de_proc, de_model
            else:
                proc, model = en_proc, en_model
            lines = []
            for crop in crops:
                try:
                    lines.append(recognize_line(proc, model, crop, device))
                except Exception as e:
                    lines.append("")
                    logf.write(json.dumps({"page_id": pid, "line_error": str(e)}) + "\n")
            text = "\n".join(t for t in lines if t)
            dt = time.perf_counter() - t_page
            page_times.append(dt)
            preds[pid] = text
            rec = {"page_id": pid, "sec": round(dt, 3), "n_lines": len(crops), "seg": seg, "chars": len(text)}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            write_preds()
            print(f"[{i}/{len(ids)}] {pid} {seg} {len(crops)} lines {dt:.1f}s {len(text)} chars")

    n = len(page_times)
    st = sorted(page_times)
    summary = {
        **config,
        "status": "ok",
        "load_sec": round(load_s, 3),
        "infer_wall_sec": round(time.perf_counter() - t0, 3),
        "n_predicted": len(preds),
        "sec_per_page_mean": round(sum(page_times) / n, 3) if n else None,
        "sec_per_page_p50": st[n // 2] if n else None,
        "sec_per_page_p95": st[int(0.95 * (n - 1))] if n else None,
        "peak_mem_mb": None,
        "seg_stats": seg_stats,
        "preds_csv": str(pred_path),
    }
    (run_dir / "infer_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
