"""Page-level VLM inference: image in, transcript out.

Device: cuda → mps → cpu. 4-bit is CUDA-only (bitsandbytes).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import image_path, page_ids, read_metadata, read_train_rows, runs_dir
from prompts import build_prompt

MODEL_ALIASES = {
    "churro": "stanford-oval/churro-3B",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "olmocr-2-7b": "allenai/olmOCR-2-7B-1025",
    "deepseek-ocr": "deepseek-ai/DeepSeek-OCR",
}

LARGE_MODELS = {
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "allenai/olmOCR-2-7B-1025",
}

DEEPSEEK_OCR_ID = "deepseek-ai/DeepSeek-OCR"

# DeepSeek-OCR is task-prompted, not instruction-following: the shared verbatim
# prompt makes it emit nothing at all, so it runs on its own canonical prompt.
DEEPSEEK_OCR_PROMPT = "Free OCR."


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS requested but not available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(requested: str, device: str, model_id: str) -> str:
    import torch

    if requested != "auto":
        if requested == "4bit" and device != "cuda":
            raise SystemExit("4-bit quantization is CUDA-only (bitsandbytes)")
        return requested
    if device == "cuda" and model_id in LARGE_MODELS:
        return "4bit"
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16"
    if device in {"cuda", "mps"}:
        return "fp16"
    return "fp32"


def torch_dtype(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def peak_memory_mb(device: str) -> float | None:
    import torch

    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    if device == "mps" and hasattr(torch, "mps"):
        allocated = torch.mps.current_allocated_memory()
        driver = torch.mps.driver_allocated_memory()
        return max(allocated, driver) / (1024 * 1024)
    return None


def reset_peak_memory(device: str) -> None:
    import torch

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def load_existing_preds(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["page_id"]: row["text"] for row in csv.DictReader(f)}


def write_preds(path: Path, page_order: list[str], preds: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["page_id", "text"])
        w.writeheader()
        for pid in page_order:
            if pid in preds:
                w.writerow({"page_id": pid, "text": preds[pid]})


def require_transformers_v4() -> None:
    """DeepSeek-OCR's remote code targets transformers 4.x and cannot run on 5.x.

    With use_mla=False its decoder borrows transformers' own LlamaAttention,
    whose contract changed in 5.x: rotary embeddings are now passed in by the
    parent model rather than computed from position_ids. Run this model under
    an overlay instead:

        uv run --with "transformers==4.46.3" python benchmark/infer_vlm.py ...
    """
    import transformers

    if int(transformers.__version__.split(".")[0]) >= 5:
        raise SystemExit(
            f"DeepSeek-OCR needs transformers 4.x (found {transformers.__version__}); "
            'rerun with: uv run --with "transformers==4.46.3" python benchmark/infer_vlm.py ...'
        )


def restore_v4_config_defaults(config) -> int:
    """Re-apply defaults that a transformers 4.x config __init__ would have set.

    In 5.x the base __init__ rebuilds the instance dict, so attributes a remote
    config assigns before calling super() are dropped and only config.json keys
    survive. Reading the defaults back off each __init__ signature restores them.
    """
    import inspect

    import transformers.configuration_utils as cfg_utils

    # 5.x renamed PretrainedConfig to PreTrainedConfig.
    base_config = getattr(cfg_utils, "PreTrainedConfig", None) or cfg_utils.PretrainedConfig

    restored = 0
    seen: set[int] = set()
    stack = [config]
    while stack:
        cfg = stack.pop()
        if id(cfg) in seen:
            continue
        seen.add(id(cfg))
        for klass in type(cfg).__mro__:
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            for name, param in inspect.signature(init).parameters.items():
                if param.default is not inspect.Parameter.empty and not hasattr(cfg, name):
                    setattr(cfg, name, param.default)
                    restored += 1
        stack.extend(v for v in vars(cfg).values() if isinstance(v, base_config))
    return restored


def restore_vision_position_ids(model) -> None:
    """The vision encoder registers position_ids as arange, but the checkpoint
    omits it and 5.x re-initialises missing buffers, leaving indices that send
    the position embedding lookup out of bounds."""
    import torch

    for module in model.modules():
        num_positions = getattr(module, "num_positions", None)
        if num_positions is None or not hasattr(module, "position_ids"):
            continue
        ids = torch.arange(num_positions, device=module.position_ids.device)
        module.position_ids = ids.expand((1, -1))


def load_deepseek_ocr(model_id: str, device: str):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if device != "cuda":
        raise SystemExit("DeepSeek-OCR requires CUDA; its infer() hardcodes .cuda()")

    require_transformers_v4()

    import transformers

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    restore_v4_config_defaults(config)

    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModel.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        **{dtype_kw: torch.bfloat16},
    )
    model = model.eval().cuda().to(torch.bfloat16)
    restore_vision_position_ids(model)
    return model, tokenizer


def load_model(model_id: str, device: str, dtype: str):
    import torch
    from transformers import AutoProcessor

    if model_id == DEEPSEEK_OCR_ID:
        return load_deepseek_ocr(model_id, device)

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if dtype == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
        kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        kwargs["dtype"] = torch_dtype(dtype)
        if device == "cuda":
            kwargs["device_map"] = "auto"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoVLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if dtype != "4bit" and device != "cuda":
        model.to(device)
    return model, processor


def move_inputs(inputs, device: str, model):
    model_dtype = getattr(model, "dtype", None)
    out = {}
    for k, v in inputs.items():
        if not hasattr(v, "to"):
            out[k] = v
            continue
        v = v.to(device)
        if model_dtype is not None and v.is_floating_point() and v.dtype != model_dtype:
            v = v.to(model_dtype)
        out[k] = v
    return out


def transcribe_deepseek_ocr(model, tokenizer, img_file: Path) -> str:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = model.infer(
            tokenizer,
            prompt=f"<image>\n{DEEPSEEK_OCR_PROMPT}",
            image_file=str(img_file),
            output_path=tmp,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
            eval_mode=True,
        )

    text = (out or "").strip()
    for stop in ("<|end\u2581of\u2581sentence|>", "<|end_of_sentence|>", "<\uff5cend\u2581of\u2581sentence\uff5c>"):
        if text.endswith(stop):
            text = text[: -len(stop)]
    return text.strip()


def transcribe_page(model, processor, img_file: Path, prompt: str, device: str,
                    max_new_tokens: int, max_pixels: int) -> str:
    import torch
    from PIL import Image

    if type(model).__name__ == "DeepseekOCRForCausalLM":
        return transcribe_deepseek_ocr(model, processor, img_file)

    image = Image.open(img_file).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image, "max_pixels": max_pixels},
            {"type": "text", "text": prompt},
        ],
    }]
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        process_vision_info = None

    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(messages)
            kwargs = {"text": [text], "images": image_inputs, "padding": True, "return_tensors": "pt"}
            if video_inputs:
                kwargs["videos"] = video_inputs
            inputs = processor(**kwargs)
        else:
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    else:
        inputs = processor(images=image, text=prompt, return_tensors="pt")

    inputs = move_inputs(inputs, device, model)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    in_len = inputs["input_ids"].shape[1]
    out_ids = generated[:, in_len:]
    text_out = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    return text_out.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF id or alias (churro, qwen2.5-vl-3b, ...)")
    p.add_argument("--split", default="smoke", choices=["smoke", "val", "train"])
    p.add_argument("--run-id", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32", "4bit"])
    p.add_argument("--with-metadata", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--limit", type=int, default=0, help="optional cap on number of pages")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_id = resolve_model(args.model)
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device, model_id)

    ids = page_ids(args.split)
    if args.limit:
        ids = ids[: args.limit]

    train_index = {r["page_id"]: r for r in read_train_rows()}
    metadata = read_metadata() if args.with_metadata else {}

    run_dir = runs_dir() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "preds.csv"
    log_path = run_dir / "pages.jsonl"
    config = {
        "run_id": args.run_id,
        "model": args.model,
        "model_id": model_id,
        "split": args.split,
        "device": device,
        "dtype": dtype,
        "with_metadata": args.with_metadata,
        "prompt_override": DEEPSEEK_OCR_PROMPT if model_id == DEEPSEEK_OCR_ID else None,
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "n_pages": len(ids),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(config, indent=2))
    reset_peak_memory(device)
    t_load = time.perf_counter()
    try:
        model, processor = load_model(model_id, device, dtype)
    except Exception as e:
        fail = {
            **config,
            "status": "load_failed",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
        (run_dir / "metrics.json").write_text(json.dumps(fail, indent=2) + "\n", encoding="utf-8")
        print(fail["error"], file=sys.stderr)
        raise SystemExit(2)

    load_s = time.perf_counter() - t_load
    print(f"loaded in {load_s:.1f}s  peak_mem_mb={peak_memory_mb(device)}")

    preds = load_existing_preds(pred_path) if args.resume else {}
    page_times: list[float] = []
    t0 = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as logf:
        for i, pid in enumerate(ids, start=1):
            if pid in preds:
                print(f"[{i}/{len(ids)}] {pid} skip (resume)")
                continue
            img = image_path(pid)
            if not img.exists():
                print(f"[{i}/{len(ids)}] {pid} MISSING IMAGE {img}", file=sys.stderr)
                preds[pid] = ""
                write_preds(pred_path, ids, preds)
                continue
            row = train_index.get(pid, {})
            meta = metadata.get(row.get("doc_id", ""), {}) if args.with_metadata else None
            prompt = build_prompt(meta, args.with_metadata)
            t_page = time.perf_counter()
            try:
                text = transcribe_page(
                    model, processor, img, prompt, device,
                    args.max_new_tokens, args.max_pixels,
                )
            except Exception as e:
                text = ""
                err = f"{type(e).__name__}: {e}"
                print(f"[{i}/{len(ids)}] {pid} FAIL {err}", file=sys.stderr)
                logf.write(json.dumps({"page_id": pid, "error": err}) + "\n")
                logf.flush()
                preds[pid] = text
                write_preds(pred_path, ids, preds)
                continue
            dt = time.perf_counter() - t_page
            page_times.append(dt)
            preds[pid] = text
            rec = {
                "page_id": pid,
                "sec": round(dt, 3),
                "chars": len(text),
                "peak_mem_mb": peak_memory_mb(device),
            }
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            write_preds(pred_path, ids, preds)
            print(f"[{i}/{len(ids)}] {pid} {dt:.1f}s {len(text)} chars")

    wall = time.perf_counter() - t0
    page_times_sorted = sorted(page_times)
    n = len(page_times_sorted)
    summary = {
        **config,
        "status": "ok",
        "load_sec": round(load_s, 3),
        "infer_wall_sec": round(wall, 3),
        "n_predicted": len(preds),
        "sec_per_page_mean": round(sum(page_times) / n, 3) if n else None,
        "sec_per_page_p50": page_times_sorted[n // 2] if n else None,
        "sec_per_page_p95": page_times_sorted[int(0.95 * (n - 1))] if n else None,
        "peak_mem_mb": peak_memory_mb(device),
        "preds_csv": str(pred_path),
    }
    (run_dir / "infer_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
