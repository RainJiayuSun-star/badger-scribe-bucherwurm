"""Qwen-family page-transcription model and batch helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

PROMPT = "Transcribe every visible text character exactly as written. Preserve spelling, case, punctuation, and reading order. Output only the transcription."


def load_model_and_processor(config: dict[str, Any], local_rank: int):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig

    # The auto class resolves Qwen2.5-VL (Churro, olmOCR) and Qwen3-VL alike.
    from transformers import AutoModelForImageTextToText as ModelClass

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    mode = str(config.get("quantization", "4bit")).lower()
    if mode not in {"4bit", "none"}:
        raise ValueError(f"quantization must be '4bit' or 'none', got {mode!r}")
    extra = {}
    if mode == "4bit":
        extra["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    processor = AutoProcessor.from_pretrained(config["model_id"], trust_remote_code=True)
    model = ModelClass.from_pretrained(
        config["model_id"],
        dtype=dtype,
        device_map={"": local_rank},
        trust_remote_code=True,
        **extra,
    )
    model.config.use_cache = False
    if mode == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    language_suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        suffix = name.rsplit(".", 1)[-1]
        is_visual = name.startswith("visual.") or ".visual." in name
        is_projector = "merger" in name or "projector" in name
        if suffix in language_suffixes and not is_visual:
            targets.append(name)
        elif is_projector:
            targets.append(name)
        elif config.get("adapter_policy") == "vision_inclusive" and is_visual and suffix in {"qkv", "proj"}:
            targets.append(name)
    targets = sorted(set(targets))
    if not targets:
        raise RuntimeError("no LoRA target modules found; inspect this model's module names")
    lora = LoraConfig(
        r=int(config.get("lora_rank", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    return model, processor


def max_pixels_for(config: dict[str, Any], category: str | None) -> int:
    by_category = config.get("max_pixels_by_category") or {}
    return int(by_category.get(category, config.get("max_pixels", 1280 * 28 * 28)))


def messages(image: str | Path | Image.Image, text: str | None = None, max_pixels: int | None = None) -> list[dict[str, Any]]:
    image_value = str(image) if isinstance(image, Path) else image
    image_item: dict[str, Any] = {"type": "image", "image": image_value}
    if max_pixels:
        image_item["max_pixels"] = int(max_pixels)
    content: list[dict[str, Any]] = [image_item, {"type": "text", "text": PROMPT}]
    result = [{"role": "user", "content": content}]
    if text is not None:
        result.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    return result


def _vision_inputs(message_batch: list[list[dict[str, Any]]], patch_size: int = 14):
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install qwen-vl-utils to train Qwen-family VLMs") from exc
    images, videos = process_vision_info(message_batch, image_patch_size=patch_size)
    return images, videos


def mask_illegible(labels, tokenizer) -> int:
    """Set the label of every target token containing `#` to -100. Returns how many."""
    masked = 0
    for index, token_id in enumerate(labels.view(-1).tolist()):
        if token_id != -100 and "#" in tokenizer.decode([token_id]):
            labels.view(-1)[index] = -100
            masked += 1
    return masked


def collate_train(rows: list[dict[str, Any]], processor, device, epoch: int, augmenter, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build labels that score assistant tokens only, not the instruction."""
    config = config or {}
    patch = int(config.get("image_patch_size", 14))
    materialized = []
    for row in rows:
        with Image.open(row["image_path"]) as source:
            image = augmenter.apply(source, row["page_id"], epoch) if augmenter else source.convert("RGB")
        materialized.append((row, image, max_pixels_for(config, row.get("category"))))
    # Train on the cleaned target when finetune.prep supplied one.
    full_messages = [messages(image, row.get("target", row["text"]), pixels) for row, image, pixels in materialized]
    prompt_messages = [messages(image, None, pixels) for _, image, pixels in materialized]
    full_texts = [processor.apply_chat_template(item, tokenize=False, add_generation_prompt=False) for item in full_messages]
    prompt_texts = [processor.apply_chat_template(item, tokenize=False, add_generation_prompt=True) for item in prompt_messages]
    images, videos = _vision_inputs(full_messages, patch)
    batch = processor(text=full_texts, images=images, videos=videos, padding=True, return_tensors="pt")
    prompt_images, prompt_videos = _vision_inputs(prompt_messages, patch)
    prompts = processor(text=prompt_texts, images=prompt_images, videos=prompt_videos, padding=True, return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    prompt_lengths = prompts["attention_mask"].sum(dim=1).tolist()
    for index, length in enumerate(prompt_lengths):
        labels[index, : int(length)] = -100
    if config.get("mask_illegible", True):
        mask_illegible(labels, processor.tokenizer)
    batch["labels"] = labels
    max_len = int(config.get("max_seq_length", 0) or 0)
    if max_len and batch["input_ids"].shape[1] > max_len:
        # Image tokens come first, so cutting the tail drops only target text.
        prompt_max = max(int(length) for length in prompt_lengths)
        if prompt_max >= max_len:
            raise ValueError(f"{rows[0]['page_id']}: image and prompt use {prompt_max} tokens, over max_seq_length={max_len}; lower max_pixels")
        print(f"warning: {rows[0]['page_id']} truncated from {batch['input_ids'].shape[1]} to {max_len} tokens")
        for key in ("input_ids", "attention_mask", "labels"):
            batch[key] = batch[key][:, :max_len]
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def generate_texts(model, processor, rows: list[dict[str, Any]], device, max_new_tokens: int, config: dict[str, Any] | None = None) -> list[str]:
    import torch

    config = config or {}
    patch = int(config.get("image_patch_size", 14))
    outputs: list[str] = []
    for row in rows:
        item = messages(row["image_path"], None, max_pixels_for(config, row.get("category")))
        text = processor.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
        images, videos = _vision_inputs([item], patch)
        inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [ids[len(source):] for source, ids in zip(inputs["input_ids"], generated)]
        outputs.extend(processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return outputs
