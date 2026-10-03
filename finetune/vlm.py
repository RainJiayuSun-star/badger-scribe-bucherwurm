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

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError:  # pragma: no cover - supports future transformer aliases
        from transformers import AutoModelForVision2Seq as ModelClass

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(config["model_id"], trust_remote_code=True)
    model = ModelClass.from_pretrained(
        config["model_id"],
        torch_dtype=dtype,
        quantization_config=quantization,
        device_map={"": local_rank},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
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


def messages(image: str | Path | Image.Image, text: str | None = None) -> list[dict[str, Any]]:
    image_value = str(image) if isinstance(image, Path) else image
    content: list[dict[str, Any]] = [{"type": "image", "image": image_value}, {"type": "text", "text": PROMPT}]
    result = [{"role": "user", "content": content}]
    if text is not None:
        result.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    return result


def _vision_inputs(message_batch: list[list[dict[str, Any]]]):
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install qwen-vl-utils to train Qwen-family VLMs") from exc
    images, videos = process_vision_info(message_batch)
    return images, videos


def collate_train(rows: list[dict[str, Any]], processor, device, epoch: int, augmenter) -> dict[str, Any]:
    """Build labels that score assistant tokens only, not the instruction."""
    import torch

    materialized = []
    for row in rows:
        with Image.open(row["image_path"]) as source:
            image = augmenter.apply(source, row["page_id"], epoch) if augmenter else source.convert("RGB")
        materialized.append((row, image))
    full_messages = [messages(image, row["text"]) for row, image in materialized]
    prompt_messages = [messages(image) for _, image in materialized]
    full_texts = [processor.apply_chat_template(item, tokenize=False, add_generation_prompt=False) for item in full_messages]
    prompt_texts = [processor.apply_chat_template(item, tokenize=False, add_generation_prompt=True) for item in prompt_messages]
    images, videos = _vision_inputs(full_messages)
    batch = processor(text=full_texts, images=images, videos=videos, padding=True, return_tensors="pt")
    prompt_images, prompt_videos = _vision_inputs(prompt_messages)
    prompts = processor(text=prompt_texts, images=prompt_images, videos=prompt_videos, padding=True, return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    prompt_lengths = prompts["attention_mask"].sum(dim=1).tolist()
    for index, length in enumerate(prompt_lengths):
        labels[index, : int(length)] = -100
    batch["labels"] = labels
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def generate_texts(model, processor, rows: list[dict[str, Any]], device, max_new_tokens: int) -> list[str]:
    import torch

    outputs: list[str] = []
    for row in rows:
        item = messages(row["image_path"])
        text = processor.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
        images, videos = _vision_inputs([item])
        inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [ids[len(source):] for source, ids in zip(inputs["input_ids"], generated)]
        outputs.extend(processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return outputs
