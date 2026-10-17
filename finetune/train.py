"""Accelerate + QLoRA trainer for Badger Scribe page transcription.

Run with: accelerate launch --num_processes 2 -m finetune.train --config ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .augmentation import OCRAugmenter, load_profile
from .config import load_config, resolve_curriculum, resolved_config_json
from .prep import prepare_rows
from .data import fold_partition, image_path, held_ids, make_partition, read_rows, selected_rows, write_manifest
from .vlm import collate_train, generate_texts, load_model_and_processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--curriculum", help="Override the curriculum in the config")
    parser.add_argument("--run-id", help="Override the generated run ID")
    parser.add_argument("--max-steps", type=int, help="Stop after this many optimizer steps; useful for smoke tests")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in this run directory")
    parser.add_argument("--evaluate-holdout", action="store_true", help="Score the selected best adapter once on the locked Release 09 holdout")
    parser.add_argument("--fold", type=Path, action="append", help="Leave-one-document-out fold file from finetune.folds; its held-out documents become the development set. With --eval-only it may be repeated")
    parser.add_argument("--eval-only", action="store_true", help="Score the untrained model (zero-shot) on each --fold's held-out pages, then exit")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def version_info() -> dict[str, str]:
    import accelerate
    import peft
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "peft": peft.__version__,
    }


def gpu_info() -> list[dict[str, Any]]:
    import torch

    return [
        {"index": index, "name": torch.cuda.get_device_name(index), "memory_bytes": torch.cuda.get_device_properties(index).total_memory}
        for index in range(torch.cuda.device_count())
    ]


class MetricWriter:
    def __init__(self, run_dir: Path, enabled_wandb: bool, config: dict[str, Any], accelerator):
        self.run_dir = run_dir
        self.accelerator = accelerator
        self.start = time.monotonic()
        self.jsonl = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
        self.csv_path = run_dir / "metrics.csv"
        self.csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
        fields = [
            "step", "elapsed_seconds", "epoch", "stage", "train_loss", "learning_rate", "gradient_norm",
            "examples_per_second", "peak_gpu_memory_mb", "val_macro_cer", "val_wer", "val_length_ratio",
            "val_runaway_count", "val_cer_kade_letters", "val_cer_dominy_accounts", "val_cer_survey_notes",
        ]
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fields, extrasaction="ignore")
        if self.csv_file.tell() == 0:
            self.writer.writeheader()
        from torch.utils.tensorboard import SummaryWriter

        self.tensorboard = SummaryWriter(run_dir / "tensorboard")
        self.wandb = None
        if enabled_wandb and os.environ.get("WANDB_API_KEY"):
            import wandb

            self.wandb = wandb.init(
                project=config["wandb_project"],
                group=config["model_name"],
                name=config["run_id"],
                # Keep W&B telemetry scalar-only: no page IDs, prompts, images,
                # text, local paths, or serialized manifests leave this machine.
                config={key: config[key] for key in ("model_id", "curriculum", "seed", "epochs", "learning_rate", "effective_batch_size", "adapter_policy", "lora_rank") if key in config},
                settings=wandb.Settings(silent=True),
            )

    def log(self, payload: dict[str, Any], step: int) -> None:
        payload = {**payload, "step": step, "elapsed_seconds": round(time.monotonic() - self.start, 3)}
        self.jsonl.write(json.dumps(payload, sort_keys=True) + "\n")
        self.jsonl.flush()
        self.writer.writerow({key: payload.get(key, "") for key in self.writer.fieldnames})
        self.csv_file.flush()
        for key, value in payload.items():
            if isinstance(value, (int, float)) and key not in {"step", "epoch"}:
                self.tensorboard.add_scalar(key, value, step)
        self.tensorboard.flush()
        if self.wandb:
            self.wandb.log(payload, step=step)

    def close(self) -> None:
        self.jsonl.close()
        self.csv_file.close()
        self.tensorboard.close()
        if self.wandb:
            self.wandb.finish()


def write_predictions(path: Path, rows: list[dict[str, str]], texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["page_id", "text"])
        writer.writeheader()
        writer.writerows({"page_id": row["page_id"], "text": text} for row, text in zip(rows, texts))


def evaluate(model, processor, rows: list[dict[str, str]], device, config: dict[str, Any], output: Path) -> dict[str, float]:
    """Evaluate only on the fixed human development set."""
    from importlib.util import module_from_spec, spec_from_file_location

    metric_path = Path(config["data_root"]) / "metric.py"
    spec = spec_from_file_location("badger_metric", metric_path)
    metric = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(metric)
    model.eval()
    preds = generate_texts(model, processor, rows, device, int(config.get("eval_max_new_tokens", 2048)), config)
    write_predictions(output, rows, preds)
    by_category: dict[str, list[float]] = defaultdict(list)
    wers: dict[str, list[float]] = defaultdict(list)
    ratios: list[float] = []
    runaways = 0
    page_scores = []
    for row, pred in zip(rows, preds):
        reference = row["text"]
        page_cer = float(metric.page_cer(pred, reference))
        page_scores.append({"page_id": row["page_id"], "doc_id": row["doc_id"], "category": row["category"],
                            "label_source": row["label_source"], "cer": round(page_cer, 4),
                            "length_ratio": round(len(pred) / max(1, len(reference)), 3)})
        by_category[row["category"]].append(page_cer)
        wers[row["category"]].append(float(metric.page_wer(pred, reference)))
        ratio = len(pred) / max(1, len(reference))
        ratios.append(ratio)
        runaways += int(ratio > 2.5)
    result: dict[str, float] = {"val_macro_cer": float(np.mean([np.mean(values) for values in by_category.values()]))}
    result["val_wer"] = float(np.mean([value for values in wers.values() for value in values]))
    result["val_length_ratio"] = float(np.mean(ratios))
    result["val_runaway_count"] = float(runaways)
    for category, values in by_category.items():
        result[f"val_cer_{category}"] = float(np.mean(values))
    with output.with_name(output.stem + "-scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(page_scores[0]))
        writer.writeheader()
        writer.writerows(page_scores)
    model.train()
    return result


def trainable_groups(model, language_lr: float, vision_lr: float) -> list[dict[str, Any]]:
    """Vision-tower adapters get vision_learning_rate; everything else learning_rate."""
    vision, other = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (vision if (".visual." in name or name.startswith("visual.")) and "merger" not in name else other).append(parameter)
    groups = [{"params": other, "lr": language_lr}]
    if vision:
        groups.append({"params": vision, "lr": vision_lr})
    return groups


def stage_rows(partition: dict[str, Any], curriculum: dict[str, Any]) -> list[list[dict[str, str]]]:
    return [selected_rows(partition, stage, "train") for stage in curriculum["stages"]]


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda path: int(path.name.rsplit("-", 1)[-1]))
    return checkpoints[-1] if checkpoints else None


def main() -> int:
    args = parse_args()
    try:
        import torch
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm
        from transformers import get_cosine_schedule_with_warmup
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install CUDA fine-tuning dependencies: uv sync --extra cuda --extra finetune") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Run this command on the A5000 host.")
    config = load_config(args.config)
    curriculum = resolve_curriculum(config, args.curriculum)
    config["curriculum"] = curriculum["name"]
    config["model_name"] = config.get("model_name", Path(config["model_id"]).name.lower().replace("/", "-"))
    folds = args.fold or []
    if len(folds) > 1 and not args.eval_only:
        raise SystemExit("training takes one --fold; repeat --fold only with --eval-only")
    if args.eval_only and not folds:
        raise SystemExit("--eval-only needs at least one --fold")
    fold_suffix = "-zeroshot" if args.eval_only else (f"-fold-{folds[0].stem}" if folds else "")
    config["run_id"] = args.run_id or f"{config['model_name']}-{curriculum['name']}-seed{config['seed']}{fold_suffix}"
    run_dir = Path(config["runs_root"]) / config["run_id"]
    requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if int(config["effective_batch_size"]) % (requested_world_size * int(config["per_device_batch_size"])):
        raise SystemExit("effective_batch_size must divide evenly by world_size * per_device_batch_size")
    grad_accum = int(config["effective_batch_size"]) // (requested_world_size * int(config["per_device_batch_size"]))
    accelerator = Accelerator(gradient_accumulation_steps=grad_accum)
    world_size = accelerator.num_processes
    if world_size != requested_world_size:
        raise SystemExit(f"Accelerate initialized {world_size} workers, expected {requested_world_size}")
    config["world_size"] = world_size
    config["gradient_accumulation_steps"] = grad_accum
    config["gpu_topology"] = gpu_info()
    set_seed(int(config["seed"]))
    seed_everything(int(config["seed"]))

    rows = read_rows(Path(config["data_root"]))
    if config.get("use_cleaned_labels", True):
        # "target" is what the model learns; "text" stays the original key for scoring.
        cleaned = {record["page_id"]: record["text"] for record in prepare_rows(rows)}
        for row in rows:
            row["target"] = cleaned[row["page_id"]]
    if folds:
        if args.evaluate_holdout:
            raise SystemExit("--evaluate-holdout and --fold are separate protocols; use one")
        partition = fold_partition(rows, folds[0])
        config["fold"] = partition["fold"]
    else:
        partition = make_partition(rows, held_ids(Path(config["holdout"])), int(config["seed"]))
    stages = stage_rows(partition, curriculum)
    for stage in stages:
        for row in stage:
            row["image_path"] = str(image_path(Path(config["data_root"]), row["page_id"]))
    dev_rows = [dict(row, image_path=str(image_path(Path(config["data_root"]), row["page_id"]))) for row in partition["development"]]
    profile_path = Path(config["augmentation_profile"])
    if curriculum["name"] == "kade-no-augmentation":
        profile_path = profile_path.with_name("none.yaml")
    profile = load_profile(profile_path)
    config["augmentation_profile"] = str(profile_path)
    config["augmentation"] = profile

    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(resolved_config_json(config), encoding="utf-8")
        (run_dir / "environment.json").write_text(json.dumps(version_info(), indent=2) + "\n", encoding="utf-8")
        write_manifest(run_dir / "manifest.json", partition, curriculum, stages)
    accelerator.wait_for_everyone()
    model, processor = load_model_and_processor(config, accelerator.local_process_index)
    if args.eval_only:
        # The LoRA adapter starts at zero, so this scores the base model unchanged.
        summary = {}
        for fold_path in folds:
            fold_rows = [dict(row, image_path=str(image_path(Path(config["data_root"]), row["page_id"])))
                         for row in fold_partition(rows, fold_path)["development"]]
            result = evaluate(model, processor, fold_rows, accelerator.device, config, run_dir / "zero_shot" / f"{fold_path.stem}-preds.csv")
            result["pages"] = len(fold_rows)
            summary[fold_path.stem] = result
            print(f"{fold_path.stem}: CER {result['val_macro_cer']:.4f} over {len(fold_rows)} pages", flush=True)
            (run_dir / "zero_shot" / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    groups = trainable_groups(model, float(config["learning_rate"]), float(config["vision_learning_rate"]))
    try:
        optimizer = AdamW(groups, fused=True)
    except (TypeError, RuntimeError):
        optimizer = AdamW(groups)
    total_batches = sum(math.ceil(len(stage) / int(config["per_device_batch_size"])) for stage in stages)
    total_steps = max(1, math.ceil(total_batches * int(config["epochs"]) / grad_accum))
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(total_steps * float(config["warmup_ratio"]))), num_training_steps=total_steps)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    # from_pretrained returns the model in eval mode, where gradient checkpointing
    # and LoRA dropout are inactive. Without this the first epoch used about 2.4x
    # the GPU memory (26.5 GB vs 10.9 GB for one Churro-3B page in bf16).
    model.train()
    augmenter = OCRAugmenter(profile, int(config["seed"]))
    writer = MetricWriter(run_dir, not args.no_wandb and accelerator.is_main_process, config, accelerator) if accelerator.is_main_process else None
    global_step = 0
    resume_stage = 0
    resume_stage_epoch = 0
    if args.resume:
        checkpoint = latest_checkpoint(run_dir)
        if checkpoint:
            accelerator.load_state(checkpoint)
            state_path = checkpoint / "state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                global_step = int(state["global_step"])
                resume_stage = int(state.get("stage", 1)) - 1
                resume_stage_epoch = int(state.get("stage_epoch", 0))
    progress = tqdm(total=total_steps, initial=global_step, disable=not accelerator.is_local_main_process, desc=f"{config['model_name']} {curriculum['name']}")
    started = time.monotonic()
    best_cer = float("inf")
    try:
        for stage_index, rows_for_stage in enumerate(stages):
            if args.resume and stage_index < resume_stage:
                continue
            first_epoch = resume_stage_epoch if stage_index == resume_stage else 0
            for stage_epoch in range(first_epoch, int(config["epochs"])):
                epoch = stage_index * int(config["epochs"]) + stage_epoch
                loader = DataLoader(rows_for_stage, batch_size=int(config["per_device_batch_size"]), shuffle=True, collate_fn=lambda items: items)
                loader = accelerator.prepare(loader)
                for raw_batch in loader:
                    batch = collate_train(raw_batch, processor, accelerator.device, epoch, augmenter, config)
                    with accelerator.accumulate(model):
                        output = model(**batch)
                        accelerator.backward(output.loss)
                        if accelerator.sync_gradients:
                            grad_norm = accelerator.clip_grad_norm_(trainable, float(config.get("max_grad_norm", 1.0)))
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    if not accelerator.sync_gradients:
                        continue
                    global_step += 1
                    elapsed = max(time.monotonic() - started, 1e-6)
                    payload = {
                        "epoch": epoch + 1,
                        "stage": stage_index + 1,
                        "train_loss": float(accelerator.gather(output.loss.detach()).mean().item()),
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                        "gradient_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
                        "examples_per_second": round(global_step * int(config["effective_batch_size"]) / elapsed, 3),
                        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
                    }
                    if writer and (global_step % int(config["logging_steps"]) == 0 or global_step == 1):
                        writer.log(payload, global_step)
                    progress.set_postfix(loss=f"{payload['train_loss']:.4f}", lr=f"{payload['learning_rate']:.2e}", mem=f"{payload['peak_gpu_memory_mb']:.0f}MB")
                    progress.update(1)
                    if global_step >= total_steps:
                        break
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(model)
                    result = evaluate(unwrapped, processor, dev_rows, accelerator.device, config, run_dir / "development" / f"epoch-{epoch + 1}-preds.csv")
                    result.update({"epoch": epoch + 1, "stage": stage_index + 1})
                    if writer:
                        writer.log(result, global_step)
                    if result["val_macro_cer"] < best_cer:
                        best_cer = result["val_macro_cer"]
                        unwrapped.save_pretrained(run_dir / "best_adapter")
                        processor.save_pretrained(run_dir / "best_adapter")
                accelerator.wait_for_everyone()
                checkpoint = run_dir / f"checkpoint-{global_step}"
                accelerator.save_state(checkpoint)
                if accelerator.is_main_process:
                    (checkpoint / "state.json").write_text(json.dumps({"global_step": global_step, "epoch": epoch + 1, "stage": stage_index + 1, "stage_epoch": stage_epoch + 1}) + "\n", encoding="utf-8")
                accelerator.wait_for_everyone()
                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break
    finally:
        progress.close()
        if writer:
            writer.close()
    if accelerator.is_main_process:
        if args.evaluate_holdout:
            best_adapter = run_dir / "best_adapter"
            if not best_adapter.exists():
                raise SystemExit("no best adapter was written; train at least one complete epoch first")
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.load_adapter(str(best_adapter), adapter_name="selected_best", is_trainable=False)
            unwrapped.set_adapter("selected_best")
            holdout_rows = [dict(row, image_path=str(image_path(Path(config["data_root"]), row["page_id"]))) for row in rows if row["page_id"] in held_ids(Path(config["holdout"]))]
            holdout_result = evaluate(unwrapped, processor, holdout_rows, accelerator.device, config, run_dir / "preds.csv")
            (run_dir / "holdout_metrics.json").write_text(json.dumps(holdout_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"completed {config['run_id']} with best development macro CER {best_cer:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
