# Fine-tuning

The fine-tuning harness trains page-level Qwen-family vision-language models
on Release 09 with CUDA, Accelerate, QLoRA, resumable checkpoints, and local
metrics. The default run never trains on `splits/release_09/holdout.json`. Fold
runs (`--fold`) hold out whole documents instead and do train on those pages, so
never score a fold-trained model on that holdout.

## Setup on the A5000 host

```bash
uv sync --extra cuda --extra finetune
accelerate config
```

Run each job on one GPU. Multi-GPU runs aren't supported: the other GPUs can time
out waiting during scoring, `--eval-only` writes the same files from every
process, and step counts assume one process.

```bash
# Smoke run
accelerate launch --num_processes 1 -m finetune.train \
  --config finetune/configs/churro-3b.yaml --max-steps 2
```

The curriculum names are listed in `finetune/configs/curricula.yaml`. A run
writes `runs/finetune/<run-id>/`, including its immutable page manifest,
resolved config, checkpoints, TensorBoard events, CSV/JSONL metrics, and
development predictions. Resume with `--resume`.

```bash
tensorboard --logdir runs/finetune
python -m finetune.plot_metrics runs/finetune/<run-id>
```

Set `WANDB_API_KEY` to mirror scalar metrics to Weights & Biases. Images and
transcriptions are never sent to W&B. See [design.md](design.md) for the
experiment contract.

After a smoke run, verify checkpoint resumption without changing the run ID:

```bash
accelerate launch --num_processes 1 -m finetune.train \
  --config finetune/configs/churro-3b.yaml --max-steps 2 --resume
```

## Evaluating on unseen documents (folds)

Every test page comes from a document that is not in `train.csv`, so the
default random development set flatters a model that has memorized a writer.
Score on whole held-out documents instead:

```bash
python -m finetune.prep     # cleaned targets -> runs/prep/records.jsonl (summary only)
python -m finetune.folds    # 19 fold files -> runs/folds/<name>.json
```

German letters and survey notebooks get one fold per document; the 18 small
account books are grouped into 6 folds of about 6 pages. Fold files hold page
IDs only, never label text.

```bash
# Fine-tune with one held-out document as the development set
accelerate launch --num_processes 1 --mixed_precision no -m finetune.train \
  --config finetune/configs/churro-3b-bf16.yaml --fold runs/folds/kade_103.json

# Zero-shot score of the untouched base model on several folds
accelerate launch --num_processes 1 --mixed_precision no -m finetune.train \
  --config finetune/configs/churro-3b-bf16.yaml --eval-only \
  --fold runs/folds/kade_007.json --fold runs/folds/kade_103.json
```

Each evaluation writes predictions plus a `*-scores.csv` with per-page CER,
length ratio and whether the loop guard fired, so bad pages are easy to find.

`best_adapter` is chosen on all held-out pages. English folds are mostly machine
drafts and hold very few human pages, so `val_human_macro_cer` and
`val_human_pages` report the human-labeled pages alone, and they should be
checked too.

## Training and decoding options

| Config key | Default | Effect |
| --- | --- | --- |
| `quantization` | `4bit` | `none` trains a bf16 LoRA (the `*-bf16.yaml` configs) |
| `max_pixels`, `max_pixels_by_category` | 1.0 MP | Image budget per page; the bf16 configs give survey spreads 2560 image tokens (2.0 MP on Churro, 2.6 MP on Qwen3-VL) |
| `use_cleaned_labels` | true | Train on `finetune.prep` targets (envelope Postmark lines removed). Scoring always uses the original key |
| `mask_illegible` | true | No loss on `#` characters, so the model never learns to emit them |
| `human_repeats` | 1 | Repeat each human-labeled training page this many times per epoch (training only) |
| `loop_guard` | true | Stop decoding at a repetition loop and keep its first copy (`finetune/loopguard.py`) |
| `loop_retry_temperatures` | `[]` | If set, re-decode looping or near-empty pages with sampling at these temperatures |
| `keep_checkpoints` | 2 | FIFO: after a new resume checkpoint is complete, delete the oldest beyond N (0 keeps all) |
| `min_free_disk_gib`, `min_free_gpu_gib` | 2, 12 | Preflight thresholds; 0 turns a check off |

No option reads `label_source` at inference time; the competition forbids
behaving differently on silver-labelled pages.

## Checkpoints and run diagnostics

After each epoch the run saves a resume checkpoint holding only what training
changes: the LoRA weights, optimizer and scheduler state and random-number
state (about 0.4 GB for Churro-3B, not the 7.4 GB of a full model). It is
written under a temporary name and renamed when complete, and only then are
older ones deleted, so the last complete epoch is always on disk. If the disk
lacks room or a write fails, the checkpoint is skipped with a warning and training
continues. `best_adapter` is swapped in the same way, so a failed save keeps the
previous best.
`--resume` with the same `--run-id` continues from the newest one.

Every run also writes two files to its run directory for diagnosing a run
that stopped:

- `status.json`: the current phase (`loading_model`, `training`, `evaluating`,
  `saving_checkpoint`, `finished`, `failed`), step, epoch, page within an
  evaluation, and RAM, GPU memory and free disk at the last update.
- `events.jsonl`: one line per event (each phase, evaluation score, checkpoint
  saved or skipped, signal received, crash with traceback).

`python -m finetune.status` lists every run with its best score and how it
ended; `python -m finetune.status <run-id>` shows one run's position, scores
per epoch, last resource snapshot, recent events and, after a crash, the end of
the traceback.

Before loading the model, a run checks that the data is in place, that the run
and checkpoint directories have `min_free_disk_gib` free, that the GPU has
`min_free_gpu_gib` free, that the same run is not already running, and that it
would not start over on existing checkpoints without `--resume`, or over a
finished `best_adapter` with no checkpoint to continue from. If
anything fails it exits before doing any work and lists every problem (above
accelerate's own traceback).

How to read the last line of `events.jsonl`:

| Last event | Meaning |
| --- | --- |
| `finished` | The run completed |
| `failed` | Python error; the `traceback` field says where |
| `signal` (SIGTERM) | Stopped from outside, for example the workspace was shut down or evicted. A pod's signal may not reach a run inside tmux; then see the last row |
| anything else | Killed without warning (out of memory, node failure); `status.json` shows where it was |

## EFabric notes

- `uv` is at `/opt/conda/bin/uv` and is not on the PATH of a non-interactive SSH command.
- The home directory is JuiceFS, so the first import of torch and transformers takes a few minutes.
- The home directory has a 64 GB quota, and run directories (with their small
  checkpoints) belong there. Don't write large files to `/tmp`: about 15 GB of old
  full-size checkpoints there got two workspaces restarted, most likely by the pod's
  ephemeral-storage limit.
- `scp` does not work on the workspace; copy single files with `ssh host 'cat > path' < path`.

After locking a candidate on the human development set, score it once on the
frozen Release 09 holdout. This writes `preds.csv` and `holdout_metrics.json`
inside the run directory using the dataset's official `metric.py` functions.

```bash
accelerate launch --num_processes 1 -m finetune.train \
  --config finetune/configs/churro-3b.yaml --resume --evaluate-holdout
```
