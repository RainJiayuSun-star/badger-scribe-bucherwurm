# Fine-tuning

The fine-tuning harness trains page-level Qwen-family vision-language models
on Release 09 with CUDA, Accelerate, QLoRA, resumable checkpoints, and local
metrics. It never trains on `splits/release_09/holdout.json`.

## Setup on the A5000 host

```bash
uv sync --extra cuda --extra finetune
accelerate config
```

Use one GPU to prove a configuration, then scale the same configuration without
changing its effective global batch size:

```bash
# One A5000 smoke run
accelerate launch --num_processes 1 -m finetune.train \
  --config finetune/configs/churro-3b.yaml --max-steps 2

# Two A5000 GPUs
accelerate launch --num_processes 2 -m finetune.train \
  --config finetune/configs/qwen25vl-7b.yaml

# Four A5000 GPUs
accelerate launch --num_processes 4 -m finetune.train \
  --config finetune/configs/qwen25vl-7b.yaml --curriculum kade-dominy-survey
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
python -m finetune.prep     # cleaned targets and spread splits -> runs/prep/records.jsonl (summary only)
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
| `checkpoint_root` | run directory | Where resume checkpoints go (about 7.4 GB each for Churro-3B). Metrics, predictions and `best_adapter` always stay in `runs_root` |
| `keep_checkpoints` | 0 (all) | Keep only the newest N checkpoints; the oldest is deleted after a new one is written |

No option reads `label_source` at inference time; the competition forbids
behaving differently on silver-labelled pages.

## EFabric notes

- `uv` is at `/opt/conda/bin/uv` and is not on the PATH of a non-interactive SSH command.
- The home directory is JuiceFS, so the first import of torch and transformers takes a few minutes.
- The home directory has a 64 GB quota, which three Churro checkpoints nearly fill. Add
  `checkpoint_root: /tmp/badger-checkpoints` and `keep_checkpoints: 2` to the config so
  checkpoints go to the workspace's local disk (lost when the workspace ends) and results stay in home.
- `scp` does not work on the workspace; copy single files with `ssh host 'cat > path' < path`.

After locking a candidate on the human development set, score it once on the
frozen Release 09 holdout. This writes `preds.csv` and `holdout_metrics.json`
inside the run directory using the dataset's official `metric.py` functions.

```bash
accelerate launch --num_processes 1 -m finetune.train \
  --config finetune/configs/churro-3b.yaml --resume --evaluate-holdout
```
