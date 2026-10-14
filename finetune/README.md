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

After locking a candidate on the human development set, score it once on the
frozen Release 09 holdout. This writes `preds.csv` and `holdout_metrics.json`
inside the run directory using the dataset's official `metric.py` functions.

```bash
accelerate launch --num_processes 1 -m finetune.train \
  --config finetune/configs/churro-3b.yaml --resume --evaluate-holdout
```
