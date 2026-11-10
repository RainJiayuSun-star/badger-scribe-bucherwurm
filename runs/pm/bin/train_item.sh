#!/bin/bash
# Queue item, run from inside a clean clone: train_item.sh <run id> <config> <fold name>
# Exits 0 at once if the run already finished; resumes from its checkpoint if one exists.
id=$1; cfg=$2; fold=$3
run=runs/finetune/$id
grep -q '"phase": "finished"' "$run/status.json" 2>/dev/null && exit 0
resume=""; ls -d "$run"/checkpoint-* >/dev/null 2>&1 && resume="--resume"
exec ~/badger-scribe-bucherwurm/.venv/bin/accelerate launch --num_processes 1 --mixed_precision no -m finetune.train \
  --config "$cfg" --fold "runs/folds/$fold.json" --run-id "$id" --no-wandb $resume
