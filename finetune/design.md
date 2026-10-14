# Fine-tuning experiment contract

## Objective

Adapt `stanford-oval/churro-3B` and `Qwen/Qwen2.5-VL-7B-Instruct` to Badger
Scribe while measuring improvements on human-labelled material. The frozen
119-page Release 09 holdout is never used for training or checkpoint choice.

## Data contract

The trainer reads `badger-scribe-data-09/train.csv` and excludes every ID in
`splits/release_09/holdout.json`. It makes one deterministic internal
development partition: 55 human Kade pages, both eligible human Dominy pages,
and four eligible human survey pages. Kade therefore trains on human labels;
Dominy and survey include `silver_claude` machine drafts as deliberately noisy
supervision. Metrics always record label-source counts and human-only
development CER.

## Curricula

Every curriculum is run on both bases with the same split, seed, generation
settings, effective global batch size, and augmentation profile:

1. `kade`, `dominy`, `survey`
2. `kade-dominy`, `kade-survey`, `dominy-survey`, `kade-dominy-survey`
3. `kade-then-dominy`, `dominy-then-survey`, `kade-then-dominy-then-survey`
4. Reverse controls: `dominy-then-kade`, `survey-then-dominy`,
   `survey-then-dominy-then-kade`

Joint curricula sample all selected categories together. Sequential curricula
continue the same adapter through each stage, so comparisons detect order
effects and catastrophic forgetting. Select a checkpoint on internal
development CER, then run the locked candidate once on the Release 09 holdout.

## Augmentation

`finetune/augmentation/default.yaml` is applied on the fly to every training
arm. It contains small rotation, brightness/contrast, blur, JPEG artifacts,
scan noise, and small perspective changes. It never flips, crops text, applies
large rotation, elastic warping, or changes the transcript. The profile version
and seed are stored in each run manifest. `kade-no-augmentation` is the control
for measuring whether augmentation helps.

## Execution and evidence

QLoRA (NF4, BF16 compute) with gradient checkpointing is the default. The
trainer works through `accelerate launch` on one, two, or four-plus A5000 GPUs;
it adjusts gradient accumulation to preserve the configured effective global
batch. Each run writes resumable checkpoints, a terminal progress bar,
TensorBoard events, `metrics.csv`, `metrics.jsonl`, predictions, and config /
environment / GPU provenance. W&B receives scalar metrics only when
`WANDB_API_KEY` is present.
