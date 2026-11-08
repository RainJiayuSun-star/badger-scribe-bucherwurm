# Bücherwurm: Badger Scribe transcription pipeline (UW–Madison MLM26)

Team Bücherwurm's entry for [Kaggle Badger Scribe](https://www.kaggle.com/competitions/badger-scribe): verbatim transcription of 19th- and 20th-century handwritten archival pages with open-weight models, on one GPU.

- **Challenge rules:** [ChallengeDescription.md](ChallengeDescription.md)
- **Team workflow:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Fine-tuning and prediction:** [finetune/README.md](finetune/README.md)
- **Operations dashboard:** [ops_dashboard/README.md](ops_dashboard/README.md)
- **Approaches and metrics:** [design.md](design.md), [measurements.md](measurements.md), [results/](results/)

## Submission pipeline

Each page is transcribed by a vision-language model with a LoRA adapter for its
category, chosen from the `category` column of `test.csv`:

| Category | Base model (pinned commit) | Adapter |
| --- | --- | --- |
| German Kurrent letters (`kade_letters`) | `stanford-oval/churro-3B` @ `ca2150e` | LoRA on the language layers, projector and vision encoder, trained on all German training pages |
| Survey field notes (`survey_notes`) | `stanford-oval/churro-3B` @ `ca2150e` | LoRA trained on all English training pages |
| Account books (`dominy_accounts`) | `allenai/olmOCR-2-7B-1025` @ `e52d6f0` | LoRA trained on all English training pages |

`finetune.predict` loads one base model at a time, decodes its categories and frees
it before loading the next, so the whole run needs one GPU (peak 18.4 GB). Every
model is open-weight and nothing is sent to an external service. Prediction sees only
`page_id`, `doc_id` and `category`, never `label_source`. Choices were made on
held-out training documents (see [finetune/README.md](finetune/README.md)), not on the leaderboard.

### Reproduce a submission

1. Install: `uv sync --extra cuda --extra finetune`, and put the competition data in
   `badger-scribe-data/` (or set `BADGER_SCRIBE_DATA`).
2. Download the adapters from the Hugging Face repo
   [`bucherwurm-mlm26/bucherwurm-adapters`](https://huggingface.co/bucherwurm-mlm26/bucherwurm-adapters)
   (published with the final submission):
   `hf download bucherwurm-mlm26/bucherwurm-adapters --local-dir adapters`.
   Its model card lists which folder goes with which category.
3. Run the prediction. The base models download from Hugging Face at the pinned
   commits in the configs; set `HF_HUB_CACHE` to use a local copy instead.

```bash
python -m finetune.predict --config finetune/configs/churro-3b-bf16.yaml --run-id submission \
  --adapter kade_letters=adapters/<German folder> \
  --adapter survey_notes=adapters/<survey folder> \
  --adapter dominy_accounts=adapters/<account-book folder> \
  --base dominy_accounts=finetune/configs/olmocr2-7b-bf16-english.yaml
```

The CSV lands in `runs/finetune/submission/submission.csv`, and `summary.json` next to
it records the commit, models, adapters, peak GPU memory and wall time.

## Repository map

| Directory | What it does |
| --- | --- |
| `finetune/` | LoRA fine-tuning of Churro-3B and olmOCR-2-7B (label cleaning, document-level folds, training, evaluation, repetition-loop guard) and `finetune.predict`, which writes submissions |
| `benchmark/` | Zero-shot inference with local VLMs and OCR baselines, frozen holdout splits, scoring with the official `metric.py` |
| `pipeline/` | Early prototype: batch transcription through the BadgerBrain gateway (served Churro). Not the submitted pipeline |
| `ops_dashboard/` | Local dashboard for GPU jobs, agents, runs and Kaggle submissions |
| `splits/` | Saved validation page IDs (label text is rebuilt locally, never committed) |
| `results/` | Committed benchmark tables |

## License

The code is MIT licensed (see `LICENSE`). The base models and adapters keep their
own licenses: Churro-3B is under the Qwen Research License, olmOCR-2-7B under
Apache-2.0. Competition data is not included; it is for non-commercial research in
connection with the challenge.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/). Packages come from `pyproject.toml` + `uv.lock`.
- Competition data in `badger-scribe-data/` (not committed): `train.csv`, `metric.py`, `images/`,
  optional `metadata.csv`. Override the location with `BADGER_SCRIBE_DATA`.
- 4-bit runs need an NVIDIA GPU (~12 GB). Mac/CPU works in bf16/fp16 but cannot run `--dtype 4bit`.
- Fine-tuning needs CUDA: `uv sync --extra cuda --extra finetune`. A bf16 LoRA on Churro-3B
  peaks at about 12 GB, so it fits one 24 GB A5000. See [finetune/README.md](finetune/README.md).

## Earlier prototype: remote Churro workflow, infer then measure

This is how the team benchmarked before the local pipeline existed. It sends pages
to a team gateway serving Churro, so it is not the submitted pipeline. Use it in
two explicit steps. The first writes
predictions; the second compares them with the frozen ground truth. Do not
score Kaggle `test.csv`: it has no ground-truth text.

First, connect GlobalProtect, activate the project environment for the metric
script, and load the team gateway key in that same terminal:

```bash
source .venv/bin/activate
source ../badgerchat/env.sh
```

### 1. Run inference

Start with the 12-page Release 09 smoke set. Predictions are checkpointed
after every successful page, so `--continue` recovers an interrupted run.

```bash
python3 pipeline/transcribe_gateway.py \
  --input-csv splits/release_09/smoke_solution.csv \
  --image-dir badger-scribe-data/images \
  --output-csv runs/gateway-churro-release09-smoke/preds.csv \
  --continue
```

For the complete Release 09 validation set, change only the input and run ID:

```bash
python3 pipeline/transcribe_gateway.py \
  --input-csv splits/release_09/holdout_solution.csv \
  --image-dir badger-scribe-data/images \
  --output-csv runs/gateway-churro-release09-val/preds.csv \
  --continue
```

### 2. Compute metrics

Point the existing evaluator at the matching ground-truth CSV. It uses the
official, unchanged `metric.py` and writes detailed diagnostics to the same
run directory.

```bash
python benchmark/eval.py \
  --run-id gateway-churro-release09-smoke \
  --split release09-smoke \
  --solution-csv splits/release_09/smoke_solution.csv
```

The command prints macro CER, per-category CER, human/Kade diagnostics, and
writes `runs/gateway-churro-release09-smoke/metrics.json`. Use the matching
`holdout_solution.csv` and run ID for the full 119-page validation set. Do not
pass `--append-results` for Release 09; `results/wave1.*` is reserved for the
original frozen baseline.

To inspect one page before a full run, add `--page-id kade_007_p005` to the
inference command. A partial output is useful for debugging but not for a
meaningful score, because missing pages score as empty transcriptions.

### Optional: diagnostic score on all training pages

Use the full training set only to check fit or find bad transcriptions; it is
not a generalization metric. `train_full.json` records the same 619 pages for
inspection, while `train.csv` is the CSV input required by the runner and
evaluator:

```bash
python3 pipeline/transcribe_gateway.py \
  --input-csv badger-scribe-data/train.csv \
  --image-dir badger-scribe-data/images \
  --output-csv runs/gateway-churro-release09-train/preds.csv \
  --continue

python benchmark/eval.py \
  --run-id gateway-churro-release09-train \
  --split release09-train-diagnostic \
  --solution-csv badger-scribe-data/train.csv
```

## Commands

### Frozen validation splits

`splits/holdout.json` is the original 95-page Wave 1 validation holdout. Do
not regenerate or modify it: it is the comparison baseline for existing
results.

The September dataset release has a separate, extended 119-page holdout in
`splits/release_09/`. It retains every original validation page and adds 24
deterministically selected survey pages from the eight newly released survey
documents. Recreate it only from the supplied release directory:

```bash
python3 benchmark/make_holdout_release09.py \
  --data-root badger-scribe-data
```

The `*_solution.csv` files and `splits/release_09/train_full.json` contain
competition label text, so they are **not committed**. After a fresh clone, run
`python benchmark/make_solutions.py` to rebuild the four solution CSVs from
`train.csv` and the saved split IDs (it does not touch the split JSON files).
`train_full.json` is rebuilt by `benchmark/make_holdout_release09.py`.

Use `splits/release_09/smoke_solution.csv` for the 12-page Release 09 smoke
set and `splits/release_09/holdout_solution.csv` for the full extended
validation set. Keep results from this split separate from `results/wave1.*`.

`splits/release_09/train_full.json` records all labeled Release 09 training
pages in source order for diagnostic benchmarking. It is useful for detecting
training-set transcription errors or overfitting, but it is not a validation
set and should never be used to select a final model.

`make_holdout.py` — freezes the train/val split and writes `splits/` + solution CSVs.
Deterministic; run once and never re-run mid-benchmark.

`infer_vlm.py` — transcribes a split with one VLM, writing `runs/<run-id>/preds.csv`.
Resumes from existing predictions by default (`--no-resume` to disable).

```bash
uv run python benchmark/infer_vlm.py --model qwen2.5-vl-7b --split val --run-id qwen25vl-7b-zeroshot
```

Key flags: `--model` (alias or HF id), `--split smoke|val|train`, `--device auto|cuda|mps|cpu`,
`--dtype auto|bf16|fp16|fp32|4bit`, `--with-metadata`, `--limit N`.
`auto` picks CUDA then MPS then CPU, and 4-bit for 7B/8B models on CUDA.

`infer_pipeline.py` — the non-VLM baseline: line segmentation (Kraken, falling back to
projection) plus TrOCR recognizers. Same `--split` / `--run-id` flags.

`eval.py` — scores a run against the holdout and prints category CER.

```bash
uv run python benchmark/eval.py --run-id <run-id> --split val --append-results   # score + add to results/
uv run python benchmark/eval.py --run-id rebuild --split val --write-table       # redraw wave1.md only
```

`--append-results` appends to `results/wave1.csv` and redraws `results/wave1.md`.
`--write-table` only rebuilds the markdown, and is skipped unless `--run-id` names a run
that has no `preds.csv` — hence the placeholder above.

`error_report.py` — builds a local HTML page for reviewing errors: each training page's image
next to an inline diff of prediction against reference, worst pages first, filterable by
category and label source. `--baseline` adds a per-page comparison with a second run.

```bash
python benchmark/error_report.py --preds 'runs/finetune/<run>/zero_shot/*-preds.csv' --out runs/reports/<run>.html
open runs/reports/<run>.html
```

The report contains label text, so it stays under `runs/` and is never committed or published.

`run_wave1.py` — runs every model and eval in sequence, continuing past failures.

```bash
uv run python benchmark/run_wave1.py
```

## Notes

Do not mix Mac and 3060 wall-clock in one ranking; CER is comparable, `sec/page` is not.

DeepSeek-OCR is the one exception to the shared setup. It needs transformers 4.x, supplied by an
overlay that reuses the project's torch:

```bash
uv run --with "transformers==4.46.3" python benchmark/infer_vlm.py --model deepseek-ocr --split smoke --run-id smoke-deepseek-ocr
```

It is also task-prompted rather than instruction-following: the shared verbatim prompt yields empty
output, so it runs on `Free OCR.` instead. Its CER is therefore not a like-for-like comparison; each
run records the substitution in `runs/<run-id>/config.json` as `prompt_override`.
