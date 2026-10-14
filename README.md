# Badger Scribe Project - UW Madison MLM26 Challenge Team bucherwurm

UW–Madison MLM26 / [Kaggle Badger Scribe](https://www.kaggle.com/competitions/badger-scribe): verbatim transcription of 19th-century archival pages with open-weight models.

- **Plan:** [PLAN.md](PLAN.md)
- **Challenge rules:** [ChallengeDescription.md](ChallengeDescription.md)
- **Approaches:** [design.md](design.md)
- **Metric definitions:** [measurements.md](measurements.md)
- **Numbers from runs:** [results/](results/)
- **Remote Churro pipeline:** [pipeline/README.md](pipeline/README.md)

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/). Packages come from `pyproject.toml` + `uv.lock`.
- Competition data in `badger-scribe-data/` (not committed): `train.csv`, `metric.py`, `images/`,
  optional `metadata.csv`. Override the location with `BADGER_SCRIBE_DATA`.
- 4-bit runs need an NVIDIA GPU (~12 GB). Mac/CPU works in bf16/fp16 but cannot run `--dtype 4bit`.

## Remote Churro workflow: infer, then measure

Use the remote Churro pipeline in two explicit steps. The first writes
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
  --image-dir /Users/rains/rainworkspace/UWMadison/mlm26/badger-scribe-data-09/images \
  --output-csv runs/gateway-churro-release09-smoke/preds.csv \
  --continue
```

For the complete Release 09 validation set, change only the input and run ID:

```bash
python3 pipeline/transcribe_gateway.py \
  --input-csv splits/release_09/holdout_solution.csv \
  --image-dir /Users/rains/rainworkspace/UWMadison/mlm26/badger-scribe-data-09/images \
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
  --input-csv /Users/rains/rainworkspace/UWMadison/mlm26/badger-scribe-data-09/train.csv \
  --image-dir /Users/rains/rainworkspace/UWMadison/mlm26/badger-scribe-data-09/images \
  --output-csv runs/gateway-churro-release09-train/preds.csv \
  --continue

python benchmark/eval.py \
  --run-id gateway-churro-release09-train \
  --split release09-train-diagnostic \
  --solution-csv /Users/rains/rainworkspace/UWMadison/mlm26/badger-scribe-data-09/train.csv
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
  --data-root ../badger-scribe-data-09
```

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
