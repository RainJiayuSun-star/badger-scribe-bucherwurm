# Badger Scribe Project - UW Madison MLM26 Challenge Team bucherwurm

UW–Madison MLM26 / [Kaggle Badger Scribe](https://www.kaggle.com/competitions/badger-scribe): verbatim transcription of 19th-century archival pages with open-weight models.

- **Plan:** [PLAN.md](PLAN.md)
- **Challenge rules:** [ChallengeDescription.md](ChallengeDescription.md)
- **Approaches:** [design.md](design.md)
- **Metric definitions:** [measurements.md](measurements.md)
- **Numbers from runs:** [results/](results/)

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/). Packages come from `pyproject.toml` + `uv.lock`.
- Competition data in `badger-scribe-data/` (not committed): `train.csv`, `metric.py`, `images/`,
  optional `metadata.csv`. Override the location with `BADGER_SCRIBE_DATA`.
- 4-bit runs need an NVIDIA GPU (~12 GB). Mac/CPU works in bf16/fp16 but cannot run `--dtype 4bit`.

## Getting started

```bash
uv sync                # Mac / CPU / MPS
uv sync --extra cuda   # NVIDIA: adds bitsandbytes + CUDA torch wheels

uv run python benchmark/make_holdout.py                                             # once
uv run python benchmark/infer_vlm.py --model churro --split smoke --run-id smoke-churro
uv run python benchmark/eval.py --run-id smoke-churro --split smoke --append-results
```

## Commands

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
