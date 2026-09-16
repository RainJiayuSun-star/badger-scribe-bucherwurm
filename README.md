# Cooler Badger Scribe

UW–Madison MLM26 / [Kaggle Badger Scribe](https://www.kaggle.com/competitions/badger-scribe): verbatim transcription of 19th-century archival pages with open-weight models.

- **Plan:** [PLAN.md](PLAN.md)
- **Challenge rules:** [ChallengeDescription.md](ChallengeDescription.md)
- **Approaches:** [design.md](design.md)
- **Metric definitions:** [measurements.md](measurements.md)
- **Numbers from runs:** [results/](results/)

Data lives in `badger-scribe-data/` (not committed). Packages are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`).

```bash
# Mac / CPU / MPS (no 4-bit)
uv sync

# Windows RTX 3060 (bitsandbytes 4-bit)
uv sync --extra cuda

uv run python benchmark/make_holdout.py
uv run python benchmark/infer_vlm.py --model churro --split smoke --run-id smoke-churro
uv run python benchmark/eval.py --run-id smoke-churro --split smoke
```

3060 7B/8B example:

```bash
uv run python benchmark/infer_vlm.py --model qwen2.5-vl-7b --split smoke --run-id smoke-qwen25vl-7b --device cuda --dtype 4bit
uv run python benchmark/eval.py --run-id smoke-qwen25vl-7b --split smoke --append-results
```

`--device auto` prefers CUDA, then MPS, then CPU. Do not mix Mac and 3060 wall-clock in one ranking; CER is comparable.

`uv pip install -r benchmark/requirements.txt` still works as a pip-style fallback. The lockfile is `uv.lock`.
