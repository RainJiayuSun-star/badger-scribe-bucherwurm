# Cooler Badger Scribe

UW–Madison MLM26 / [Kaggle Badger Scribe](https://www.kaggle.com/competitions/badger-scribe): verbatim transcription of 19th-century archival pages with open-weight models.

- **Plan:** [PLAN.md](PLAN.md)
- **Challenge rules:** [ChallengeDescription.md](ChallengeDescription.md)
- **Approaches:** [design.md](design.md)
- **Metric definitions:** [measurements.md](measurements.md)
- **Numbers from runs:** [results/](results/)

Data lives in `badger-scribe-data/` (not committed). Freeze the local val split, then infer and score:

```bash
python benchmark/make_holdout.py
python benchmark/infer_vlm.py --model churro --split smoke --run-id smoke-churro
python benchmark/eval.py --run-id smoke-churro --split smoke
```

On the Windows RTX 3060 (CUDA, 4-bit for 7B/8B):

```bash
pip install -r benchmark/requirements.txt bitsandbytes
python benchmark/infer_vlm.py --model qwen2.5-vl-7b --split smoke --run-id smoke-qwen25vl-7b --device cuda --dtype 4bit
python benchmark/eval.py --run-id smoke-qwen25vl-7b --split smoke --append-results
```

`--device auto` already prefers CUDA, then MPS, then CPU. Do not mix Mac and 3060 wall-clock in one ranking; CER is comparable.
