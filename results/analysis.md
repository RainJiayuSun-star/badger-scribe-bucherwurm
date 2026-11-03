# Wave 1 analysis

Zero-shot comparison of six open-weight VLMs plus one non-VLM pipeline on the frozen local
holdout. Numbers come from [wave1.csv](wave1.csv) / [wave1.md](wave1.md); the metric is the
official category-macro capped CER from `badger-scribe-data/metric.py`.

## Dataset and the portion used

The released `train.csv` is 354 labelled pages across 23 documents and three categories. It is
badly imbalanced, and category is almost perfectly aligned with label quality:

| Category | Pages | human | silver_claude |
| --- | --- | --- | --- |
| kade_letters (German) | 295 | 295 | 0 |
| dominy_accounts (English) | 36 | 2 | 34 |
| survey_notes (English) | 23 | 1 | 22 |
| **Total** | **354** | **298** | **56** |

Page transcripts average 619 characters (median 598, range 27–1998).

`make_holdout.py` freezes a deterministic document-level split — whole documents go to val, so no
document appears on both sides:

| Split | Pages | Composition | Used for |
| --- | --- | --- | --- |
| train | 259 | 218 Kade / 26 Dominy / 15 Survey | **untouched**, reserved for LoRA |
| val | 95 | 77 Kade / 10 Dominy / 8 Survey, 8 documents | all ranking numbers |
| smoke | 12 | 4 / 4 / 4, subset of val | fast pre-flight before a full run |

**So Wave 1 scored 95 of 354 pages (27%).** The remaining 259 training pages were deliberately
never inferred on, to keep them clean for fine-tuning. Kaggle's own test set was never touched.

## Protocol

Every model got the same treatment, which is what makes the comparison meaningful:

- **Zero-shot**, one shared verbatim prompt (`benchmark/prompts.py`), no few-shot examples.
- **Greedy decoding** (`do_sample=False`), `max_new_tokens=2048`, image budget capped at
  1,003,520 pixels so no model gets a resolution advantage.
- **One GPU** (RTX 3060, 12 GB). 3B models in bf16; 7B/8B in 4-bit NF4 via bitsandbytes, which is
  the only way they fit. Quantization is therefore confounded with size — see caveats.
- 14 runs total (7 smoke + 7 val) = 749 page transcriptions.

Two models deviate, both recorded in their `runs/<run-id>/config.json`:

- **Churro** additionally ran *with* catalog metadata appended to the prompt, as an ablation.
- **DeepSeek-OCR** is task-prompted, not instruction-following. The shared prompt produced empty
  output, so it ran on `Free OCR.` and is smoke-only. Its numbers are not like-for-like.

## Results (val, 95 pages)

| run | macro CER | Kade | Dominy | Survey | median CER | WER | sec/page | peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| olmocr-2-7b-zeroshot | **0.2956** | 0.4873 | 0.3216 | **0.0779** | 0.4392 | 0.7721 | **8.57** | 6041 |
| churro3b-zeroshot | 0.3189 | **0.2698** | 0.3743 | 0.3125 | **0.1698** | **0.4577** | 12.24 | 7401 |
| qwen25vl-7b-zeroshot | 0.3447 | 0.5129 | 0.4023 | 0.1188 | 0.4493 | 0.7674 | 11.79 | 6041 |
| churro3b-zeroshot-meta | 0.3938 | 0.3135 | 0.4442 | 0.4237 | 0.1799 | 0.5005 | 14.59 | 7402 |
| qwen3vl-8b-zeroshot | 0.4461 | 0.6055 | 0.6185 | 0.1142 | 0.4956 | 0.8317 | 31.22 | 6557 |
| qwen25vl-3b-zeroshot | 0.4585 | 0.5564 | 0.4861 | 0.3331 | 0.4825 | 0.8347 | 11.40 | 7401 |
| pipeline-kraken-trocr | 0.8930 | 0.9668 | 0.8319 | 0.8803 | 0.9694 | 0.9938 | 0.37 | — |

Smoke-only: `smoke-deepseek-ocr` 0.9289 (Kade 1.0000).

## Findings

**1. The winner depends on the aggregation, and the two answers disagree.**
olmOCR wins macro CER (0.2956 vs 0.3189). Churro wins median page CER by a wide margin
(0.1698 vs 0.4392) and WER (0.4577 vs 0.7721). Both are correct: macro-averaging gives each
category one-third weight, but 77 of 95 pages are Kade letters, where Churro is far stronger.
olmOCR is better on the *official metric*; Churro is better on *most actual pages*.

**2. The models have near-opposite strengths.**
Churro leads Kade German (0.2698 vs olmOCR's 0.4873); olmOCR leads survey notes by 4x
(0.0779 vs 0.3125). A per-category router — Churro for Kade, olmOCR for Dominy and Survey —
would score roughly 0.2231 macro CER, beating every single model here. This is the strongest
Wave 2 lead.

**3. Catalog metadata hurts, consistently.**
Churro with metadata degrades in all three categories (macro 0.3189 → 0.3938), worst on survey
notes (0.3125 → 0.4237). Adding archival context biases the model toward what it expects the
page to say, which is the opposite of what a verbatim metric rewards. **Do not ship metadata in
the prompt.**

**4. Newer and bigger is not better.**
Qwen3-VL-8B is worse than Qwen2.5-VL-7B (0.4461 vs 0.3447) and ~3x slower (31.2 vs 11.8
sec/page). Among Qwen models, the 3B is worst, so scale helps within a generation but the
generation jump did not.

**5. The Kraken/TrOCR pipeline is not competitive.**
0.8930 macro CER, effectively a failure. Two causes: Kraken never ran (not installed), so all 107
pages fell back to projection segmentation, and that segmenter found only one line on most pages
(most `kade_007_p*` pages produced 7–33 characters against ~600-character references). It is 20x
faster than any VLM, which is its only redeeming property.

**6. DeepSeek-OCR does not transfer to handwriting.**
0.9289 macro CER and exactly 1.0000 on Kade, with output degenerating into repeated loops. It is
a printed-document model.

## Caveats

- **Label quality is perfectly confounded with category.** In val, all 77 human-labelled pages are
  Kade and all 18 silver_claude pages are Dominy or Survey. `human_cer_mean` is therefore
  *identical* to `kade_cer_mean` in every row, and nothing here can separate "model is worse at
  English accounts" from "silver labels are noisier". Dominy and Survey CER are partly scoring
  Claude, not the ground truth.
- **Small per-category samples.** Survey CER rests on 8 pages and Dominy on 10, across only 8
  documents. Treat the non-Kade columns as indicative, not stable.
- **Quantization is confounded with model size.** Every 7B/8B ran in 4-bit and every 3B in bf16,
  so some of the large models' weakness may be quantization damage rather than the model.
- **Timings are single-run, one GPU, no batching**, and `sec/page` for resumed runs covers only
  the pages computed in that session.
- Smoke (12 pages) roughly tracked val at the top of the ranking but not exactly — it put
  Qwen2.5-VL-7B (0.3190) ahead of Churro. It is a pre-flight check, not a predictor.

## Next steps

1. Build the per-category router from finding 2 — the cheapest large win available.
2. LoRA Churro on human-labelled German (Kade train documents), per the existing pick in
   `wave1.md`. Churro's Kade lead and low median CER make it the right base despite olmOCR's
   headline win.
3. Re-check Dominy and Survey against human labels before trusting those columns.
4. Optionally install Kraken to see whether real segmentation rescues the pipeline; it would have
   to close a very large gap to matter.
