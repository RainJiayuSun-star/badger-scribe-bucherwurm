# OCR / HTR measurements

Common metrics for **accuracy** (did we read the page?) and **efficiency** (how much compute, time, and memory?). Formulas differ across tools. Two papers reporting “CER 0.08” can be incomparable if one divides by ground-truth length, another by `edits + correct`, and a third collapses Unicode differently.

For this project the **only ranking number** is the official Badger Scribe CER in `metric.py`. Everything else is diagnostic.

---

## 1. Accuracy

### 1.1 Edit distance (the shared core)

Almost every text metric starts from **Levenshtein distance**: the minimum number of single-character (or single-word) operations to turn the prediction into the ground truth, or vice versa.

Operations:

- **Insertion** (`i`) — extra character/word in the prediction
- **Deletion** (`d`) — character/word missing from the prediction
- **Substitution** (`s`) — wrong character/word

Rice (1996) made this the standard for OCR evaluation (ISRI tools). Implementations differ (`rapidfuzz`, `jellyfish`, custom DP). Always use one implementation when comparing runs.

Whitespace, Unicode normalization (NFC vs NFD), ligatures (`ſ` vs `s`, `ä` vs `a` + combining diaeresis), and punctuation stripping change the number. Neudecker et al. (2021) show that five common eval tools disagree on the same pages for exactly these reasons.

**Badger Scribe:** collapse all whitespace runs to a single space on both sides, then Levenshtein on the remaining characters. Casing, punctuation, and historical spelling count. `#` in the *reference* matches any predicted character (illegible, not scored). `#` in the *prediction* is a normal character.

### 1.2 Character Error Rate (CER)

The default HTR / historical-OCR metric. Lower is better.

**Reference-length CER** (Rice / most papers / Badger Scribe):

```
CER = (i + s + d) / n
```

`n` = number of characters in the **ground truth**. This can exceed 1.0 if the model dumps extra text.

**Capped CER** (Badger Scribe):

```
CER = min(1.0, levenshtein(pred, gt) / len(gt))
```

Empty or missing prediction → 1.0. Padding with junk cannot score worse than 1.0.

**Normalized CER** (OCR-D `CER_n`):

```
CER_n = (i + s + d) / (i + s + d + c)
```

`c` = number of correct (identity) characters. Bounded in `[0, 1]` without an arbitrary cap.

**Character accuracy** (ISRI):

```
Accuracy = 1 − CER
```

sometimes reported as a percentage. Churro (Semnani et al., 2025) reports **normalized Levenshtein similarity**:

```
NLS = 1 − levenshtein(pred, gt) / max(len(pred), len(gt))
```

Also in `[0, 1]`, higher better. Close to `1 − CER` when lengths match; more forgiving of length mismatch than reference-length CER.

| Variant | Denominator | Can exceed 1? | Used by |
| --- | --- | --- | --- |
| Reference-length CER | `len(gt)` | yes | Rice, Transkribus, most HTR papers |
| Capped CER | `len(gt)`, clip 1.0 | no | **Badger Scribe** |
| Normalized CER | `edits + correct` | no | OCR-D |
| NLS | `max(len(pred), len(gt))` | no | Churro-DS |

**Aggregation.** Mean of page CERs ≠ CER of concatenated text (short pages weigh the same as long ones). Badger Scribe: mean page CER **within** each category, then mean of those category means (macro-average). That stops a large German split from hiding English failure.

Report **median** as well as mean when a few ruined pages dominate (Hodel et al., 2021 do this for Kurrent).

### 1.3 Word Error Rate (WER)

Same idea at word granularity:

```
WER = (i_w + s_w + d_w) / n_words_gt
```

or the OCR-D normalized form with `+ c_w` in the denominator.

A **word** is usually a whitespace-delimited token after stripping edge punctuation (Unicode TR29). One wrong letter makes the whole word wrong, so WER is typically 2–4× CER on the same page.

Variants:

- **Word accuracy** = `1 − WER`
- **Non-stopword / significant-word accuracy** (Rice; Tanner, Muñoz, Ros 2009 on British Library newspapers) — ignore `the`, `and`, etc. Closer to “can a historian find names?”
- **Phrase accuracy** — exact match of *k*-word windows

**Badger Scribe:** WER is printed by `metric.py` as a diagnostic. It does not rank the leaderboard.

### 1.4 Order-independent text metrics

Page-level CER/WER assume a single reading order. On account books and multi-column pages, a correct transcription in the wrong order looks like a disaster.

| Metric | What it ignores | Notes |
| --- | --- | --- |
| **Bag-of-words WER (bWER / unordered WER)** | Sequence | Multiset of words; count mismatches. OCR-D; ocrevalUAtion. |
| **ΔWER = WER − bWER** | — | Tracks reading-order error. Correlates with NSFD (Romero et al., 2023). |
| **Hungarian WER (hWER)** | Rigid line alignment | Optimal word matching; ≈ bWER in practice. |
| **Flexible Character Accuracy (FCA / FCER)** | Global reading order | Clausner, Pletschacher, Antonacopoulos: chunk-wise edit distance so column swaps are not fully penalized. Used in ICDAR/PRImA competitions. |
| **NSFD** (Normalised Spearman’s Footrule Distance) | — | Explicit reading-order distance between region sequences. |

If page CER is high but bWER/FCA is low, the recognizer is fine and **layout/order** is the bug — the usual Dominy / survey-notes failure.

### 1.5 Layout, lines, and tables (pipeline metrics)

These score the CV stages, not the final string.

**Detection / segmentation**

- **IoU** (Intersection over Union) of predicted vs GT polygons/boxes.
- **Precision / Recall / F1** at an IoU threshold (often 0.5–0.7 for lines).
- **mAP** (mean Average Precision) over IoU thresholds — standard for YOLO / Mask R-CNN line detectors (Meunier et al., 2024).
- PRImA Layout Evaluation also counts **merge, split, miss, false alarm, misclassification**, and scenario-weighted errors (Clausner, Pletschacher, Antonacopoulos).

**Reading order**

- Pairwise region order correctness (Clausner et al.).
- NSFD / ΔWER as above.

**Tables** (relevant to Dominy books)

- **TEDS** (Tree-Edit-Distance-based Similarity) — HTML table trees; cell text compared by edit distance (Zhong, ShafieiBavani, Jimeno Yepes, 2020, PubTabNet).
- **TEDS-IoU** — same tree edit, but cell cost is box IoU, so OCR errors do not dominate structure score.

A pipeline should log line-F1 and a cheap bWER alongside page CER. The official metric will not tell you *which stage* broke.

### 1.6 Other accuracy-related scores

| Metric | Role |
| --- | --- |
| **BLEU / chrF** | Token n-gram overlap. Shows up in OCR *post-correction* papers (e.g. Kraken+ByT5). Not a substitute for CER on verbatim HTR. |
| **Figure of Merit** (IMPACT / NCSR) | Weighted edit distance: substitutions cost ~5× deletions, as a proxy for **human correction effort**. |
| **Confidence / lexicon “simple quality”** | GT-free: fraction of words in a dictionary, or mean engine confidence. Weakly correlated with CER (Neudecker et al., 2021). Historical spelling makes lexicons lie — do not use this as a train/eval target here. |
| **Second-system agreement** | Two models’ CER against each other as a stand-in for confidence on unlabelled pages. Organizers mention this; not validated. Useful for the writeup’s “scaling to unlabelled pages” section. |
| **olmOCR-Bench unit tests** | Programmatic checks (reading order, table structure, math) rather than one CER. Different task (modern PDFs). |
| **Downstream NLP** | NER F1 collapses when WER goes from ~1% to ~7% (Hamdi et al., cited in Neudecker et al.). Not our leaderboard, but it is why libraries care about CER at all. |

### 1.7 What we should log locally

Always compute with `metric.py` so local CER ≡ leaderboard CER.

Also keep, per page and per `category` × `label_source`:

- CER (official)
- WER (official diagnostic)
- Median CER (outlier-robust)
- Optional: bWER or FCA if we have a pipeline (order vs recognition)
- Line-level IoU / F1 if we segment
- Confusion notes (Kurrent `e/n/u`, long-s, dates) — qualitative, not a metric

Do **not** optimize against `silver_claude` CER as if it were gold.

---

## 2. Efficiency

Accuracy without cost is incomplete. Libraries care whether a pipeline runs overnight on one GPU. Badger Scribe writes `peak_vram` and `eval_wall_clock` on the submission card (informational, not scored) and hard-caps inference at **one GPU, ≤ 96 GB VRAM**.

OCR-D’s QA spec already treats resource use as a first-class evaluation axis: CPU time, wall time, I/O, memory, disk; GPU peak/average memory listed as “not in use yet” but recommended.

### 2.1 Time

| Metric | Definition | Why it matters |
| --- | --- | --- |
| **Wall-clock time** | Elapsed real time, start → last page written | What a librarian waits for. **Badger Scribe `eval_wall_clock`:** full test set, one number. |
| **CPU time** | Sum of process CPU seconds | Can exceed wall time with threads; useful for CPU OCR (Tesseract). |
| **Latency (per page)** | Time to finish one page | Mean is not enough. Report **P50 / P95 / P99**. P95 is the production SLA number (NVIDIA NIM OCR docs; typical Document-AI practice). |
| **Throughput** | Pages / second (or / hour, / day) | Inverse of mean latency only if there is no batching/queueing. NVIDIA reports images/s at a fixed batch and concurrency. |
| **Time-to-first-token / generation time** | VLM-specific | Decoder length ≈ page length; dense account books are slower than short letters. |
| **Training GPU-hours** | Wall time × GPUs for SFT/LoRA | Prep is unconstrained; still log it for the writeup. Official starter: ~1 GPU-hour for a reported holdout drop. |

Measure **end-to-end** (image load → CSV row) and, if useful, **stage times** (segment / recognize / decode). Production write-ups (Blue Guardrails on VLM OCR; microservice Document-AI papers) find that layout or CPU pre/post often dominates, not the recognizer.

Exclude model download and first-iteration warmup from the official-style wall-clock, or report them separately.

### 2.2 Memory and hardware

| Metric | Definition | Why it matters |
| --- | --- | --- |
| **Peak VRAM** | Max allocated GPU memory during the eval run | **Hard constraint ≤ 96 GB.** Submission card field. Sequential load/unload is allowed; measure the peak of the *resident* model, not the sum of all stages if they never coexist. |
| **Average VRAM** | Mean over the run | Utilization, not the constraint. |
| **System RAM / disk** | Host memory, scratch for images | OCR-D lists these; huge TIFFs and PageXML can OOM eval tools themselves (Neudecker: CER on 60k-character newspaper pages crashed several tools). |
| **Model size** | Parameters; checkpoint GB on disk | 3B vs 72B; quantization (INT8/INT4) trades CER for VRAM. |
| **Tokens in / out** | VLM context and generated tokens per page | Drives both latency and VRAM (KV cache). Cap image pixels / max new tokens. |

`nvidia-smi` / `torch.cuda.max_memory_allocated()` are the usual peak-VRAM tools. Reset the peak counter at the start of the eval script.

### 2.3 Cost and energy (optional, useful for the writeup)

| Metric | Typical unit |
| --- | --- |
| **Cost per 1,000 pages** | USD, including GPU-hours, storage, egress |
| **Energy** | kWh or gCO₂e per 1,000 pages (rarely reported; WattBot-adjacent) |
| **Human correction time** | Minutes per page at a given CER — the number libraries actually budget. IMPACT Figure of Merit is a crude automatic proxy. |

Churro reports a cost comparison vs Gemini (15.5× cheaper at higher NLS). We can put a back-of-envelope `$ / 1k pages` next to `eval_wall_clock` if we know the GPU rate.

### 2.4 What we should log for every eval run

```
n_pages
wall_clock_sec          # full set, excluding download/warmup
sec_per_page_p50_p95
pages_per_hour
peak_vram_gb            # torch / nvidia-smi
model_id + quantization
max_new_tokens / image_pixels   # if VLM
hardware                # e.g. RTX 4090 24 GB
```

Same hardware and same image resolution when comparing two models. Throughput without batch size and GPU name is not a number.

---

## 3. How metrics interact (and how they lie)

**Accuracy vs efficiency.** A 72B VLM may beat a 3B Churro zero-shot and still lose as a library tool if it is 20× slower and needs 80 GB. The challenge scores only CER; the writeup should still report VRAM and wall-clock.

**Line CER ≠ page CER.** TrOCR-kurrent’s ~2.7% is line-level on similar hands. Page CER includes missed lines, order errors, and unseen hands. Do not quote line papers as predicted leaderboard scores.

**Macro vs micro.** Macro-CER (Badger Scribe) treats categories equally. Micro-CER (concatenate everything) would let German volume dominate.

**Normalization.** Verbatim CER punishes helpful modernization. BLEU and lexicon scores reward it. Train and select models with the official metric.

**Confidence is not CER.** Engine softmax / VLM logprobs correlate poorly with edit rate on historical pages (Neudecker et al.).

**Tool disagreement.** If we ever compare to published CERs, say which tool and which Unicode/whitespace rules. Prefer `metric.py` for all internal tables.

---

## 4. Quick map for this repo

| Question | Metric |
| --- | --- |
| Are we winning the competition? | Macro-averaged capped page CER (`metric.py`) |
| Is English or German the hole? | Per-category CER |
| Are silver labels fooling us? | CER on `human` vs `silver_claude` |
| Is it recognition or reading order? | CER vs bWER / FCA; line F1 |
| Can a library run this? | Peak VRAM ≤ 96 GB, `eval_wall_clock`, pages/hour, P95 latency |
| Is the 7B LoRA worth it vs 3B zero-shot? | ΔCER vs Δwall-clock vs ΔVRAM |

---

## References

Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and reversals. *Soviet Physics Doklady*, 10(8), 707–710.

Rice, S. V. (1996). *Measuring the Accuracy of Page-Reading Systems*. PhD thesis, University of Nevada, Las Vegas. Basis of the ISRI OCR evaluation tools (character / word / non-stopword / phrase accuracy).

Rice, S. V., Kanai, J., & Nartker, T. A. (1995). An evaluation of OCR accuracy. ISRI Technical Report. University of Nevada, Las Vegas.

Holley, R. (2009). How good can it get? Analysing and improving OCR accuracy in large scale historic newspaper digitisation programs. *D-Lib Magazine*, 15(3/4). Australian Newspaper Digitisation Program.

Tanner, S., Muñoz, T., & Ros, P. H. (2009). Measuring mass text digitization quality and usefulness: Lessons learned from assessing the OCR accuracy of the British Library’s 19th Century Online Newspaper Archive. *D-Lib Magazine*, 15(7/8). Introduces significant-word accuracy for newspapers.

Carrasco, R. C. (2014). An open-source OCR evaluation tool. In *DATeCH 2014* (pp. 179–184). ACM. https://doi.org/10.1145/2595188.2595221

Clausner, C., Pletschacher, S., & Antonacopoulos, A. (2011). Scenario driven in-depth performance evaluation of document layout analysis methods. In *ICDAR*. PRImA layout evaluation (merge/split/miss, reading order).

Clausner, C., Pletschacher, S., & Antonacopoulos, A. (2020). Flexible character accuracy measure for reading-order-independent OCR evaluation. *Pattern Recognition Letters*. https://www.primaresearch.org/www/assets/papers/PRL_Clausner_FlexibleCharacterAccuracy.pdf

Pletschacher, S., & Antonacopoulos, A. (2010). The PAGE (Page Analysis and Ground-truth Elements) format framework. In *ICPR* (pp. 257–260).

Neudecker, C., Baierer, K., Gerber, M., Clausner, C., Antonacopoulos, A., & Pletschacher, S. (2021). A survey of OCR evaluation tools and metrics. In *HIP ’21* (pp. 13–18). ACM. https://doi.org/10.1145/3476887.3476888 — PDF: https://www.primaresearch.org/www/assets/papers/HIP21_CNeudecker_OcrEvalSurvey.pdf

OCR-D coordination project. Quality Assurance in OCR-D (evaluation metrics: CER, WER, bag-of-words, IoU, wall/CPU time, memory; planned GPU metrics). https://ocr-d.de/en/spec/ocrd_eval

Hodel, T., Schoch, D., Schneider, C., & Vogeler, G. (2021). General models for handwritten text recognition: Feasibility and state-of-the-art. German Kurrent as an example. *Journal of Open Humanities Data*, 7, 13. https://doi.org/10.5334/johd.46 — reports mean and median CER.

Romero, V., et al. (2023). End-to-end page-level assessment of handwritten text recognition. arXiv:2301.05935. WER vs bag-of-words WER (bWER), Hungarian WER, NSFD, ΔWER.

Zhong, X., ShafieiBavani, E., & Jimeno Yepes, A. (2020). Image-based table recognition: Data, model, and evaluation (TEDS). In *ECCV*. PubTabNet.

Raja, S., Mondal, A., & Jawahar, C. V. (2022). Evaluating table structure recognition: A new perspective (TEDS-IoU). arXiv:2208.00385.

Meunier, J.-L., et al. (2024). Historical text line segmentation using deep learning algorithms: Mask-RCNN against U-Net networks. *Journal of Imaging*, 10(3), 65. https://doi.org/10.3390/jimaging10030065 — line segmentation IoU / impact on HTR.

Momtaz, Y., et al. (2025). Modular pipeline for text recognition in early printed books using Kraken and ByT5. *Electronics*, 14(15), 3083. Example of CER + BLEU for OCR plus post-correction.

Semnani, S. J., Zhang, H., He, X., Tekgürler, M., & Lam, M. S. (2025). Churro: Making history readable with an open-weight large vision-language model for high-accuracy, low-cost historical text recognition. arXiv:2509.19768. Normalized Levenshtein similarity; cost vs closed VLMs.

Poznanski, J., et al. (2025). olmOCR 2. Allen AI. https://allenai.org/blog/olmocr-2 — unit-test-style document OCR evaluation; open 7B page model.

Endemann, C., Prater, S., & Chovanec, K. (2026). *Badger Scribe*. Kaggle. https://www.kaggle.com/competitions/badger-scribe — official capped, whitespace-normalized, category-macro CER; WER diagnostic; `metric.py`; writeup fields `peak_vram`, `eval_wall_clock`.

IMPACT Centre of Competence. ocrevalUAtion. https://github.com/impactcentre/ocrevalUAtion — CER/WER and unordered WER; Figure of Merit lineage via NCSR tools.

Qurator / Staatsbibliothek zu Berlin. dinglehopper. https://github.com/qurator-spk/dinglehopper — CER/WER with visual alignment for historical OCR.

ISRI OCR Evaluation Tools (Unicode update). https://github.com/eddieantonio/ocreval — modernized Rice tools.

Unicode Consortium. Unicode Standard Annex #29: Unicode Text Segmentation (word boundaries used when defining WER tokens). https://unicode.org/reports/tr29/

NVIDIA. Performance for NIM for Image OCR (NeMo Retriever OCR). https://docs.nvidia.com/nim/ingestion/image-ocr/latest/performance.html — P50 latency (ms) and throughput (images/s) by GPU; batch vs latency modes.

NVIDIA. Optimization for NIM for Image OCR. https://docs.nvidia.com/nim/ingestion/image-ocr/2.0.0/optimization.html — VRAM vs batch size / engine count.

Blue Guardrails. Open source OCR with vision language models: High throughput and low cost. https://blueguardrails.com/en/blog/high-throughput-vlm-ocr — pages/s, stage-level wall-clock, CPU vs GPU bottlenecks.

Operationalizing Document AI: A microservice architecture for OCR and LLM pipelines in production. arXiv:2605.18818. Per-page GPU latency, P95 document latency, peak RAM.

Skywork / production OCR ops notes. DeepSeek-OCR cloud deployment: best practices. Cost per 1,000 pages; P95/P99; pages/minute. https://skywork.ai/blog/ai-agent/deploying-deepseek-ocr-cloud-production-best-practices/
