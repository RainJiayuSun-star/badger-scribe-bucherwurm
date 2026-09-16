# Design notes: approaches to archival transcription

Literature review for Badger Scribe. Two families of systems, then production/community systems (companies, archives, shared model hubs), then a model shortlist for 19th–early-20th-century English and German Kurrent.

Numbers below are **not interchangeable**. Line-level CER on similar hands is much easier than page-level CER on an unseen collection. Treat published scores as relative evidence, not a predicted leaderboard number.

---

## Why this split matters here

Badger Scribe scores a **full page transcription** (verbatim CER, macro-averaged over categories). German Kurrent is most of the launch test set. English pages are faded survey notes and dense account books. The submitted run must be open-weight and fit **one GPU ≤ 96 GB**.

That maps cleanly onto two research traditions:

1. **Modular HTR pipeline** — classical document CV plus a line recognizer (the digital-humanities default: Transkribus, eScriptorium / Kraken).
2. **Single image-to-text model** — one network sees the page (or a crop) and emits the transcription.

A third pattern exists in recent papers — OCR draft + VLM correction — but it is a hybrid of the two, not a third independent family.

---

## 1. Pipeline: traditional CV + a deep recognition model

This is the dominant archival workflow for the last decade. Platforms such as **Transkribus** and **eScriptorium** implement it as: preprocess → layout / regions → text-line detection → line recognition → reading-order assembly → optional language-model or transformer post-correction.

Romein et al. (2025) and Hodel et al. (2021) treat this modular stack as the current production standard for historical HTR. Stanford’s Churro paper (Semnani et al., 2025) still describes collection-specific line/paragraph models as the state of the art in libraries, with annotation cost as the bottleneck.

### 1.1 Typical stages

| Stage | Job | Classical CV | Deep-learning replacements |
| --- | --- | --- | --- |
| Preprocess | Make ink readable | Deskew, crop, contrast; Sauvola / Niblack / Otsu binarization; bleed-through filters | Learned denoisers; many modern recognizers prefer the raw scan |
| Page / region layout | Find text blocks, tables, margins, illustrations | Projection profiles, connected components, XY-cut | dhSegment, Doc-UFCN, YOLO, Mask R-CNN |
| Line segmentation | Extract baselines or line polygons | Adaptive projection, contour following | ARU-Net, Kraken segmenter, Doc-UFCN, Mask R-CNN |
| Recognition | Image crop → character sequence | HMM / DTW (pre-2015) | CNN–LSTM–CTC (PyLaia, Kraken, Calamari, HTR+); TrOCR; Party |
| Reading order | Lines → page text | Geometric sort (top-to-bottom, LTR/RTL) | Learned order; VLMs do this implicitly |
| Post-correction | Fix systematic OCR errors | Dictionary / n-gram LM | ByT5, character LMs, open-weight LLM (allowed only if open-weight) |

Error **compounds**. A missed line never reaches the recognizer. A swapped reading order looks like a large CER even if every glyph is right. That is the main risk on Dominy account books and survey notebooks.

### 1.2 Preprocessing

Historical scans are faded, stained, and often show bleed-through from the verso.

- **Local adaptive thresholding** (Sauvola, Niblack) is the classical fix when the background is uneven (Sauvola & Pietikäinen, 2000).
- Momtaz et al. (2025) put a bleed-through removal stage (non-local means, GMM, biweight, Gaussian blur) in front of Kraken on 15th-century incunabula and report CER falling from ~38% to ~15% after Kraken + ByT5 correction. That corpus is printed, not Kurrent, but the degradation problem is the same class.
- **Do not assume binarization helps.** dhSegment and Doc-UFCN take RGB/gray pages. Aggressive thresholding can erase pencil survey notes. For Badger Scribe, treat preprocess as an ablation: raw vs. contrast-enhanced vs. binarized, split by category.

### 1.3 Layout and line segmentation

Line detection is the hard CV step on these collections. Survey notes are irregular; account books are tabular; letters have margins and insertions.

**U-Net family (pixel labeling, then connected components):**

- **dhSegment** (Oliveira et al., 2018) — ResNet-50 encoder–decoder; page extraction, baselines, layout classes. Generic historical-document tool; slower at inference.
- **Doc-UFCN** (Boillet et al., 2020) — smaller U-shaped FCN trained from scratch on documents; faster than dhSegment; used for line and region detection.
- **ARU-Net** — attention residual U-Net, common baseline for historical baselines.

Boillet et al. (2022) show that generic models trained on many historical sets can segment unseen pages, but **annotation conventions must be unified** or downstream HTR suffers.

**Instance-segmentation / detection family:**

- **Mask R-CNN** vs U-Net on historical line segmentation (Meunier et al., 2024): Mask R-CNN beat ARU-Net, dhSegment, and Doc-UFCN on cBAD, DIVA-HisDB, HOME-Alcar, and a private register set, and the better masks also improved HTR. They argue evaluation should not stop at raw masks.
- **YOLO** (v8/v11 class) is now a common layout + line detector in new toolkits (DocWorkflow; see Moins, 2025). Fast, easy to fine-tune on a few dozen pages.

**Kraken / eScriptorium segmenter** (Kiessling et al., 2019): baseline detector trained for historical material; still the open-source default in DH labs. Output is PageXML / ALTO, which every later recognizer can consume.

For this challenge:

- Letters (Kade): roughly regular prose lines → Kraken or Doc-UFCN is enough.
- Survey notes: pencil, terse, uneven → expect missed short lines; Mask R-CNN or a YOLO fine-tune on a handful of train pages is worth it.
- Dominy books: columns, sums, ditto marks → region classes (body vs. totals vs. headings) matter more than a single “text line” class.

### 1.4 Line / region recognizers (the “deep learning model” in the pipeline)

Once you have a line crop, recognition is a sequence model.

**CNN–LSTM–CTC (the workhorse):**

- **PyLaia** (Puigcerver, 2017/2024) — CNN + stacked LSTMs + CTC; the open engine inside Transkribus for years. Fine-tunes on a few thousand lines. Weak zero-shot on a new script.
- **Kraken recognizer** — same family, designed for historical / low-resource scripts; trains from PageXML.
- **Calamari** — voting ensemble of CRNNs; strong on historical print.
- **HTR+** (Planet AI / Rostock) — CNN–LSTM with heavier line normalization. Strong published Kurrent numbers; **not openly deployable** the way it once was (license; removed from Transkribus). Treat as literature, not a submission engine.
- **IDA** (Planet AI) — CNN + LSTM or Conformer; proprietary.

Hodel, Schoch, Schneider, and Vogeler (2021), *General Models for Handwritten Text Recognition*, is the key Kurrent paper in this family. On **unseen 19th-century Kurrent hands**:

| Model | Engine | Mean CER | Median CER | Worst |
| --- | --- | --- | --- | --- |
| Large in-domain Kurrent model | HTR+ | 1.73–3.41% (seen / related) | — | — |
| Transkribus German Kurrent | HTR+ | 5.90% | 4.85% | 10.20% |
| Generic PyLaia (not script-matched) | PyLaia | 18.77% | 13.30% | 51.05% |

Lesson: a **script-matched** CTC model is excellent; a generic one is not. For Kade letters, a Kurrent-specific PyLaia/Kraken fine-tune on the 436 expert pages is a serious baseline.

**Transformer line recognizers (still pipeline — they need a line image):**

- **TrOCR** (Li et al., 2023, AAAI) — BEiT/ViT encoder + RoBERTa decoder; no CNN, no CTC. IAM line CER: Small 4.22%, Base 3.42%, Large 2.89%. Needs a **line crop**; a full page as input fails.
- **dh-unibe/trocr-kurrent** (Widmer & Hodel) — TrOCR-base-handwritten fine-tuned on ~293k 19th-century Kurrent lines (Zürich protocols, Humboldt notes, Huber diary, Semper, Greifswald, etc.). **Eval CER 2.83%, test CER 2.66%.** Closest public specialist to the Kade collection. Line-level, similar-hand split — expect worse on unseen immigrant letters, but this is the obvious Kurrent recognizer.
- **dh-unibe/trocr-kurrent-XVI-XVII** — earlier centuries; test CER 5.42%. Secondary for 1930s Kade letters.
- **TrOCR-f / Titan** (Romein et al., 2025) — fine-tuned TrOCR and Transkribus’s Titan “supermodel” win Latin-script historical sets out of the box. Titan is **proprietary** (closed weights) → disallowed in the submitted run.

**Page-aware but still modular recognizer:**

- **Party** (Kiessling, “PAge-wise Recognition of Text-y”) — Swin encoder + tiny Llama decoder (~40–100M), octet tokenizer, **baseline positional prompts**. Full-page image + line geometry → text. Trained on a large multilingual historical mix (German manuscripts included). Needs PageXML/ALTO lines first. Language tokens help on mixed pages. Author says **fine-tune to match transcription guidelines**. Fits this challenge’s “verbatim, not modernized” rule.

### 1.5 Post-correction

Momtaz / Electronics 2025: Kraken OCR + **ByT5** line-level correction on early print, CER 38% → 15%. Gutteridge et al. (2025), *Judge a Book by Its Cover* (COLM 2026): on multi-page handwriting, **OCR + one page image in a VLM** (`ocr+page1` / `ocr+pageN`) matched or beat end-to-end MLLM transcription while using less vision context. That is a pipeline idea we can reuse with an **open** VLM as the corrector (Qwen), never a closed API on test images.

Hazard for this metric: silver Claude labels and LLM correctors **modernize spelling**. Post-correction must be constrained to “fix glyphs, do not copy-edit.”

### 1.6 Strengths and weaknesses for Badger Scribe

**Strengths**

- Best published numbers on **Kurrent** are in this family (TrOCR-kurrent, Transkribus Kurrent).
- Cheap to run; easy to swap one stage.
- Line errors are inspectable (you can see which crop failed).
- Fine-tuning a recognizer on a few thousand lines is well understood.

**Weaknesses**

- Layout failure on tables / pencil notes dominates CER.
- Reading order is a separate model you can get wrong.
- You must maintain three models (layout, lines, recognizer) under 96 GB (sequential load is fine).
- English train labels are mostly silver; a CTC model will copy their modernization if you train naively.

**Open pipeline stacks worth cloning**

- eScriptorium + Kraken
- DocWorkflow (YOLO layout/lines → Kraken or VLM-per-line)
- medieval-ocr-pipeline (Kraken → TrOCR → ByT5) — architecture, not the medieval weights

---

## 2. Single model: image in, transcription out

One network produces the page text. No explicit line boxes at inference (the model may still have been trained with them). This is what the challenge starters assume, and what Churro / olmOCR argue is replacing pipelines for page-level work.

Useful subcategories:

1. Specialized HTR / OCR transformers (narrow task, often still happiest on a line or cropped region)
2. Document-foundation / OCR-specialized VLMs (trained to dump a page)
3. General multimodal foundation VLMs (instruction-tuned, OCR is one skill)
4. Domain-adapted historical VLMs (general VLM + historical fine-tune)

### 2.1 Specialized HTR / OCR transformers

These are “one model” at recognition time, but many were designed for **line images**. On a full Badger Scribe page they need either (a) a silent crop/resize that destroys layout, or (b) you admit they are the recognizer in a pipeline.

| Model | Size | Input | Why it might work on old writing | Risk |
| --- | --- | --- | --- | --- |
| **microsoft/trocr-large-handwritten** | 558M | Line | Strong IAM CER (2.89%); good English cursive prior | Modern IAM ≠ 1830s survey pencil or Kurrent |
| **dh-unibe/trocr-kurrent** | ~334M | Line | **Best public 19th-c. Kurrent specialist** (CER ~2.7% on its test) | Needs line seg; unknown on Kade immigrant hands |
| **Donut** (Kim et al., 2022) | ~200–400M | Page | Swin + BART, no OCR engine; trained for document parsing | Weak on handwriting in independent Gothic/HTR bakeoffs |
| **GOT-OCR2** (Wei et al., 2024) | 580M | Page / slice | Unified “OCR-2.0”; trained with IAM, CASIA, NorHand pages | Built for general characters (tables, formulas); not Kurrent-specific |
| **Party** | Swin + ~40M Llama | Page + baselines | Pretrained on historical DE/EN/FR/LA manuscripts | Needs geometry; not a pure single crop→text model |
| **PaddleOCR / PARSeq / ABINet** | small | Line / word | Strong scene-text priors | Scene text ≠ archival ink |

**Verdict:** for a *pure* single-model page submission, TrOCR and Party are the wrong interface unless you wrap them. They remain the strongest **specialists** if you allow an implicit line detector. Donut/GOT are true page models but are not the current historical-HTR leaders.

### 2.2 Document-foundation / OCR-specialized VLMs

These are VLMs (or VLM fine-tunes) whose training objective is “read this page.” They are the closest drop-in to a single Badger Scribe model.

**Churro (Semnani et al., 2025; Stanford OVAL)**  
Fine-tune of **Qwen2.5-VL-3B** on Churro-DS: ~100k pages, 155 historical corpora, 46 language clusters, 3rd century BCE–20th century, diplomatic page-level text. Open weights.

On Churro-DS (normalized Levenshtein similarity, higher is better):

- Printed 82.3%, handwritten **70.1%** — beats Gemini 2.5 Pro (80.9% / 63.6%) and all other VLMs they tested.
- vs. base Qwen2.5-VL-3B: +14.5 printed, **+27.2 handwritten**.
- **German handwriting 81.4** vs Gemini 45.6 in their per-language examples — the single most relevant published VLM result for Kurrent-like German.
- English handwriting reported very high on their slice (treat as in-domain historical English, not IAM).

They also zero-shot evaluated open VLMs on the same historical test. Approximate handwritten averages:

| Model | Handwritten NLS (avg) | Notes |
| --- | --- | --- |
| Churro 3B (fine-tuned) | 70.1 | Best; historical specialist |
| Gemini 2.5 Pro | 63.6 | Closed — prep only |
| **Qwen2.5-VL-72B** | **54.5** | Best *general* open VLM in that table |
| NuMarkdown 8B | 51.2 | OCR-specialized |
| RolmOCR 8B | 49.0 | OCR-specialized |
| Nanonets-OCR 3B | 43.2 | OCR-specialized |
| **olmOCR 8B** | **41.5** | Strong on modern/print PDFs; weaker on historical HW |
| Gemma 3 27B | 34.1 | Weak historical HTR |

Churro’s own finding: **OCR-specialized VLMs win on print; a large general VLM (Qwen 72B) wins on handwriting** unless you fine-tune on historical pages. That is the central design claim for approach 2.

**olmOCR 2** (AllenAI, 2025; `olmOCR-2-7B-1025`)  
Qwen2.5-VL-7B fine-tuned on 270k PDF pages, including **+20k hard handwritten/typewritten** pages. Apache-2.0 weights, data, and trainer. Page → Markdown in one pass. They specifically advertise a fix on **Lincoln’s 1864 handwriting**. Excellent open document OCR; Churro’s table says it is **not** the best zero-shot historical HTR (41.5 handwritten NLS). Still a top candidate to **fine-tune** on Badger train pages because the recipe (SFT + RL on Qwen2.5-VL-7B) matches the official starter.

**RolmOCR, Nanonets-OCR, NuMarkdown**  
8B / 3B page OCR VLMs. Competitive on printed historical pages in Churro-DS; mid-pack on handwriting. Useful as small ensemble members, not as the Kurrent bet.

**LightOnOCR / other ~4B OCR VLMs**  
Low VRAM page OCR. Worth a zero-shot smoke test; little published Kurrent evidence.

### 2.3 General multimodal foundation VLMs

Instruction-tuned models that happen to be good at reading. The challenge writeup example is `Qwen/Qwen2.5-VL-7B-Instruct` for a reason.

**Qwen2.5-VL (3B / 7B / 72B)** — Bai et al., 2025  
Native variable-resolution images (important for dense account books). Official starter family. 7B BF16 ~13 GB; 72B needs quantization to fit 96 GB comfortably. Churro: 72B is the strongest **open zero-shot** historical handwriting model they measured (54.5 NLS). 3B is the Churro base and jumped +27 points after historical SFT — so **7B LoRA on Badger train** is the highest-leverage single-model experiment.

**Qwen3-VL (4B / 8B / larger)** — 2025  
Qwen docs: OCR expanded to 32 languages; “rare and ancient characters”; long-document structure. `Qwen3-VL-8B-Instruct` is the current default generalist to try *instead of* 2.5-7B. Community LoRA `small-models-for-glam/Qwen3-VL-8B-catmus` (medieval Latin) shows the same recipe works: 37.8% → 20.0% CER on CATMuS. That is the wrong script for us, but the method transfers: LoRA on Kade + Dominy + survey pages.

**InternVL 3 / 3.5**  
Strong open VLM family; Churro included InternVL 3.5 30B in the bakeoff (did not lead handwriting). Keep as a second generalist if Qwen plateaus.

**Gemma 3 27B**  
Open weights, used in Gutteridge et al. as the open MLLM. Churro handwritten avg 34.1 — **weak** on historical HTR. Low priority.

**Phi-4-Multimodal (~5B), MiMo-VL, Nemotron Nano VL**  
Small multimodal models. Convenient VRAM; Churro did not find them competitive on historical pages.

**Closed VLMs (GPT / Claude / Gemini)**  
Gemini 2.5 Pro was #2 on Churro-DS overall and #1 among closed models on handwriting. **Allowed for prototyping and silver training data only. Forbidden on test images and in the submitted pipeline.**

### 2.4 Domain-adapted historical VLMs (the category most likely to win among single models)

Pattern: take a mid-size Qwen-VL, supervise on diplomatic historical pages, optionally LoRA per script.

| Model / recipe | Base | Adaptation | Fit for Badger Scribe |
| --- | --- | --- | --- |
| **Churro 3B** | Qwen2.5-VL-3B | Full SFT on 97k historical pages | Strongest published open historical page model; German HW is a highlight. Try zero-shot, then LoRA on our train. |
| **olmOCR 2 7B** | Qwen2.5-VL-7B | 270k PDFs + HW mix | Best documented open fine-tune stack; more modern-doc biased. Fine-tune on our pages. |
| **MEDUSA** (ENC-PSL) | VLM 4B/9B | Medieval multilingual HTR | Wrong period/script; proof that line- or page-VLM LoRA works |
| **Qwen3-VL-8B-catmus** | Qwen3-VL-8B | CATMuS medieval Latin LoRA | Same |
| **Our own LoRA** | Qwen2.5-VL-7B or Qwen3-VL-8B | Human German + careful English | What the official fine-tune starter already claims (~0.23 → ~0.06 CER on a local holdout) |

### 2.5 Strengths and weaknesses for Badger Scribe

**Strengths**

- Reading order, tables, and “what is a line” are implicit — better prior for Dominy and survey pages.
- Catalog metadata (`metadata.csv`) can go in the prompt (names, dates).
- One checkpoint to verify; sequential-load is trivial.
- Fine-tuning on page-level `train.csv` matches the metric (page CER), unlike line CTC.

**Weaknesses**

- VLMs **normalize and invent**. That is lethal on a verbatim CER. Prompts and SFT must say: keep historical spelling, do not expand abbreviations, do not fix grammar.
- Kurrent glyphs (long s, e/n/u confusion, capitals) are underrepresented in general VLM pretraining. Zero-shot Qwen on Kade is the organizers’ ~0.29 CER class, not TrOCR-kurrent’s ~0.03.
- High-resolution pages blow the token budget; need tiling or Qwen’s native dynamic resolution with a max-pixel cap.
- Hallucinated names/dates: metadata helps, but the page wins when it is clear (challenge example: catalog April 6 vs. page March 4).

---

## 3. Community, industry, and production systems

Academic papers understate how much of real archival HTR is built by cooperatives, libraries, companies, and shared model hubs. Those systems are often the ones that have actually seen Kurrent, 19th-century English letters, and faded registers. Many are **closed-weight or API-only**, so they cannot go in the submitted run — they are still useful as: (a) an accuracy ceiling, (b) a source of public models/datasets, (c) a workflow to copy with open parts.

### 3.1 Production HTR platforms (pipeline family)

These are what archives actually buy or self-host. Almost all are **layout → lines → recognizer**, with a GUI for correction and model training.

| System | Who | Open for our submission? | What it is good at | Notes for Badger Scribe |
| --- | --- | --- | --- | --- |
| **Transkribus** (READ-COOP) | Cooperative / company; 500k+ users, 300+ community models | **No** for Titan / Genius / hosted engines. Public **PyLaia** community models are sometimes usable only *inside* Transkribus (weights often not exportable). | The industry default for historical HTR. Text Titan I ter: ~30% lower CER than original Titan; English 11.5%, German 8.6% CER on their 2,000-page held-out historical sample — they report beating ChatGPT / Gemini / Claude on that set. | **German Genius** (4.5% val CER) and **German Kurrent 17th–20th** (PyLaia, ~3M words, 5.4% val CER) are the closest production models to Kade. Use as a *quality target* and, if allowed, to draft extra train pages — never on test. |
| **eScriptorium + Kraken** | EPHE/PSL academic community; instances at Inria CREMMA, UB Mannheim, OpenITI | **Yes.** Apache-2.0 engine + public models | Open Transkribus-class workflow. Trainable layout, reading order, recognition; PageXML/ALTO. | This is the open pipeline we can actually ship. Community models live on **HTRMoPo** / Zenodo (`kraken list` / `kraken get DOI`). |
| **OCR4all + OCR-D + Calamari** | Würzburg / DFG OCR-D (German library mass-digitization) | **Yes** | Semi-automatic workflow for historical **print** (VD16–18). OCR4all beat ABBYY on 19th-c. novels when mixed models existed. | Weak fit for cursive Kurrent / pencil notes. Useful for layout ideas and Calamari as a print/Fraktur engine, not as the HTR core. |
| **Loghi + Laypa** (KNAW HuC) | Dutch national humanities infrastructure | **Yes** | Detectron2 region/baseline segmenter (Laypa) + CNN–LSTM HTR. PageXML tooling, Docker, also merges PyLaia output. | Production-grade open pipeline from an archive shop. Good layout prior for messy pages; recognizer still needs our fine-tune. |
| **Arkindex** (Teklia) | French document-AI company; community edition is OSS | Engine depends on what you plug in | Institution-scale batch HTR: custom workflows, Doc-UFCN heritage, APIs, Docker steps. | Steal the *orchestration* idea (swap YOLO/Kraken/VLM per stage). Do not need the whole platform. |
| **Planet AI IDA** | German company (ex-HTR+ lineage) | **No** | Markets faded scans and **Kurrent / Sütterlin / Gothic** as a product add-on; on-prem or cloud. PerceptionMatrix keeps alternate readings. | Closed. Confirms that industry still treats Kurrent as a specialist module, not a generic VLM job. |
| **ABBYY FineReader / FineReader XIX** | Classic commercial OCR | **No** | XIX is strong on 1600–1937 **print** (Fraktur, Schwabacher). ICR is **hand-print in form fields**, not full-page cursive. | Not a competitor on Kade letters. Occasional use for printed headings on Dominy pages only if we had weights — we do not. |

**Transkribus community models that match this corpus (closed runtime, open lesson):**

- **German Kurrent / Sütterlin / Fraktur, 17th–20th c.** (PyLaia, 2021) — 3M words, 5.4% val CER. Includes Greifswald “Jack of all Trades” 17th–18th data plus later hands.
- **German Genius** (super model, 2025) — Kurrent + Sütterlin + Fraktur + modern German; 4.5% val CER. Broader than script-specific models.
- **Text Titan I / I ter / II** — TrOCR-based general Latin-script historical supermodel. Titan I ter trained on 31M words; beats their language-specific supers on English/German/Dutch in-house. Romein et al. (2025) also found Titan strongest out-of-the-box on Latin-script historical sets.
- **English Elder** — language-specific English supermodel (Titan I ter reports 11.5% vs Elder 12.3% on their bench).

Hodel et al. (2021) already showed the same pattern in public: a **Kurrent-matched** Transkribus model ~6% CER on unseen 19th-c. hands vs generic PyLaia ~19%. Community supermodels are that idea at archive scale.

### 3.2 Cloud document AI (closed APIs ≈ “one model”)

Big-tech OCR is a single API call: image in, text + boxes out. That is approach 2 from the user’s point of view, but **weights are closed**, so these are prep-only (and **never on test images**).

| Service | Handwriting claim | Historical evidence | Use for us |
| --- | --- | --- | --- |
| **Azure Document Intelligence Read** | Print + handwriting; DE/EN among supported HW languages | Kelly (2020, English diaries 1853–1888): Azure ~78% “accuracy” vs Google ~74% vs AWS ~16%. Churro-DS still had VLMs beating Azure-style pipeline OCR on page text. | Best generic cloud HTR for **modern-ish English cursive**. Unlikely to read Kurrent well. Fine for silver drafts of extra English pages we curate. |
| **Google Cloud Vision** | Handwriting | Kelly: better than Azure on the hardest 1853 diary page | Same rules as Azure. |
| **AWS Textract** | Forms + some handwriting | Weak on historical cursive in independent tests | Skip. |
| **Mistral OCR** | Long dense documents | In Churro’s OCR-system comparison; print-oriented | Prep only. |
| **Gemini / GPT / Claude vision** | Strong general document VLMs | Churro: Gemini 2.5 Pro #2 overall, #1 closed on handwriting (63.6 NLS) — still behind fine-tuned Churro 3B | Organizers already used Claude for `silver_claude`. We may distill from them **off test**. |

Industry takeaway: cloud OCR is tuned for **modern business handwriting and PDFs**, not Kurrent. Archives that need Kurrent buy Transkribus or Planet AI, or train Kraken. That is a vote for a specialist (or a historically fine-tuned VLM), not vanilla Azure.

### 3.3 Shared model and data hubs (the actual community)

This is where reusable **open** artifacts live.

**HTRMoPo** (INRIA / kraken.re) — catalog of Kraken/Party models with DOIs. Download with `kraken get`. Relevant entries:

- Revolutionary City English archival HTR (~95k 18th-c. lines)
- Joseph Hooker 19th-c. English correspondence (CER 11–12%)
- Community Kurrent Kraken models (e.g. Alze’s Louise Le Beau diary model — only 12 pages; useful as a fine-tune start, not a finished engine)
- Party multilingual historical base (German manuscripts in the mix)

**HTR-United** — catalog of **ground-truth datasets** (PageXML), not models. Filter by language/script/century. Primary place to find extra diplomatic GT for Kraken/PyLaia/TrOCR training.

**Hugging Face community**

- `dh-unibe/trocr-kurrent` and `trocr-kurrent-XVI-XVII` — university lab, public weights, the best open Kurrent recognizer
- `dh-unibe/kraken-medieval-german-v2` — wrong century, shows the same group ships Kraken weights
- `small-models-for-glam/*` — GLAM (galleries/libraries/archives) LoRAs on Qwen3-VL (e.g. CATMuS medieval Latin)
- Scattered Kraken `.mlmodel` / `.safetensors` (Fraktur print, medieval French, etc.)

**Transkribus model hub** — 300+ community models. Highest Kurrent/English coverage in the world, but many **cannot leave the platform**. Lesson: the winning community strategy is “train a specialist on a lot of similar hands,” not “one huge general model,” except for Titan-class supermodels that are closed.

**Zenodo `ocr_models` community** — older Kraken/Calamari dumps; still how eScriptorium users publish.

### 3.4 Open document-OCR products from companies and labs (single-model family)

These are closer to approach 2: page → Markdown/text. Built for PDFs and books more than Kurrent, but they are what the 2025–26 open-source community is actually shipping.

| Project | Origin | Weights | Role |
| --- | --- | --- | --- |
| **olmOCR 2** | Allen AI | Apache-2.0 7B | Best-documented open page OCR; extra handwritten/typewritten pages; Lincoln letter example. Fine-tune on our train. |
| **PaddleOCR / PaddleOCR-VL** | Baidu | Open | Industry-grade multilingual pipeline + a 0.9B VL parser (OmniDocBench 92.6 vs DeepSeek-OCR 86.5). Strong on **print and tables**; handwriting support exists but is not Kurrent-grade. Possible layout/table front end for Dominy books. |
| **DeepSeek-OCR** | DeepSeek | Open (~3B) | Page → Markdown, token-efficient. Community benches vs Paddle/olmOCR on newspapers. Worth a zero-shot pass; no Kurrent claim. |
| **Surya + Marker** | Datalab (company, open core) | Open | Layout, reading order, tables, multilingual OCR; Marker wraps it for PDF→Markdown. Fast community default for *printed* historical newspapers. Weak evidence on cursive. |
| **MinerU** | OpenDataLab / community | Open | Scientific PDF parsing. olmOCR human eval preferred olmOCR over MinerU. Low priority here. |
| **Nanonets-OCR** | Nanonets (company) | Some open 3B weights | Page OCR VLM; mid-pack on Churro historical handwriting. |
| **Polyscriptor** | UB Mannheim | Open toolkit | GUI to train/compare TrOCR, CRNN-CTC, Qwen3-VL, LightOnOCR, Party, Kraken, PaddleOCR. Not a model — a community **bakeoff harness** we can steal. |
| **FromThePage** | Company + volunteer community | N/A | Crowdsourced correction; can ingest HTR and (via API) Transkribus. Workflow lesson for the writeup’s “scaling to unlabelled pages,” not a recognizer. |

### 3.5 What the community implies for our two approaches

**Pipeline lane.** The living open stack is **eScriptorium/Kraken or Loghi**, not a paper U-Net. Start from a **community Kurrent or English model** (HTRMoPo / trocr-kurrent / Hooker), then fine-tune on our human lines. That is exactly how archives work: supermodel or public base → 20–100 pages of local GT → usable CER. Transkribus German Kurrent / Genius are the closed existence proof that this saturates around **~5% line CER** on diverse Kurrent when you have millions of words.

**Single-model lane.** Industry cloud VLMs/OCR are a poor Kurrent prior. The community systems that *are* open and page-level (olmOCR, PaddleOCR-VL, Marker/Surya, DeepSeek-OCR) are **document parsers**. They will help English account books and maybe survey notes; they will not replace a Kurrent specialist unless we fine-tune (Churro’s result). GLAM LoRAs on Qwen are the community pattern to copy.

**Hybrids the community already runs.** OCR4all and FromThePage assume a human-in-the-loop. Arkindex and Polyscriptor assume you **swap engines per collection**. That supports a routed pipeline: Kraken/TrOCR-kurrent on `kade_letters`, page VLM on English categories — sequential on one GPU.

**Hard rule reminder.** Titan, Genius, IDA, Azure, Gemini, Claude = prep, teacher, or ceiling only. If a community model’s weights are not public (typical Transkribus), it is not a submission ingredient.

---

## Models most likely to do well on *these* older writings

Ranked for **this** corpus (Kurrent letters 1932–42, English account books 1810s–40s, Wisconsin survey notes 1830s–60s), open-weight, single-GPU.

### A. If we commit to one page-level model

1. **Churro (Qwen2.5-VL-3B historical SFT)** — only open VLM with a large, published gain on historical **German handwriting**. First zero-shot baseline.
2. **Qwen2.5-VL-7B or Qwen3-VL-8B + LoRA on Badger train** — official path; 7–8B is the quality/VRAM sweet spot; prompt with category + catalog fields.
3. **olmOCR-2-7B** as the alternative base if we want a document-OCR inductive bias (layout, reading order) before LoRA.
4. **Qwen2.5-VL-72B-AWQ/INT4** as a no-train ceiling on handwriting (Churro: best open zero-shot HW). Use to generate open silver labels or as a teacher, not necessarily as the deploy model.
5. **InternVL 3.5 (quantized)** only if Qwen saturates.

Skip as primary: Gemma 3, Donut-base, generic GOT-OCR2, PaddleOCR — weak or unproven on Kurrent / 19th-c. English archival hands.

### B. If we allow a specialist that is “one recognizer” (usually with lines)

1. **dh-unibe/trocr-kurrent** on Kraken/YOLO lines — best prior for Kade.
2. **Party** multilingual historical checkpoint + language token `deu` / `eng` — one recognizer for both languages after a shared segmenter.
3. **Kraken/PyLaia or Loghi fine-tune** on human Kurrent lines, starting from an HTRMoPo/Hooker/community base; separate English model on Bentham + Hooker-like English + our human English rows.
4. Closed community **ceilings** (not submittable): Transkribus German Genius / German Kurrent / Titan I ter — use only to judge how close an open stack is.

### C. Auxiliary data that actually matches the writings

From the challenge `RESOURCES.md` and the literature:

- **Alfred Escher correspondence** — German 19th-c. letters; organizers point here for Kurrent.
- **dh-unibe Kurrent training collections** (if redistributable) / Transkribus public Kurrent GT
- **Bentham**, **BLN600**, **IAM** — English handwriting; IAM is modern and too clean
- **Joseph Hooker** correspondence (Kraken, CER 11–12%) — late-19th-c. English letters
- **Revolutionary City** Kraken model — 18th-c. English archival hands (~95k lines)
- **Churro-DS** German/English handwritten pages — page-level diplomatic GT
- **NARA** record groups — English administrative handwriting (layout closer to survey notes than IAM)

Do **not** train naively on `silver_claude` English rows; they over-normalize, which this metric punishes.

---

## Head-to-head (what the literature actually says)

| Question | Evidence | Implication |
| --- | --- | --- |
| Are VLMs already better than pipelines on historical pages? | Churro-DS: several VLMs beat Azure-style pipeline OCR on *page text*. Not a Kraken+TrOCR-kurrent bakeoff. | VLMs win on messy layout; specialists still win on a known script with good lines. |
| Does fine-tuning a small VLM beat a huge zero-shot VLM? | Churro 3B (70.1 HW) vs Qwen 72B (54.5 HW) | Domain SFT beats scale when you have historical pages. |
| Does a Kurrent specialist still matter? | TrOCR-kurrent ~2.7% line CER; generic PyLaia ~19% on unseen Kurrent; organizers’ best open page model ~29% page CER on Kade | Yes. A page VLM will likely need Kurrent SFT or a Kurrent expert in a mixture. |
| End-to-end VLM vs OCR+VLM? | Gutteridge et al. 2025: `ocr+pageN` ≥ end-to-end MLLM on multi-page HW | Hybrid is legitimate; corrector must stay open-weight and non-modernizing. |
| One model for all categories? | Macro-average + different scripts | A single VLM *can* cover all three, but a **router** (Kurrent specialist vs English VLM) is the historically winning pattern. |

---

## Practical recommendation for this repo (not a commitment)

**Phase 0 — measure, don’t guess**  
Zero-shot: Churro 3B, Qwen2.5-VL-7B, Qwen3-VL-8B, olmOCR-2-7B, each with a verbatim prompt + catalog metadata. Score with `metric.py` **by category and `label_source`**.

**Phase 1 — pick a lane from the numbers**  
- If Kurrent CER is the hole and lines look clean: pipeline with **trocr-kurrent** (or Party) + YOLO/Kraken.  
- If English layout/tables are the hole: stay on a **page VLM** and LoRA it.  
- If both fail different ways: **mixture** — Kurrent line specialist + English page VLM, one GPU, load sequentially. This is still “one submitted pipeline,” not one architecture.

**Phase 2 — fine-tune only on trustworthy text**  
Human German first. English: human rows + public HTR corpora; silver rows only as weak labels or after a second-model filter.

**Invariant**  
Prompts, CTC targets, and post-correctors must follow `transcription_conventions.md`. The literature’s biggest VLM failure mode (helpfully modernizing) is exactly what this leaderboard punishes.

---

## Sources

### Challenge and platforms

- Endemann, Prater, and Chovanec. *Badger Scribe*. Kaggle, 2026. https://www.kaggle.com/competitions/badger-scribe
- Muehlberger et al. Transkribus. *Journal of Documentation*, 2019 (platform context in Romein et al., 2025).
- Kiessling, B. et al. eScriptorium / Kraken. https://kraken.re — Kiessling et al., 2019.

### Community, industry, and production systems

- READ-COOP / Transkribus. Product and model hub: https://www.transkribus.org/ — Text Titan I ter blog: https://blog.transkribus.org/en/new-text-titan-i-ter-and-how-it-compares-to-chatgpt-gemini-and-other-llms
- Transkribus **German Kurrent** (PyLaia, ~3M words, 5.4% val CER): https://www.transkribus.org/models/german-kurrent-and-sutterlin-17th-20th-century
- Transkribus **German Genius** supermodel (4.5% val CER): https://www.transkribus.org/models/german-genius-super-model
- University of Greifswald / READ-COOP. “German_Kurrent_17th-18th — The Kurrent Jack of all Trades.” https://blog.transkribus.org/en/german_kurrent_17th-18th-the-kurrent-jack-of-all-trades
- eScriptorium community instances and model pointers: https://escriptorium.eu/community/
- HTRMoPo (Kraken/Party model catalog): https://htrmopo.inria.fr / https://htrmopo.org
- HTR-United (ground-truth catalog): https://htr-united.github.io/catalog.html
- Reul, C., et al. “OCR4all — An open-source tool… for historical printings.” *Applied Sciences* 9(22):4853, 2019. https://www.ocr4all.org/
- OCR-D (DFG). https://ocr-d.de
- KNAW HuC. Loghi / Laypa / loghi-htr. https://github.com/knaw-huc/loghi
- Teklia. Arkindex (community + enterprise). https://www.teklia.com/en/arkindex
- PLANET AI. IDA Recognition (Kurrent/Sütterlin add-on). https://planet-ai.com/ida-en/recognition/
- ABBYY FineReader Engine / FineReader XIX (historical print, field ICR). https://docs.abbyy.com/
- Microsoft Azure Document Intelligence Read (handwriting language list). https://learn.microsoft.com/azure/ai-services/document-intelligence/
- Kelly, J. “Benchmarking computer vision transcription of historical handwritten documents” (Azure vs Google vs AWS, 2020). https://justin.kelly.au/benchmarking-computer-vision-transcription-of-historical-handwritten-documents/
- CCS. HTR engine benchmark (Transkribus vs Calamari vs Tesseract vs Glyph, 18th–19th-c. Ukrainian). https://content-conversion.com/htr-benchmark/
- Datalab. Surya / Marker. https://github.com/VikParuchuri/surya
- PaddlePaddle. PaddleOCR / PaddleOCR-VL. https://github.com/PaddlePaddle/PaddleOCR
- DeepSeek-OCR. https://github.com/deepseek-ai/DeepSeek-OCR
- UB Mannheim. Polyscriptor (multi-engine HTR toolkit). https://github.com/UB-Mannheim/polyscriptor
- FromThePage (crowdsourced correction + HTR import). https://fromthepage.com/
- Alze, J. Kraken Kurrent model (Louise Le Beau diary). HTRMoPo / Zenodo 10.5281/zenodo.21747522
- Canadian Research Knowledge Network. Open OCR evals (Paddle / olmOCR / Chandra / DeepSeek vs ABBYY). https://github.com/crkn-rcdr/OCR-Evaluations

### Pipeline: layout, lines, classical HTR

- Oliveira, S. A., et al. “dhSegment: A generic deep-learning approach for document segmentation.” arXiv:1804.10371, 2018.
- Boillet, M., et al. “Doc-UFCN.” ICDAR-related / Teklia, 2020. arXiv:2012.14163.
- Boillet, M., et al. “Robust Text Line Detection in Historical Documents.” arXiv:2203.12346, 2022.
- Meunier, J.-L., et al. “Historical Text Line Segmentation… Mask-RCNN against U-Net.” *Journal of Imaging* 10(3):65, 2024. https://doi.org/10.3390/jimaging10030065
- Puigcerver, J. PyLaia (CNN–LSTM–CTC). 2017–2024. https://github.com/jpuigcerver/PyLaia
- Sauvola, J., and Pietikäinen, M. “Adaptive document image binarization.” *Pattern Recognition* 33(2), 2000.
- Momtaz, Y., et al. “Modular Pipeline for Text Recognition in Early Printed Books Using Kraken and ByT5.” *Electronics* 14(15):3083, 2025. https://www.mdpi.com/2079-9292/14/15/3083
- Momtaz implementation: https://github.com/yahyamomtaz/medieval-ocr-pipeline
- Moins, T. DocWorkflow (YOLO + Kraken / VLM-HTR). https://github.com/TheoMoins/DocWorkflow
- Schaefer, J., and Litvine, A. Joseph Hooker HTR (Kraken, 19th-c. English). Zenodo 10.5281/zenodo.8038689, 2023.
- Nelson, D. R. Revolutionary City Kraken model (18th-c. English). HTRMoPo / Zenodo 10.5281/zenodo.19238205.

### Kurrent and historical HTR evaluation

- Hodel, T., Schoch, D., Schneider, C., and Vogeler, G. “General Models for Handwritten Text Recognition: Feasibility and State-of-the-Art. German Kurrent as an Example.” *Journal of Open Humanities Data* 7:13, 2021. https://doi.org/10.5334/johd.46
- Romein, C. A., Rabus, A., Leifert, G., and Ströbel, P. “Assessing advanced handwritten text recognition engines for digitizing historical documents.” *International Journal of Digital Humanities*, 2025. https://doi.org/10.1007/s42803-025-00100-0
- Widmer, J., and Hodel, T. `dh-unibe/trocr-kurrent` (19th-c.) and `dh-unibe/trocr-kurrent-XVI-XVII`. https://huggingface.co/dh-unibe/trocr-kurrent

### Specialized single recognizers

- Li, M., et al. “TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models.” AAAI 2023. https://arxiv.org/abs/2109.10282 — IAM: Large 2.89% CER.
- Kim, G., et al. “OCR-free Document Understanding Transformer (Donut).” ECCV 2022.
- Wei, H., et al. “General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model (GOT).” arXiv:2409.01704, 2024.
- Kiessling, B. Party (Swin + Llama historical recognizer). https://github.com/mittagessen/party — base model https://doi.org/10.5281/zenodo.14616980

### Page-level / multimodal VLMs

- Bai, S., et al. “Qwen2.5-VL Technical Report.” arXiv:2502.13923, 2025.
- Qwen3-VL (OCR / ancient-character claims): https://github.com/QwenLM/Qwen3-VL
- Semnani, S. J., Zhang, H., He, X., Tekgürler, M., and Lam, M. S. “Churro: Making History Readable with an Open-Weight Large Vision-Language Model…” arXiv:2509.19768, 2025. https://github.com/stanford-oval/Churro
- Poznanski, J., et al. olmOCR / olmOCR 2 (Allen AI). https://allenai.org/blog/olmocr-2 — `allenai/olmOCR-2-7B-1025`
- Gutteridge, B., et al. “Judge a Book by Its Cover: Investigating Multi-Modal LLMs for Multi-Page Handwritten Document Transcription.” arXiv:2502.20295; COLM 2026. https://github.com/BenGutteridge/judge-a-book-by-its-cover
- OmniHandwritingOCR benchmark (MLLM vs specialized OCR). arXiv:2608.18586, 2026.
- `small-models-for-glam/Qwen3-VL-8B-catmus` — Qwen3-VL LoRA on CATMuS medieval Latin. https://huggingface.co/small-models-for-glam/Qwen3-VL-8B-catmus
- MEDUSA medieval VLMs: ENC-PSL collection on Hugging Face; wired in DocWorkflow.

### Surveys

- AlKendi, W., et al. “Advancements and challenges in handwritten text recognition: A comprehensive survey.” *Journal of Imaging* 10(1), 2024.
