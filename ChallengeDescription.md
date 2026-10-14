# Badger Scribe — Challenge Brief

Source: [Kaggle competition](https://www.kaggle.com/competitions/badger-scribe) (ML+X / UW–Madison MLM26). Educational challenge; no cash prizes. Recognition is a verified, documented pipeline the UW Libraries could actually run.

Citation: Christopher Endemann, Scott Prater, and Kevin Chovanec. *Badger Scribe*. https://www.kaggle.com/competitions/badger-scribe, 2026.

---

## What the challenge is

UW Digital Collections has thousands of scanned archival pages and almost no transcriptions. Without text, the pages cannot be searched, read by a screen reader, or studied at scale.

**Build a transcription pipeline that turns a page image into verbatim text.** The pipeline must be something a library can deploy on its own hardware: open-weight models only, one GPU, at most 96 GB VRAM at inference.

There is no required architecture. Viable approaches include:

- Traditional OCR/HTR: layout detection → line segmentation → recognition (Kraken, PyLaia, etc.)
- Page-level vision-language models
- Fine-tuning a small VLM or HTR model on the released train set and/or public auxiliary datasets
- Any combination of the above

How the pipeline is built matters as much as which model you pick.

---

## Goal

Given a single page image, produce a **faithful transcription**:

- Full page text, in natural reading order
- Original spelling, casing, and punctuation
- No summarization, paraphrasing, or “cleanup” of historical spelling

Inputs include handwritten manuscripts and degraded historical scans. The evaluation set is built to be hard: variable hands, faded ink, irregular layouts, dense account books, and German Kurrent (a different script from English cursive).

---

## Hard requirements

These are pass/fail. A high leaderboard score without them does not count.

### 1. Open-weight models only in the submitted run

Every model in the submitted pipeline — recognizer, layout detector, language router, post-corrector — must be open-weight. Closed APIs (GPT, Claude, Gemini) are forbidden **anywhere** in the submitted run, including “just for post-correction.”

Closed models **are** allowed for preparation: prototyping, generating silver training labels, distillation. Document that use in the writeup.

**Hard line:** never run a closed model on the **test** images. Pseudo-labeling the test set and training on it disqualifies.

### 2. Single GPU, ≤ 96 GB VRAM at inference

The limit binds the **final inference run** that produces predictions, not training.

- Sequential load/unload of models is fine
- Quantization is allowed and encouraged
- Training, fine-tuning, and search can use clusters, cloud, multi-GPU — unconstrained

The 96 GB figure matches campus RTX Pro 6000-class hardware. Starter notebooks also fit Kaggle’s free GPU tier.

### 3. Predictions must come from the pipeline

No human transcription or manual correction of test pages. Humans may build, tune, and debug the pipeline. Verification re-runs the posted code and checks that it reproduces the submitted CSV (normal LLM sampling variation is expected).

### 4. Public, reproducible repo

- Repo must be **public**
- License must be **MIT or Apache 2.0**
- Pin a git tag/commit that produced the selected leaderboard submission
- Weights must be public, or reproducible from a public base plus a published adapter
- `WRITEUP.md` must live in that same tagged commit

### 5. Writeup is required

A score with no writeup is not a complete submission. See [Writeup](#writeup) below.

### 6. Team and submission limits

- Max team size: 5
- Team mergers allowed if the merged team stays within size and submission limits
- **2 prediction submissions per team per day**
- Select up to 2 submissions as final scored entries
- One writeup per team (revise by moving the submission tag; edit the Writeup Index comment in place)

---

## Data

Everything the challenge scores is already in the Kaggle dataset. Do **not** crawl UWDC or Winterthur catalogs. If more UW material is needed for training, ask in Discussion and organizers will arrange an export.

### Files

| File | Role |
| --- | --- |
| `train.csv` | `page_id`, `doc_id`, `text`, `category`, `label_source` — labels to learn from |
| `test.csv` | `page_id`, `doc_id`, `category`, `label_source` — pages to predict (no `text`) |
| `sample_submission.csv` | `page_id`, `text` — valid format, empty predictions (scores 1.0) |
| `images/<page_id>.jpg` | one image per page, train and test |
| `metadata.csv` | catalog context per document, join on `doc_id` |
| `transcription_conventions.md` | **read first** — defines what “faithful” means, character by character |
| `metric.py` | exact leaderboard scorer; also runnable locally |
| `RESOURCES.md` | labelled auxiliary datasets mapped to these collections |
| `WRITEUP_TEMPLATE.md` | copy into the repo as `WRITEUP.md` |
| `CONTRIBUTION_TEMPLATE.md` | report a bad label or contribute pages |
| `contributing_transcriptions.md` | how to add pages to the shared train set |

### Collections

| Collection | What it is | Why it is hard |
| --- | --- | --- |
| Max Kade Institute German letters | Immigrant correspondence, 1932–1942 | Kurrent script — a different alphabet from modern German cursive |
| Dominy Craftsmen account books | Woodworkers’ / clockmakers’ books, 1810s–1840s | Dense entries, names, currency |
| Wisconsin Land Survey Field Notes | Surveyors’ notebooks, 1830s–1860s | Terse abbreviations, pencil on weathered paper |

Documents are assigned **whole** to train or test so pages of one letter cannot leak. Exception: the survey notes are one bound volume, so they are split **page by page** (otherwise the category would be unscored).

At launch, German letters are **141 of 182 test pages**. The metric macro-averages categories, so failing Kurrent costs about half the score no matter how well English is read.

### Label sources

| `label_source` | Meaning |
| --- | --- |
| `human` | A person stands behind the text (expert transcription, or a machine draft someone corrected) |
| `silver_claude` | Unreviewed machine draft |

All 436 German pages have expert transcriptions. English collections have almost none, so they ship as drafts and convert to `human` as review completes.

Drafts are useful but not gold. On German pages where both exist, drafts run about **0.061 CER** (median 0.039) vs. a best-open-model ~0.288. On English pages that can be checked, drafts measure ~0.078 CER vs. best-open ~0.198.

**Do not treat silver labels as gold.** Their errors are systematic. The drafting model over-normalizes historical spelling. Training on them naively teaches the pipeline to modernize text that this metric scores verbatim.

### Metadata

`metadata.csv` has catalog fields (`title`, `creator`, `date`, `summary`, collection, rights, item URL). Free to use as model context — names, dates, and subject nouns are what OCR most often gets wrong.

The **page image is authoritative**. Transcribe what is written, even when it differs from the catalog. Example in the train set: `dominy_020_p002` is catalogued April 6, 1806; the page plainly reads March 4.

### The data will change

Ground truth arrives during the competition:

- Pages may be added to train, test, or both
- `silver_claude` test labels may become `human` (text can change)
- Every change is announced in Discussion
- All submissions are **rescored automatically** — no need to resubmit

Build for generalization across hands and languages. A standing that depends on the launch-day (mostly German) distribution will not survive the first batch of verified English.

---

## Scoring

Primary metric: **character error rate (CER), macro-averaged across categories. Lower is better.**

```
page CER      = levenshtein(prediction, ground_truth) / len(ground_truth)   (capped at 1.0)
category CER  = mean of page CERs within the category
score         = mean of category CERs
```

- Empty or missing prediction → CER 1.0
- Unrelated dump of text is also capped at 1.0 — nothing to gain by padding
- Casing, punctuation, and historical spelling **all count**
- Only normalization: whitespace runs (spaces, newlines) collapse to a single space on both sides
- `#` in the **reference** matches whatever the prediction has there (illegible, not scored). `#` in the **prediction** is scored like any other character

Word Error Rate and per-category CER are printed as diagnostics; they are not the ranking metric.

Local scoring (same file as the leaderboard):

```bash
python metric.py --solution train.csv --submission my_train_predictions.csv
```

### Predictions CSV

| column | meaning |
| --- | --- |
| `page_id` | must match `test.csv` exactly; every page present |
| `text` | predicted transcription |

Line breaks may be real newlines (quoted CSV field) or the two-character escape `\n`. Safest path: fill `sample_submission.csv`’s `text` column.

Public leaderboard = fixed subset of test pages. Final standings = the rest, revealed at close.

---

## Writeup

A complete submission is **predictions + writeup**. Both must exist at the deadline.

1. Copy `WRITEUP_TEMPLATE.md` → repo `WRITEUP.md`
2. Include it in the same git tag that produced the score
3. Post one comment in the pinned **Writeup Index** Discussion thread: the submission card plus repo link
4. Edit that comment in place as the pipeline improves

Aim for ≤ 2,500 words. Explain the learning journey: what was tried, what worked, what did not.

Open with this card (values are examples from the official template):

```
code_url: https://github.com/team/pipeline/tree/v1.0-submission
models: Qwen/Qwen2.5-VL-7B-Instruct
peak_vram: 18 GB
leaderboard_cer: 0.183
external_data: released train set (LoRA); Bentham line pairs; 120 self-transcribed survey lines
hardware: RTX 4090 24 GB
eval_wall_clock: 38 min
```

- `code_url` must open in a **private browser window** (if it 404s, the repo is private or the commit is not pushed)
- `hardware` and `eval_wall_clock` are informational only — deployability signal for the Libraries, not scored
- Template structure is a suggestion **except** two required sections: the **submission card** and **Scaling to unlabelled pages**
- Checked pass/fail, not graded on prose. An honest “we did not get to this” satisfies a required section; silence does not

---

## Verification

Organizers clone the tagged repo and re-run the pipeline. They check that it:

- Reproduces the submitted predictions (normal sampling variation is fine)
- Fits the 96 GB single-GPU budget
- Uses only open-weight models
- Did not hand-transcribe test pages

There is no separate withheld exam at the end. The test set grows and is rescored throughout, so generalization is measured continuously.

---

## Things to be aware of

### Transcription conventions come first

Read `transcription_conventions.md` before building anything. The train set is the worked example of what “faithful” means. Scoring is verbatim: do not lowercase, strip punctuation, or modernize spelling.

### Verbatim, but not naive

Where the writing is clear, the page always wins — even against catalog dates. Where a mark genuinely supports two readings, external evidence can decide (weekday vs. date, the same surname in other documents, public records). Example: `dominy_002_p002` could be “March 3d” or “March 30”; the printed line says Tuesday, and March 3, 1846 was a Tuesday. When nothing settles it, the character is `#` and is not scored.

Catalog metadata is worth feeding the model. It resolves ambiguity; it does not overrule a clear reading.

### German Kurrent is not optional

If the team wants a competitive score, Kurrent has to work. It is a different script, stayed in everyday use into the 1940s, and dominates the launch test set. See the Alfred Escher German correspondence dataset in `RESOURCES.md`.

### Silver labels can teach the wrong habit

`silver_claude` drafts over-normalize historical spelling. Fine-tuning on them without filtering or treating them as noisy can make CER **worse** against verbatim ground truth. Prefer `human` rows when they exist. Use silver rows carefully (e.g. as weak labels, or after a human pass).

### Do not game `label_source`

`test.csv` publishes whether each test page is `human` or `silver_claude`. Using that to behave differently — imitating the drafting model’s quirks — is gaming the metric and is what verification looks for. Fix a bad label in Discussion; do not learn to reproduce it.

### Do not harvest UWDC / Winterthur

No crawlers, no bulk downloaders, no automated harvesting of the public catalogs. That can get the whole challenge blocked at the network level. Ask in Discussion for a proper export.

### External data is allowed — document it

Public HTR corpora in `RESOURCES.md` (Bentham, BLN600, NARA, IAM, Alfred Escher, etc.) are fair game if lawfully licensed. Self-transcribed publicly browsable UW pages are also allowed. Disclose everything in `external_data` on the writeup card.

### Contributing transcriptions helps everyone equally

Contributed pages join **train only**, never test (the contributor knows the text). Drafting with a closed model and correcting the output is fine for donated transcription. Use `contributing_transcriptions.md` and `CONTRIBUTION_TEMPLATE.md`. To challenge a silver test label, post the page id and reading in the Label corrections thread.

### Confidence on unlabelled pages is an open problem

There is no page-by-page way to know which machine drafts a librarian could accept unread. One partial idea: two differently trained models agreeing on a page is weak evidence the page is right. Organizers have not validated this well. A better method belongs in the writeup and is worth more to the Libraries than a leaderboard place. That is why **Scaling to unlabelled pages** is a required writeup section.

### Leaderboard is a guidepost, not a verdict

Launch test set is mostly German Kurrent plus English pages scored against provisional machine labels. English volume and label quality will change. Public LB ≠ private LB. Two submissions a day is intentional: this challenge rewards a documented pipeline, not grinding.

### Sharing is expected

Educational, collaborative, no cash prizes. Share repos early, post findings to Discussion, build on other teams’ approaches. Private sharing of competition code **outside the team** is not allowed (standard Kaggle rule). Public sharing on the competition forums/notebooks is encouraged.

### Rights differ by collection

Challenge use is non-profit education / research (fair use). That does not make the images free to republish commercially.

| Collection | Held by | Rights |
| --- | --- | --- |
| Max Kade German letters | Max Kade Institute, UW–Madison | Copyright Undetermined (UND) |
| Dominy Craftsmen papers | Winterthur Library | No Copyright – United States |
| Wisconsin Land Survey Field Notes | WI Board of Commissioners of Public Lands | In Copyright – Educational Use Permitted |

Credit the Max Kade Institute Kurrent Transcription Group when using the German transcriptions. Challenge materials (`metric.py`, starters, conventions) are MIT, © 2026 ML+X.

---

## Organizers

- Chris Endemann (`endemann@wisc.edu`) — Research Cyberinfrastructure / DoIT
- Scott Prater — UW Digital Collections Center
- Kevin Chovanec — Division of Extension

---

## Suggested first steps

1. Read `transcription_conventions.md` and open a few train images against their `train.csv` rows (especially `dominy_020_p002` and `dominy_002_p002`).
2. Run the official metric on a dummy prediction to lock the CSV format.
3. Split local eval by `category` and `label_source` so German / English / human / silver are not mixed into one number.
4. Start from the official starter notebooks (they fit Kaggle’s free GPU). The fine-tuning starter reports a local holdout drop of about 0.23 → 0.06 CER in roughly a GPU-hour — treat that as a published baseline, not a guarantee on the hidden test set.
5. Plan for Kurrent early; do not leave it until after English looks good.
