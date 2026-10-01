# Archive Processing Pipeline

The first working pipeline calls UW–Madison's shared, OpenAI-compatible
`churro-3b` endpoint for one faithful transcription per page. It is designed
for quick, resumable benchmarks and presentation demos while we build a
locally-runnable Churro backend for the final competition submission.

## Structure

`transcribe_gateway.py` is the stable command-line entry point. Its smaller
modules separate dataset access, output checkpointing, versioned prompts, the
remote gateway, and the page-by-page runner. A future local Churro backend
only needs to implement the same `transcribe(image)` contract as the gateway.

The default `verbatim-v1` system prompt is intentionally short and is logged
by name and SHA-256 for every request. A system prompt is optional to the API,
but keeping a versioned instruction makes transcription behavior reproducible.
Use `--prompt-file path/to/prompt.txt` only for a deliberate prompt experiment.

## Run it

1. Connect GlobalProtect and load the gateway key into the current terminal.
   The existing `../../badgerchat/env.sh` helper reads it from 1Password; it
   does not store a key in this repository. Do not put the key in a `.env`,
   notebook, or command history.
2. From the project root, make a one-page smoke request:

   ```bash
   source ../badgerchat/env.sh
   python3 pipeline/transcribe_gateway.py \
     --input-csv badger-scribe-data/test.csv \
     --image-dir badger-scribe-data/images \
     --output-csv runs/gateway-churro/smoke.csv \
     --limit 1
   ```

3. Remove `--limit 1` to process the full input. The output is immediately
   rewritten after each successful request, so rerunning resumes incomplete
   work. `runs/gateway-churro/smoke.requests.jsonl` keeps timing and errors.

To process one known page rather than the first page, use its ID:

```bash
python3 pipeline/transcribe_gateway.py \
  --input-csv badger-scribe-data/test.csv \
  --image-dir badger-scribe-data/images \
  --output-csv runs/gateway-churro/one-page.csv \
  --page-id kade_007_p005
```

Repeat `--page-id` to run a selected set of pages. `--page-id` takes
precedence over `--limit`.

### Resume a stopped run

Each successful page is immediately checkpointed to the output CSV. If a
request fails or the terminal closes, rerun the same command with the same
output path and add `--continue`:

```bash
python3 pipeline/transcribe_gateway.py \
  --input-csv splits/smoke_solution.csv \
  --image-dir badger-scribe-data/images \
  --output-csv runs/gateway-churro-smoke/preds.csv \
  --continue
```

Only page IDs absent from `preds.csv` are sent again. `--continue` is explicit
documentation of the default resume behavior; use `--no-resume` only when you
intend to disregard existing predictions and run every page again.

For a scored validation run, substitute `splits/holdout_solution.csv` as the
input and write to a separate run directory, then use the project evaluator.
The input CSV only needs a `page_id` column.

## Tests

These tests require no GPU, API access, or credentials:

```bash
python3 -m unittest discover -s pipeline/tests -v
```

## What this proves — and what it does not

This is an operational baseline: page image → Churro → Kaggle-format CSV,
with fixed decoding, retries, logs, and per-page resume. The service is backed
by an open-weight model, but an external hosted endpoint cannot be the final
competition dependency: organizers must be able to reproduce the submission
on their own hardware. Keep the request/response contract stable when adding
a local Churro backend.
