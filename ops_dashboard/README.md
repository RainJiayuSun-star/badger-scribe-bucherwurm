# Operations dashboard

A local dashboard for your own EFabric GPU workspaces and fine-tuning agents. Each
developer runs their own copy against their own workspaces. It reads run settings and
scores over SSH. It never reads transcription text, predictions or images, and it never
starts or changes a job.

## Set up

1. Copy the two example files and fill them in:
   ```sh
   cp ops_dashboard/config.example.json ops_dashboard/config.json
   cp ops_dashboard/history_notes.example.json ops_dashboard/history_notes.json
   ```
   In `config.json`, set `ssh_host` and each workspace's `user` from your EFabric SSH
   line. A workspace on a different host can set its own `host`. `remote_runs` is the
   runs folder on EFabric, and `local_runs` is the runs folder in this checkout. Under
   `roles`, map each of your agents' worker names to `coordinator`, `code`, `analysis`
   or `review`, and give them display names under `names`.
2. Make sure `ssh -p 2200 <user>@<host>` works without a password prompt.
3. Optional: put a Kaggle API token in `~/.kaggle/kaggle.json` (Kaggle, Settings, API,
   Create New Token) so the dashboard reads your uploads and public scores itself.
   `kaggle_competition` in `config.json` sets the competition (default `badger-scribe`).
4. Optional: an `estimates.json` in your EFabric `runs/pm/` folder (`{"run id": minutes}`)
   gives the dashboard a planned duration for jobs that haven't started yet.

Both files, and everything the dashboard generates, are ignored by git.

## Run

From the repository root:

```sh
python3 ops_dashboard/server.py        # then open http://127.0.0.1:8766
python3 ops_dashboard/history.py       # builds and opens the fine-tuning log
```

The server listens only on your machine and checks each workspace every 2 minutes. If a
workspace can't be reached, it keeps the last data and marks it stale. Every 10 minutes
it also refreshes the run history (for the Training tab) and, with a token, your Kaggle
uploads. `history.py` caches what it reads in `history_runs.json`, so the log still works
after the workspaces stop. Use `--offline` to skip EFabric and `--no-open` to skip the browser.

## What it shows

- Kaggle submission: whether a finished `submission.csv` hasn't been uploaded yet (the box
  flashes until it is), the last upload's public CER compared with the one before, and when
  the next queued submission should be ready. With a Kaggle token this updates itself;
  without one, add the run ID to `submitted` in `history_notes.json` after uploading, and
  its score to `public_scores` once Kaggle shows it.
- Latest submission by collection: the model and adapter used for each collection, with
  a held-out CER estimated from fold runs trained the same way.
- To do: your agents' tasks that are waiting or blocked.
- GPU workspaces: the running job, its epoch and progress, how long the job, the epoch
  and the workspace have been running, time left before the 24-hour stop, GPU
  utilization, memory and queued jobs.
- Agents: your coordinator and workers, by role. Agents report through `telemetry.py`;
  give every agent `WORKER_REPORTING.md`. A task counts as active only if it was
  reported in the last 5 minutes.
- Runs: the jobs in your current queues.
- Training: training-loss curves of the latest submission's adapters, public CER by
  submission, and for any fine-tuning run its loss and held-out CER by epoch.

## Tests

```sh
python3 ops_dashboard/test_dashboard.py
python3 ops_dashboard/test_history.py
node ops_dashboard/test_app.cjs
```
