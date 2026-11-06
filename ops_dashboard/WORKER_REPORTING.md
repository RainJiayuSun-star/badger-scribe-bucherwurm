# Operations reporting protocol

The coordinator includes this file in every worker assignment. Report at assignment,
start, meaningful progress, blocked and completion, using the same task ID. During
active work, report at least every five minutes or after the next long tool returns.
This is a short local tool call; no separate status-writing agent is needed.

Run it from the repository root:

```sh
python3 ops_dashboard/telemetry.py \
  --task task-id --worker worker-name --title 'Task title' \
  --model churro --stage evaluation --state active --note 'Checking numeric run metadata'
```

States: waiting, active, completed, blocked, cancelled.
Models: operations, churro, olmocr, lighton, weight-space.
Stages: development, validation, fine-tuning, evaluation, release.

Completion means the stated task finished, not that the model is approved. Give
implementation, review, GPU probe and GPU evaluation separate task IDs. Report
numeric results or short summaries only; never labels, prediction text, keys or images.
Report returned work as completed and unexecuted work as waiting. Report failures
as blocked. Do not use an unconditional heartbeat process: it would falsely imply
that an agent is still doing useful work. Late active events become unknown.

The dashboard places agents by worker name: `root` is the coordinator, and every
other worker needs a role (code, analysis or review) under `roles` in your
`config.json`, or it shows under "Role not set". Questions for a developer go in that
developer's state file, not in this task store.

The coordinator records assignment events and any tool-reported agent error. Workers
report their own transitions. This protocol must be passed to new workers: the local
server cannot discover chat agent lifecycle events automatically.
