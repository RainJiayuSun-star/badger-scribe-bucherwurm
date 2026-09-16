# Contributing

We're 3 people, each driving a coding agent, on one repo. That means more
branches, more commits, and bigger diffs than usual — these rules exist to
keep `main` clean and reviewable despite that.

## Branches

- Name: `<github-username>/<short-slug>`, e.g. `ian/ocr-preprocessing`.
  The username makes ownership obvious at a glance; nobody else pushes to
  someone else's branch without asking.
- One branch = one topic. If your agent starts solving a second, unrelated
  problem mid-branch, stop and split it off.
- Never commit or push directly to `main`. `main` only changes via merged PRs.
- Rebase/merge `main` into your branch before opening a PR if it's more than
  a day old, so review happens on a diff against current `main`.

## Pull requests

- Keep PRs small: one logical change, ideally under ~400 changed lines
  (excluding generated files/lockfiles). If an agent produces a huge diff,
  split it into a stack of smaller PRs before requesting review.
- PR description should say *what* changed and *why* in a couple sentences —
  not a restatement of the diff. Link the issue/task it addresses if one
  exists.
- Squash-merge into `main`. That keeps `main`'s history to one commit per
  reviewed change, regardless of how many intermediate commits the agent
  made on the branch.

## Review

- Every PR needs approval from one of the other two teammates — no
  self-approving your own agent's work, since agent output needs a second
  set of eyes just like anyone's code.
- Reviewer reads the diff like they'd review a human's PR: does the change
  make sense, is it scoped to what it claims, does it touch anything it
  shouldn't. Don't rubber-stamp because "an agent wrote it."
- If a PR sprawls beyond what's reasonable to review, ask the author to
  split it rather than approving it wholesale.

## What an agent may touch without asking first

**Freely, on its own branch:**
- Files inside the module/directory the task is scoped to.
- Tests and docs for the code it's changing.

**Ask the human first (in the PR description or before starting) before:**
- Editing shared config: `requirements.txt`/`pyproject.toml`, CI workflows,
  `.github/`, environment/setup files.
- Renaming or deleting files outside the current task's scope.
- Changing a shared data schema, shared utility module, or anything another
  teammate's branch is likely also touching.
- Force-pushing, rewriting history, or touching branch protection.
- Anything that isn't reversible by just closing the PR (e.g. calling
  external services, modifying data files in place).

When in doubt, the agent should ask its human before doing it, not after.

## Avoiding collisions between agents

- Before starting a task, post a one-line "claiming X" note in the team
  channel or as a comment on the relevant issue, naming the files/module
  you're about to touch.
- Prefer directory-based ownership: each person's active work lives in a
  distinct part of the tree (e.g. `src/preprocessing/` vs `src/model/`) so
  two agents rarely need to touch the same file in the same week.
- Keep branches short-lived and PRs small — the less time a branch exists,
  the less chance someone else's agent edits the same file underneath it.
- If you discover mid-task that another open PR touches the same file,
  pause, merge/rebase after theirs lands, then continue instead of racing it.

## Commit messages

Format: `type: short summary`, imperative mood, under ~70 characters on the
first line. Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.

```
feat: add sliding-window tokenizer for scribe input
```

- Add a short body only when the *why* isn't obvious from the diff (a
  workaround, a non-obvious constraint, a tradeoff).
- Don't leave in agent-generated noise ("Update file", "final fix", "wip") —
  squash or reword before pushing if the agent produced a pile of those.
- No need to credit the agent in the message; the PR is reviewed by a human
  either way.

## Repo layout

Put new files where the rest of the project will expect to find them:

```
src/                  # importable code, organized by component/module
  <component>/         # e.g. preprocessing/, model/, eval/
tests/                # mirrors src/ layout, one test module per source module
notebooks/            # exploration only — nothing here is imported by src/
scripts/              # one-off / CLI entry points (training runs, data prep)
data/                 # datasets; large or generated data stays out of git
docs/                 # design notes, write-ups
.github/workflows/    # CI
```

Rules of thumb:
- If it's exploratory/throwaway, it goes in `notebooks/`, not `src/`.
- If other code imports it, it belongs in `src/`, not a notebook or script.
- A test always lives at the path in `tests/` mirroring its source file.
- Don't add a new top-level directory without a quick note in the PR
  explaining why the existing ones don't fit.
