#!/bin/bash
# Source from a queue script run in tmux on EFabric, from inside a clean clone:
#   source ~/badger-scribe-bucherwurm/runs/pm/bin/queue_lib.sh
#   run_step ft5-english-dominy_group1 $VENV/accelerate launch ...
# Appends to runs/pm/run_commits.log: refuses a dirty tree, records the commit (training runs
# don't record one themselves) and the exit code captured on the very next line.
VENV=$HOME/badger-scribe-bucherwurm/.venv/bin
PMLOG=$HOME/badger-scribe-bucherwurm/runs/pm/run_commits.log
LOGS=$HOME/badger-scribe-bucherwurm/runs/logs
run_step() {
  local id=$1; shift
  local version
  version=$(git describe --always --dirty --abbrev=40)
  case "$version" in
    *-dirty) echo "$(date -u +%FT%TZ) refused-dirty $id $version" >> "$PMLOG"; return 9 ;;
  esac
  echo "$(date -u +%FT%TZ) start $id $version $(hostname)" >> "$PMLOG"
  "$@" >> "$LOGS/$id.log" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) exit=$rc $id" >> "$PMLOG"
  return $rc
}
