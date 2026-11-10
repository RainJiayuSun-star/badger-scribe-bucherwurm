#!/bin/bash
# GPU queue, run in tmux on EFabric:  CUTOFF='2026-11-06 21:31' bash ~/badger-scribe-bucherwurm/runs/pm/bin/queue.sh <queue file>
# Queue lines: <run id> <minutes to finish, or to first checkpoint for training> <clone under ~> <command...>; '#' lines are comments.
# Starts the first line whose id is not in <queue file>.done, so lines can be added or reordered while it runs; polls when empty.
# Rerun after a kill: the item that was running starts again, so every command must be resume-safe (train_item.sh, predict).
# A failed item is recorded with its rc and not retried. An item that can't reach its minutes by CUTOFF minus 15 is skipped.
source ~/badger-scribe-bucherwurm/runs/pm/bin/queue_lib.sh
export VENV
Q=$1; DONE=$Q.done; touch "$DONE"
end=$(( $(date -u -d "${CUTOFF:?set CUTOFF to the workspace stop time, UTC} UTC" +%s) - 900 ))
skipped=" "; idle=0
while (( $(date -u +%s) < end )); do
  id=""
  while read -r i mins clone cmd; do
    case "$i" in ''|\#*) continue ;; esac
    grep -q "^$i " "$DONE" && continue
    case "$skipped" in *" $i "*) continue ;; esac
    if (( $(date -u +%s) + mins * 60 > end )); then
      echo "$(date -u +%FT%TZ) cutoff-guard skipped $i" >> "$PMLOG"; skipped="$skipped$i "; continue
    fi
    id=$i; break
  done < "$Q"
  if [ -z "$id" ]; then
    (( idle )) || echo "$(date -u +%FT%TZ) queue idle $(basename "$Q") $(hostname)" >> "$PMLOG"
    idle=1; sleep 120; continue
  fi
  idle=0
  (cd ~/"$clone" && run_step "$id" bash -c "$cmd")
  rc=$?
  echo "$id rc=$rc $(date -u +%FT%TZ)" >> "$DONE"
done
echo "$(date -u +%FT%TZ) queue stopped at cutoff $(basename "$Q") $(hostname)" >> "$PMLOG"
