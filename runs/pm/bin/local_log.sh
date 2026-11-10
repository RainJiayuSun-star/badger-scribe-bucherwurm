#!/bin/bash
# Queue wrapper: local_log.sh <run id> <command...>
# Runs the command with stdout and stderr on container-local disk, then appends that log to runs/logs/<id>.log.
# Why: two GPU 1 runs (11-05/06) died with "Bad file descriptor" writing their long-open log on the JuiceFS home.
id=$1; shift
d=/tmp/pm-logs; mkdir -p "$d"
"$@" > "$d/$id.log" 2>&1
rc=$?
cat "$d/$id.log" >> "$HOME/badger-scribe-bucherwurm/runs/logs/$id.log"
exit $rc
