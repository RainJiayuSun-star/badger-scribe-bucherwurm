#!/bin/bash
# usage: startq.sh <user> <queue name> <cutoff "YYYY-MM-DD HH:MM">
ssh -n -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -p 2200 "$1@128.105.218.242" \
  "tmux has-session -t q-$2 2>/dev/null && { echo already running; exit 0; }; tmux new-session -d -s q-$2 \"CUTOFF='$3' bash ~/badger-scribe-bucherwurm/runs/pm/bin/queue.sh ~/badger-scribe-bucherwurm/runs/pm/queue-$2.txt\"; sleep 3; tmux ls; hostname"
