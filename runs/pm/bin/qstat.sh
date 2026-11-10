#!/bin/bash
# Prints the queue state on both GPUs, briefly.
for u in igloege_ian-badgerscribe_0:gpu1 igloege_ian-badgerscribe2_0:gpu2; do
ssh -n -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -p 2200 "${u%%:*}@128.105.218.242" "q=${u##*:}"'; cd ~/badger-scribe-bucherwurm/runs; echo "== $q $(date -u +%H:%M) $(tmux ls 2>&1 | cut -c1-30) gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"; cat pm/queue-$q.txt.done 2>/dev/null; grep -v "^#" pm/queue-$q.txt | while read i r; do grep -q "^$i " pm/queue-$q.txt.done || { s=$(grep -o "\"phase\": \"[a-z_]*\"" finetune/$i/status.json 2>/dev/null); e=$(tail -1 finetune/$i/events.jsonl 2>/dev/null | grep -o "\"epoch\": [0-9]*, \"step\": [0-9]*"); echo "pending $i $s $e"; }; done; df -h /home/igloege | tail -1 | awk "{print \"free \" \$4}"'
done
