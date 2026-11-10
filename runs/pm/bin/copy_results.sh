#!/bin/bash
# Copies training-page results to the Mac: scores, predictions, metrics, config, status, events, analysis.
# Never adapters or checkpoints, and never a test run (submission-* is not in the list).
A=igloege_ian-badgerscribe_0@128.105.218.242
OPTS="-n -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -p 2200"
DEST=/Users/iangloege/Desktop/badger-scribe-bucherwurm/runs/efabric
mkdir -p "$DEST"
ssh $OPTS "$A" 'cd ~/badger-scribe-bucherwurm/runs/finetune && for r in ft5-* ft8-* ft9-* ft10-* ft11-* dec-* screen-*; do [ -d "$r" ] && for f in development zero_shot metrics.csv config.json status.json events.jsonl predictions.csv pages.jsonl summary.json; do [ -e "$r/$f" ] && echo "$r/$f"; done; done | tar cf - -T -' | tar xf - -C "$DEST"
rc=$?
ssh $OPTS "$A" 'cd ~/badger-scribe-bucherwurm/runs && tar cf - analysis/step2 analysis/step3 analysis/step5_8 analysis/step8 analysis/step5_retry analysis/step6 analysis/step9 pm/run_commits.log pm/configs pm/queue-gpu1.txt pm/queue-gpu2.txt pm/queue-gpu1.txt.done pm/queue-gpu2.txt.done 2>/dev/null' | tar xf - -C "$DEST"
echo "$(date -u +%FT%TZ) copy rc=$rc $(ls "$DEST" | wc -l) dirs" >> /Users/iangloege/Desktop/badger-scribe-bucherwurm/runs/pm/copy.log
