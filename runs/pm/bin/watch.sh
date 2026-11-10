#!/bin/bash
# Mac-side watcher for a Monitor: prints one line per newly finished queue item, per missing tmux session, per ssh failure.
# Polls every 5 minutes; exits by itself when killed by the Monitor timeout.
S=/private/tmp/claude-501/-Users-iangloege-Desktop-badger-scribe-bucherwurm/756ef13b-0db7-4576-b66a-e4cb8baaf1e6/scratchpad
O="-n -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -p 2200"
seen=$S/seen.txt; touch $seen
while :; do
  for u in igloege_ian-badgerscribe_0:gpu1 igloege_ian-badgerscribe2_0:gpu2; do
    out=$(ssh $O "${u%%:*}@128.105.218.242" "tmux has-session -t q-${u##*:} 2>/dev/null || echo NO-SESSION; cat ~/badger-scribe-bucherwurm/runs/pm/queue-${u##*:}.txt.done" 2>&1) || { echo "ssh-fail ${u##*:} $(echo "$out" | tail -1 | cut -c1-80)"; continue; }
    echo "$out" | grep -q NO-SESSION && echo "NO-SESSION q-${u##*:}"
    echo "$out" | grep ' rc=' | while read -r l; do grep -qxF "$l" $seen || { echo "${u##*:} $l"; echo "$l" >> $seen; }; done
  done
  sleep 300
done
