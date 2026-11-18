#!/bin/zsh
# Wait for the VM queue (runs/final500/jobs/vm_queue/STATUS.md on omarchy-mpv0).
# Polls every 5 minutes (which also keeps the VM from idle-pausing). Exit 0 on
# "stage: done", 1 on "stage: failed". If exec is refused (cooldown after the
# free plan's 8-hour stop) it waits and retries. If the VM came back with no
# queue process while the stage says running, it restarts the queue (each step
# skips finished work). Exits 2 after MAX_SECS (default 6800) so the caller can
# re-arm it inside its own time limit.
max=${MAX_SECS:-6800}
start=$(date +%s)
R=/data/workspace/RecRAG
while :; do
  # A free-plan VM is suspended at its run limit; resume it (state is kept).
  if instabox list 2>/dev/null | grep -A3 'omarchy-mpv0' | grep -q '"state": "suspended"\|omarchy-mpv0 *Suspended'; then
    echo "$(date -u +%T) VM suspended; resuming"
    instabox resume omarchy-mpv0 2>&1 | tail -1
  fi
  out=$(instabox exec omarchy-mpv0 -- bash -lc "cat $R/runs/final500/jobs/vm_queue/STATUS.md; echo QUEUE=\$(pgrep -fc '^bash runs/final500/scripts/vm_queue.sh'); tail -1 $R/runs/final500/jobs/vm_mem.log" 2>&1)
  if echo "$out" | grep -q '^stage: done'; then echo "$out"; exit 0; fi
  if echo "$out" | grep -q '^stage: failed'; then echo "$out"; exit 1; fi
  if echo "$out" | grep -q '^stage: running' && echo "$out" | grep -q '^QUEUE=0'; then
    echo "$(date -u +%T) queue not running; restarting it"
    instabox exec omarchy-mpv0 -- bash -lc "cd $R && setsid nohup bash runs/final500/scripts/vm_queue.sh >> runs/final500/jobs/vm_queue.log 2>&1 < /dev/null &" >/dev/null 2>&1
  fi
  echo "$(date -u +%T) $(echo "$out" | grep -E '^progress|mem_used|cooldown|409' | tr '\n' ' ' | cut -c1-200)"
  [ $(( $(date +%s) - start )) -gt $max ] && { echo "re-arm"; exit 2; }
  sleep 300
done
