#!/bin/bash
# Mac side, no compute: every 5 minutes wake the Instabox VM if it paused itself
# (free plan: 120-minute runs), and exit when runs/milvus/jobs/STATUS.md on the VM
# says done or failed. Prints the final STATUS.md.
IBX=$HOME/.local/bin/instabox; VM=omarchy-mpv0
while :; do
  state=$($IBX get $VM --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("state",""))' 2>/dev/null)
  if [ "$state" != "running" ]; then
    $IBX wake $VM >/dev/null 2>&1 || $IBX resume $VM >/dev/null 2>&1
  else
    status=$($IBX exec $VM --timeout 30 -- 'cat ~/work/RecRAG/runs/milvus/jobs/STATUS.md' 2>/dev/null)
    if echo "$status" | grep -qE '^stage: (done|failed)'; then echo "$status"; exit 0; fi
  fi
  sleep 300
done
