#!/bin/bash
# B1 on the Instabox VM (runs/selection/SPEC.md): repeats 1-3, each as 2 shards of
# 125 questions, one at a time. A shard whose RESULTS line is already in its log
# is skipped, so the script can be rerun after a pause. Writes jobs/STATUS_B1.md.
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=. HF_HOME=$HOME/.cache/huggingface
J=runs/selection/jobs; mkdir -p $J
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $J/STATUS_B1.md; }
for rep in 1 2 3; do
  for s in 0 1; do
    log=$J/B1_r${rep}_s${s}.log
    grep -q '^RESULTS ' $log 2>/dev/null && continue
    st running "B1 repeat $rep shard $s of 2" none
    .venv/bin/python runs/selection/scripts/run.py B1 $rep $s 2 > $log 2>&1 \
      || { st failed "B1 repeat $rep shard $s" "$(tail -n 1 $log | cut -c1-200)"; exit 1; }
  done
done
st done "B1: 3 repeats x 2 shards" none
