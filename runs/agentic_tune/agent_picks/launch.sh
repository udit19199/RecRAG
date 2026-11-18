#!/bin/zsh
# Runs the 3 repeats in parallel, then checks each has 38 rows before stage: done.
cd /Users/uditagarwal19199/Workspace/HPE/RecRAG
R=runs/agentic_tune/agent_picks
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $R/STATUS.md; }
st running "3 repeats started" "none"
pids=()
for rep in 1 2 3; do
  TUNE_PROCESSES=3 PYTHONPATH=. uv run python $R/run.py $rep > $R/rep$rep.log 2>&1 &
  pids+=($!)
done
( while sleep 60; do n=$(cat $R/rep*.log 2>/dev/null | grep -c "saved retrieval"); st running "$n of 114 retrievals saved" "none"; done ) &
ticker=$!
for p in $pids; do wait $p; done
kill $ticker 2>/dev/null
bad=""
for rep in 1 2 3; do
  f=$R/rep$rep/results.jsonl
  n=$( [ -f $f ] && wc -l < $f | tr -d ' ' || echo 0)
  [ "$n" = 38 ] || bad="$bad rep$rep=$n"
done
if [ -z "$bad" ]; then st done "114 of 114 (3 x 38 rows checked)" "none"; else st failed "row check" "wrong row counts:$bad"; fi
