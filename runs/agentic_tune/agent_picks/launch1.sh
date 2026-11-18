#!/bin/zsh
cd /Users/uditagarwal19199/Workspace/HPE/RecRAG
R=runs/agentic_tune/agent_picks
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $R/STATUS.md; }
st running "rep1 rerun (rep2, rep3 done)" "none"
TUNE_PROCESSES=1 PYTHONPATH=. uv run python $R/run.py 1 > $R/rep1.log 2>&1 &
pid=$!
( while sleep 60; do st running "rep1: $(grep -c 'saved retrieval' $R/rep1.log) of 38" "none"; done ) & t=$!
wait $pid; kill $t
bad=""; for rep in 1 2 3; do n=$(wc -l < $R/rep$rep/results.jsonl 2>/dev/null | tr -d ' '); [ "$n" = 38 ] || bad="$bad rep$rep=$n"; done
[ -z "$bad" ] && st done "114 of 114 (3 x 38 rows checked)" "none" || st failed "row check" "wrong row counts:$bad"
