#!/bin/zsh
cd /Users/uditagarwal19199/Workspace/HPE/RecRAG
J=runs/final500/jobs/closed_book
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $J/STATUS.md; }
st running "3 repeats started" none
pids=(); for rep in 1 2 3; do PYTHONPATH=. .venv/bin/python runs/final500/scripts/closed_book.py $rep > $J/r$rep.log 2>&1 & pids+=($!); done
for p in $pids; do wait $p; done
bad=""; for ds in multihop_rag hotpotqa natural_questions; do for rep in 1 2 3; do f=runs/final500/closed_book/$ds/r$rep/results.jsonl; n=$(wc -l < $f 2>/dev/null | tr -d ' '); [ "$n" = 500 ] || bad="$bad $ds/r$rep=$n"; done; done
[ -z "$bad" ] && st done "9 of 9 (500 rows each, checked)" none || st failed "row check" "$bad"
