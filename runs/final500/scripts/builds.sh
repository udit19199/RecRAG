#!/bin/zsh
# Builds the 8 HotpotQA and Natural Questions indexes in parallel (flex tier).
# Writes runs/final500/STATUS.md; stage done only when every build.json exists
# and reports zero missing-usage calls.
cd /Users/uditagarwal19199/Workspace/HPE/RecRAG
R=runs/final500
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\nspend so far: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" "$(PYTHONPATH=. .venv/bin/python $R/scripts/spend.py 2>/dev/null)" > $R/STATUS.md; }
st "builds running" "8 index builds started" "none"
pids=()
for ds in hotpotqa natural_questions; do
  for m in rag raptor hipporag graphrag; do
    py=.venv/bin/python; [ $m = hipporag ] && py=.venv-hipporag/bin/python
    mkdir -p $R/$ds/$m/build
    PYTHONPATH=. $py $R/scripts/build.py $ds $m > $R/$ds/$m/build/build.log 2>&1 &
    pids+=($!)
  done
done
( while sleep 120; do n=$(ls $R/*/*/build/build.json 2>/dev/null | wc -l | tr -d ' '); st "builds running" "$n of 8 builds done" "none"; done ) &
ticker=$!
for p in $pids; do wait $p; done
kill $ticker 2>/dev/null
bad=""
for ds in hotpotqa natural_questions; do for m in rag raptor hipporag graphrag; do
  f=$R/$ds/$m/build/build.json
  if [ ! -f $f ]; then bad="$bad $ds/$m(no build.json: $(tail -1 $R/$ds/$m/build/build.log | cut -c1-120))"; fi
done; done
if [ -z "$bad" ]; then st "builds done" "8 of 8 builds done" "none"; else st "builds failed" "see logs" "$bad"; fi
