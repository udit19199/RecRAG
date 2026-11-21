#!/bin/bash
# Hand-off replay answers (runs/handoff/SPEC.md): 3 variants x 3 repeats, answer
# step only, 3 at a time (light: HTTP calls only). "stage: done" only after every
# job wrote 250 answers for the 250 questions.
cd "$(dirname "$0")/../../.." || exit 1
R=runs/handoff; J=$R/jobs
export PYTHONPATH=.
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $J/STATUS.md; }
st running "starting 9 answer jobs" none
for v in A2_graph_top10 A0_vector_top10 A3_picks_no_graph_rows; do
  for rep in 1 2 3; do
    while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 20; done
    .venv/bin/python $R/scripts/replay.py answer $v $rep > $J/answer_${v}_r${rep}.log 2>&1 &
    echo "$v $rep $!" >> $J/pids.txt
  done
done
wait
.venv/bin/python - <<'EOF' > $J/check.txt 2>&1
import json
from pathlib import Path
want = {f"multihop-rag-{i}" for i in json.loads(Path("runs/nograph/questions.json").read_text())["indices"]}
bad = 0
for d in sorted(Path("runs/handoff/runs").glob("*/r*")):
    p = d / "results.jsonl"
    rows = [json.loads(l) for l in open(p)] if p.exists() else []
    ids = {r["record_id"] for r in rows}
    empty = sum(not r["answer"] for r in rows)
    ok = len(rows) == 250 and ids == want
    bad += not ok
    print(f"{d.parent.name} {d.name}: {len(rows)} answers, {len(ids)} questions, {empty} empty answers{'' if ok else '  BAD'}")
print("ok" if not bad else f"{bad} bad")
raise SystemExit(1 if bad else 0)
EOF
if [ $? -eq 0 ]; then st done "9 jobs, 250 answers each (jobs/check.txt)" none
else st failed "see jobs/check.txt" "$(tail -n 1 $J/check.txt)"; exit 1; fi
