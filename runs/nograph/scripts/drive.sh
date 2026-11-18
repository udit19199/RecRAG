#!/bin/bash
# Graph ablation answers (runs/nograph/SPEC.md): 3 repeats x 2 shards at once.
# Writes runs/nograph/jobs/STATUS.md; "stage: done" only after each repeat has
# 250 rows in results.jsonl, every shard exited 0, and no row made a cypher call.
cd "$(dirname "$0")/../../.." || exit 1
R=runs/nograph; J=$R/jobs; SH=2
export PYTHONPATH=.
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $J/STATUS.md; }
st running "starting 3 repeats x $SH shards" none
pids=()
for rep in 1 2 3; do
  for s in $(seq 0 $((SH-1))); do
    .venv/bin/python $R/scripts/answer.py $rep $s $SH > $J/r${rep}_s${s}.log 2>&1 &
    pids+=($!); echo "$rep $s $!" >> $J/pids.txt
    sleep 3
  done
done
while :; do
  alive=0; for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && alive=$((alive+1)); done
  done_n=$(cat $J/r*_s*.log 2>/dev/null | grep -c "saved retrieval for")
  [ $alive -eq 0 ] && break
  st running "$done_n of 750 answers saved, $alive shards running" none
  sleep 60
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
.venv/bin/python - <<'EOF' > $J/check.txt 2>&1 || fail=1
import json, re, glob
from pathlib import Path
J = Path("runs/nograph/jobs"); reg = {}
for log in sorted(J.glob("r*_s*.log")):
    m = re.search(r"^RESULTS (.+)$", log.read_text(), re.M)
    if not m: raise SystemExit(f"{log.name}: no RESULTS line")
    p = Path(m.group(1)); d = p if p.is_dir() else p.parent
    reg.setdefault(log.name[1], []).append(str(d))
want = set(json.loads(Path("runs/nograph/questions.json").read_text())["indices"])
for rep, dirs in sorted(reg.items()):
    rows = [json.loads(l) for d in dirs for l in open(Path(d) / "retrieval.jsonl")]
    cy = sum(1 for r in rows for t in r["retrieval"]["trace"] if t["tool"] == "cypher_search")
    ids = {r["record_id"] for r in rows}
    print(f"repeat {rep}: {len(rows)} rows, {len(ids)} questions, {cy} cypher calls")
    if len(rows) != 250 or len(ids) != 250 or cy: raise SystemExit(f"repeat {rep} failed check")
json.dump(reg, open("runs/nograph/run_registry.json", "w"), indent=1)
print("ok")
EOF
if [ $fail -eq 0 ]; then st done "750 answers, 250 per repeat, 0 cypher calls (jobs/check.txt)" none
else st failed "see jobs/check.txt and shard logs" "$(tail -n 1 $J/check.txt; grep -h -m1 -i error $J/r*_s*.log | head -1 | cut -c1-200)"; exit 1; fi
