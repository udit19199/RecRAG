#!/bin/bash
# No-agent selection baselines (runs/selection/SPEC.md): B2, B3, B1 x repeats 1-3,
# one job per (variant, repeat), at most 2 at a time (Mac budget). Writes
# runs/selection/jobs/STATUS.md; "stage: done" only after every job has 250
# answered rows, 250 distinct questions, and (B2/B3) a select_evidence step per row.
cd "$(dirname "$0")/../../.." || exit 1
R=runs/selection; J=$R/jobs; mkdir -p $J
export PYTHONPATH=.
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $J/STATUS.md; }
jobs_list="B2:1 B2:2 B2:3 B3:1 B3:2 B3:3 B1:1 B1:2 B1:3"
st running "starting 9 jobs, 2 at a time" none
running=()
for job in $jobs_list; do
  v=${job%:*}; rep=${job#*:}
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do
    st running "$(grep -h -c 'saved retrieval for' $J/*.log 2>/dev/null | paste -sd+ - | bc) of 2250 questions saved" none
    sleep 30
  done
  .venv/bin/python $R/scripts/run.py $v $rep 0 1 > $J/${v}_r${rep}.log 2>&1 &
  echo "$v $rep $!" >> $J/pids.txt
  sleep 5
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
  st running "$(grep -h -c 'saved retrieval for' $J/*.log 2>/dev/null | paste -sd+ - | bc) of 2250 questions saved" none
  sleep 30
done
fail=0; for p in $(jobs -p); do wait $p || fail=1; done
.venv/bin/python - <<'EOF' > $J/check.txt 2>&1 || fail=1
import json, re
from pathlib import Path
J = Path("runs/selection/jobs"); reg = {}
want = {f"multihop-rag-{i}" for i in json.loads(Path("runs/nograph/questions.json").read_text())["indices"]}
for log in sorted(J.glob("B*_r*.log")):
    m = re.search(r"^RESULTS (.+)$", log.read_text(), re.M)
    if not m: raise SystemExit(f"{log.name}: no RESULTS line")
    d = Path(m.group(1)).parent
    v, rep = log.stem.split("_r")
    rows = [json.loads(l) for l in open(d / "results.jsonl")]
    ret = [json.loads(l) for l in open(d / "retrieval.jsonl")]
    ids = {r["record_id"] for r in rows}
    sel = sum(any(t.get("tool") == "select_evidence" for t in r["retrieval"]["trace"]) for r in ret)
    err = sum(bool(r["retrieval_error"]) for r in ret)
    print(f"{v} r{rep}: {len(rows)} rows, {len(ids)} questions, {sel} with selection, {err} retrieval errors")
    if len(rows) != 250 or ids != want or (v != "B1" and sel != 250):
        raise SystemExit(f"{v} r{rep} failed check")
    reg.setdefault(v, {})[rep] = str(d)
json.dump(reg, open("runs/selection/run_registry.json", "w"), indent=1)
print("ok")
EOF
if [ $fail -eq 0 ]; then st done "9 jobs, 250 questions each (jobs/check.txt)" none
else st failed "see jobs/check.txt and job logs" "$(tail -n 1 $J/check.txt)"; exit 1; fi
