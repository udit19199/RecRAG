#!/bin/bash
# B4_100 (top 100 straight to the answer) and WRONG10 (each question answered from another question's
# Normal RAG top 10) on the Instabox VM (runs/selection/SPEC.md, "25 Nov 2026").
# Two lanes in parallel (API-bound, one Python process each):
#   lane 1: B4_100 r1-r3        lane 2: WRONG10 r1-r3
# A run whose log already has its RESULTS line is skipped, so the script can be rerun after a pause.
# Writes runs/selection/jobs/STATUS_NEW.md and runs/selection/paired/{B4_100,WRONG10}_registry.json.
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=.
J=runs/selection/jobs; mkdir -p $J runs/selection/paired
S=$J/STATUS_NEW.md
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $S; }

lane() {
  v=$1
  for rep in 1 2 3; do
    log=$J/${v}_r${rep}.log
    grep -q '^RESULTS ' $log 2>/dev/null && continue
    echo "$(date -u +%FT%TZ) start $v r$rep" >> $J/NEW_lanes.log
    if [ "$v" = WRONG10 ]; then .venv/bin/python runs/selection/scripts/wrong10.py $rep > $log 2>&1
    else .venv/bin/python runs/selection/scripts/run.py $v $rep 0 1 > $log 2>&1; fi \
      || { echo "$(date -u +%FT%TZ) FAILED $v r$rep" >> $J/NEW_lanes.log; return 1; }
    echo "$(date -u +%FT%TZ) done $v r$rep" >> $J/NEW_lanes.log
  done
}

st running "2 lanes: B4_100, WRONG10" none
lane B4_100 & p1=$!
lane WRONG10 & p2=$!
echo "$p1 $p2" > $J/NEW_pids.txt
ok=1
for p in $p1 $p2; do wait $p || ok=0; done
if [ $ok = 0 ]; then
  st failed "a lane failed (see jobs/NEW_lanes.log)" "$(grep FAILED $J/NEW_lanes.log | tail -n 1)"; exit 1
fi

.venv/bin/python - > $J/NEW_check.txt 2>&1 <<'EOF' || { st failed "output check" "$(tail -n 1 $J/NEW_check.txt)"; exit 1; }
import json, re
from pathlib import Path
J = Path("runs/selection/jobs")
for v, k in (("B4_100", 100), ("WRONG10", 10)):
    reg = {v: {}}
    for rep in "123":
        path = re.search(r"^RESULTS (.*)$", (J / f"{v}_r{rep}.log").read_text(), re.M).group(1)
        d = Path(path).parent if Path(path).is_file() else Path(path)
        res = [json.loads(l) for l in open(d / "results.jsonl")]
        ret = [json.loads(l) for l in open(d / "retrieval.jsonl")]
        assert len(res) == len(ret) == 250, (v, rep, len(res), len(ret))
        assert all(len(r["retrieval"]["passages"]) == k for r in ret), (v, rep)
        print(v, rep, d.name, "rows 250, passages", k, "empty answers", sum(not r["answer"] for r in res))
        reg[v][rep] = [str(d.resolve())]
    Path(f"runs/selection/paired/{v}_registry.json").write_text(json.dumps(reg, indent=1) + "\n")
EOF
st done "B4_100 and WRONG10: 3 repeats x 250 rows each" none
