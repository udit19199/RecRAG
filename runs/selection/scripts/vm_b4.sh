#!/bin/bash
# B4 runs on the Instabox VM (runs/selection/SPEC.md, "B4: top k straight to the answer").
# Three lanes in parallel (API-bound, one Python process each):
#   lane 1: B4 (top 50) r1-r3, then B4_rev r1-r3
#   lane 2: B4_20 r1-r3, then B4_shuf r1-r3
#   lane 3: B4_30 r1-r3
# Each run: all 250 questions, one process. A run whose log already has its
# RESULTS line is skipped, so the script can be rerun after a pause.
# Then: registry of B4 runs (jobs/B4_registry.json) and the ceiling check.
# Writes runs/selection/jobs/STATUS_B4.md.
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=.
J=runs/selection/jobs; mkdir -p $J
S=$J/STATUS_B4.md
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $S; }

lane() {
  for v in "$@"; do
    for rep in 1 2 3; do
      log=$J/${v}_r${rep}.log
      grep -q '^RESULTS ' $log 2>/dev/null && continue
      echo "$(date -u +%FT%TZ) start $v r$rep" >> $J/B4_lanes.log
      .venv/bin/python runs/selection/scripts/run.py $v $rep 0 1 > $log 2>&1 \
        || { echo "$(date -u +%FT%TZ) FAILED $v r$rep" >> $J/B4_lanes.log; return 1; }
      echo "$(date -u +%FT%TZ) done $v r$rep" >> $J/B4_lanes.log
    done
  done
}

st running "3 lanes: B4+B4_rev, B4_20+B4_shuf, B4_30" none
lane B4 B4_rev & p1=$!
lane B4_20 B4_shuf & p2=$!
lane B4_30 & p3=$!
echo "$p1 $p2 $p3" > $J/B4_pids.txt
ok=1
for p in $p1 $p2 $p3; do wait $p || ok=0; done
if [ $ok = 0 ]; then
  st failed "a lane failed (see jobs/B4_lanes.log)" "$(grep FAILED $J/B4_lanes.log | tail -n 1)"; exit 1
fi

# Registry from the RESULTS lines; each run must hold 250 answered rows.
.venv/bin/python - > $J/B4_check.txt 2>&1 <<'EOF' || { st failed "output check" "$(tail -n 1 $J/B4_check.txt)"; exit 1; }
import json, re
from pathlib import Path
J = Path("runs/selection/jobs")
reg = {}
for v in ("B4", "B4_20", "B4_30", "B4_rev", "B4_shuf"):
    for rep in "123":
        path = re.search(r"^RESULTS (.*)$", (J / f"{v}_r{rep}.log").read_text(), re.M).group(1)
        d = Path(path).parent if Path(path).is_file() else Path(path)
        res = [json.loads(l) for l in open(d / "results.jsonl")]
        ret = [json.loads(l) for l in open(d / "retrieval.jsonl")]
        k = json.loads((d / "variant.json").read_text())["passages_to_answer"]
        assert len(res) == len(ret) == 250, (v, rep, len(res), len(ret))
        assert all(len(r["retrieval"]["passages"]) == k for r in ret), (v, rep)
        empty = sum(not r["answer"] for r in res)
        print(v, rep, d.name, "rows 250, passages", k, "empty answers", empty)
        reg.setdefault(v, {})[rep] = [str(d.resolve())]
(J / "B4_registry.json").write_text(json.dumps(reg, indent=1) + "\n")
EOF
st running "ceiling check" none
.venv/bin/python - > $J/B4_ceiling.log 2>&1 <<'EOF' || { st failed "ceiling" "$(tail -n 1 $J/B4_ceiling.log)"; exit 1; }
import json, runpy
from pathlib import Path
reg = json.loads(Path("runs/selection/run_registry.json").read_text())
reg.update(json.loads(Path("runs/selection/jobs/B4_registry.json").read_text()))
Path("runs/selection/run_registry.json").write_text(json.dumps(reg, indent=1) + "\n")
runpy.run_path("runs/selection/scripts/ceiling.py", run_name="__main__")
EOF
st done "B4, B4_20, B4_30, B4_rev, B4_shuf: 3 repeats x 250 rows each; ceiling written" none
