#!/bin/bash
# Inference + null questions (runs/selection/questions_infnull.json: the 125 + 125
# latest-run questions) on the Instabox VM. Three setups on the same frozen Normal
# RAG index: B0 (top 10 straight), B2 (LLM picks 10 of top 50), B4 (top 50 straight).
# Three lanes in parallel (API-bound, one Python process each).
# A run whose log already has its RESULTS line is skipped, so the script can be rerun.
# Writes runs/selection/jobs/STATUS_INFNULL.md and
# runs/selection/paired/infnull_registry.json.
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=.
export RECRAG_QUESTIONS=runs/selection/questions_infnull.json
J=runs/selection/jobs; mkdir -p $J runs/selection/paired
S=$J/STATUS_INFNULL.md
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $S; }

lane() {
  v=$1
  for rep in 1 2 3; do
    log=$J/${v}_infnull_r${rep}.log
    grep -q '^RESULTS ' $log 2>/dev/null && continue
    echo "$(date -u +%FT%TZ) start $v r$rep" >> $J/INFNULL_lanes.log
    .venv/bin/python runs/selection/scripts/run.py $v $rep 0 1 > $log 2>&1 \
      || { echo "$(date -u +%FT%TZ) FAILED $v r$rep" >> $J/INFNULL_lanes.log; return 1; }
    echo "$(date -u +%FT%TZ) done $v r$rep" >> $J/INFNULL_lanes.log
  done
}

st running "3 lanes: B0, B2, B4 on inference+null" none
lane B0 & p1=$!
lane B2 & p2=$!
lane B4 & p3=$!
echo "$p1 $p2 $p3" > $J/INFNULL_pids.txt
ok=1
for p in $p1 $p2 $p3; do wait $p || ok=0; done
if [ $ok = 0 ]; then
  st failed "a lane failed (see jobs/INFNULL_lanes.log)" "$(grep FAILED $J/INFNULL_lanes.log | tail -n 1)"; exit 1
fi

.venv/bin/python - > $J/INFNULL_check.txt 2>&1 <<'EOF' || { st failed "output check" "$(tail -n 1 $J/INFNULL_check.txt)"; exit 1; }
import json, re
from pathlib import Path
J = Path("runs/selection/jobs")
reg = {}
for v, k in (("B0", 10), ("B2", 10), ("B4", 50)):
    for rep in "123":
        path = re.search(r"^RESULTS (.*)$", (J / f"{v}_infnull_r{rep}.log").read_text(), re.M).group(1)
        d = Path(path).parent if Path(path).is_file() else Path(path)
        res = [json.loads(l) for l in open(d / "results.jsonl")]
        ret = [json.loads(l) for l in open(d / "retrieval.jsonl")]
        assert len(res) == len(ret) == 250, (v, rep, len(res), len(ret))
        assert all(len(r["retrieval"]["passages"]) == k for r in ret), (v, rep)
        qtypes = {}
        for r in res:
            qtypes[r["question_type"]] = qtypes.get(r["question_type"], 0) + 1
        assert qtypes == {"inference_query": 125, "null_query": 125}, (v, rep, qtypes)
        print(v, rep, d.name, "rows 250, passages", k, "empty answers", sum(not r["answer"] for r in res))
        reg.setdefault(f"{v}_infnull", {})[rep] = [str(d.resolve())]
Path("runs/selection/paired/infnull_registry.json").write_text(json.dumps(reg, indent=1) + "\n")
EOF
st done "B0, B2, B4 on inference+null: 3 repeats x 250 rows each" none
