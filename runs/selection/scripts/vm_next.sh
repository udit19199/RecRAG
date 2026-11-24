#!/bin/bash
# Next batch on the Instabox VM (25 Nov 2026):
#   lane 1: B2 on the inference+null question set (retry; the first infnull run
#           died on an API error mid r1, see INFNULL_lanes.log)
#   lane 2: B4_200 on the default 250 comparison+temporal questions
#   lane 3: B2_100 on the default 250 (LLM picks up to 10 of the top 100)
# Three lanes in parallel (API-bound, one Python process each).
# A run whose log already has its RESULTS line is skipped, so the script can be rerun.
# Writes runs/selection/jobs/STATUS_NEXT.md; updates
# runs/selection/paired/infnull_registry.json and writes B4_200/B2_100 registry
# entries into runs/selection/paired/next_registry.json.
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=.
J=runs/selection/jobs; mkdir -p $J runs/selection/paired
S=$J/STATUS_NEXT.md
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $S; }

lane() {
  v=$1; suffix=$2
  for rep in 1 2 3; do
    log=$J/${v}_${suffix}_r${rep}.log
    grep -q '^RESULTS ' $log 2>/dev/null && continue
    echo "$(date -u +%FT%TZ) start $v $suffix r$rep" >> $J/NEXT_lanes.log
    .venv/bin/python runs/selection/scripts/run.py $v $rep 0 1 > $log 2>&1 \
      || { echo "$(date -u +%FT%TZ) FAILED $v $suffix r$rep" >> $J/NEXT_lanes.log; return 1; }
    echo "$(date -u +%FT%TZ) done $v $suffix r$rep" >> $J/NEXT_lanes.log
  done
}

st running "3 lanes: B2 infnull retry, B4_200, B2_100" none
( export RECRAG_QUESTIONS=runs/selection/questions_infnull.json; lane B2 infnull ) & p1=$!
lane B4_200 r & p2=$!
lane B2_100 r & p3=$!
echo "$p1 $p2 $p3" > $J/NEXT_pids.txt
ok=1
for p in $p1 $p2 $p3; do wait $p || ok=0; done
if [ $ok = 0 ]; then
  st failed "a lane failed (see jobs/NEXT_lanes.log)" "$(grep FAILED $J/NEXT_lanes.log | tail -n 1)"; exit 1
fi

.venv/bin/python - > $J/NEXT_check.txt 2>&1 <<'EOF' || { st failed "output check" "$(tail -n 1 $J/NEXT_check.txt)"; exit 1; }
import json, re
from pathlib import Path
J = Path("runs/selection/jobs")

def check(v, suffix, k, qtypes):
    out = {}
    for rep in "123":
        path = re.search(r"^RESULTS (.*)$", (J / f"{v}_{suffix}_r{rep}.log").read_text(), re.M).group(1)
        d = Path(path).parent if Path(path).is_file() else Path(path)
        res = [json.loads(l) for l in open(d / "results.jsonl")]
        ret = [json.loads(l) for l in open(d / "retrieval.jsonl")]
        assert len(res) == len(ret) == 250, (v, rep, len(res), len(ret))
        assert all(len(r["retrieval"]["passages"]) == k for r in ret), (v, rep)
        got = {}
        for r in res:
            got[r["question_type"]] = got.get(r["question_type"], 0) + 1
        assert got == qtypes, (v, rep, got)
        print(v, suffix, rep, d.name, "rows 250, passages", k, "empty answers", sum(not r["answer"] for r in res))
        out[rep] = [str(d.resolve())]
    return out

INF = {"inference_query": 125, "null_query": 125}
CT = {"comparison_query": 125, "temporal_query": 125}

# B2 infnull: redo inside the registry vm_infnull.sh wrote (B0/B4 already done)
reg_path = Path("runs/selection/paired/infnull_registry.json")
reg = json.loads(reg_path.read_text())
reg["B2_infnull"] = check("B2", "infnull", 10, INF)
reg_path.write_text(json.dumps(reg, indent=1) + "\n")

nxt = {"B4_200": check("B4_200", "r", 200, CT), "B2_100": check("B2_100", "r", 10, CT)}
Path("runs/selection/paired/next_registry.json").write_text(json.dumps(nxt, indent=1) + "\n")
EOF
st done "B2 infnull retry, B4_200, B2_100: 3 repeats x 250 rows each" none
