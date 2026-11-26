#!/bin/bash
# 26 Nov 2026 batch (runs/selection/SPEC.md, "MultiHop-RAG inference + null chunk
# counts, and the HotpotQA top-200 rerun"):
#   - MultiHop-RAG inference + null (250 q): B4_20, B4_30, B4_100, B4_200 straight, WRONG10
#   - HotpotQA B4_200 straight (500 q, the rerun of the lost run)
# 1 repeat, flex, no retries. At most 3 lanes at once (8 OOM'd the VM on 25 Nov).
# A job whose log already has RESULTS is skipped, so the script can be rerun.
# Writes jobs/STATUS_BATCH2.md and jobs/BATCH2_check.txt.
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=. RECRAG_QUESTION_SERVICE_TIER=flex HF_HOME=$HOME/.cache/huggingface
J=runs/selection/jobs; mkdir -p $J runs/selection/paired
S=$J/STATUS_BATCH2.md
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $S; }
LANES=${LANES:-3}

runit() { # $1 tag, $2 log, rest: command
  tag=$1; log=$2; shift 2
  grep -q '^RESULTS ' $log 2>/dev/null && return 0
  echo "$(date -u +%FT%TZ) start $tag" >> $J/BATCH2_lanes.log
  "$@" > $log 2>&1 || { echo "$(date -u +%FT%TZ) FAILED $tag" >> $J/BATCH2_lanes.log; return 1; }
  echo "$(date -u +%FT%TZ) done $tag" >> $J/BATCH2_lanes.log
}

st running "6 jobs, $LANES at a time" none
running=0
launch() { # sequential launch, throttled to $LANES concurrent
  while [ "$(jobs -rp | wc -l)" -ge "$LANES" ]; do wait -n; done
  "$@" &
}

launch runit hp_B4_200 $J/hp_B4_200.log env RECRAG_DATASET=hotpotqa .venv/bin/python runs/selection/scripts/run.py B4_200 1 0 1
for v in B4_20 B4_30 B4_100 B4_200; do
  launch runit mh_$v $J/mh_${v}_infnull.log env RECRAG_QUESTIONS=runs/selection/questions_infnull.json .venv/bin/python runs/selection/scripts/run.py $v 1 0 1
done
launch runit mh_WRONG10 $J/mh_WRONG10_infnull.log .venv/bin/python runs/selection/scripts/wrong10_infnull.py 1
wait

.venv/bin/python - > $J/BATCH2_check.txt 2>&1 <<'EOF' || { st failed "output check" "$(tail -n 1 $J/BATCH2_check.txt)"; exit 1; }
import json, re
from pathlib import Path
J = Path("runs/selection/jobs")
reg = {}
def check(tag, logname, want_rows, k, qtypes=None):
    path = re.search(r"^RESULTS (.*)$", (J / logname).read_text(), re.M).group(1)
    d = Path(path).parent if Path(path).is_file() else Path(path)
    res = [json.loads(l) for l in open(d / "results.jsonl")]
    ret = [json.loads(l) for l in open(d / "retrieval.jsonl")]
    assert len(res) == len(ret) == want_rows, (tag, len(res), len(ret))
    assert all(len(r["retrieval"]["passages"]) == k for r in ret), tag
    got = {}
    for r in res:
        got[r["question_type"]] = got.get(r["question_type"], 0) + 1
    if qtypes:
        assert got == qtypes, (tag, got)
    print(tag, d.name, "rows", len(res), "passages", k, "empty answers", sum(not r["answer"] for r in res))
    return str(d.resolve())

INF = {"inference_query": 125, "null_query": 125}
reg["B4_20_infnull"] = {"1": [check("mh B4_20", "mh_B4_20_infnull.log", 250, 20, INF)]}
reg["B4_30_infnull"] = {"1": [check("mh B4_30", "mh_B4_30_infnull.log", 250, 30, INF)]}
reg["B4_100_infnull"] = {"1": [check("mh B4_100", "mh_B4_100_infnull.log", 250, 100, INF)]}
reg["B4_200_infnull"] = {"1": [check("mh B4_200", "mh_B4_200_infnull.log", 250, 200, INF)]}
reg["WRONG10_infnull"] = {"1": [check("mh WRONG10", "mh_WRONG10_infnull.log", 250, 10, INF)]}
reg["B4_200_hotpotqa"] = {"1": [check("hp B4_200", "hp_B4_200.log", 500, 200)]}
old = json.loads(Path("runs/selection/paired/infnull_registry.json").read_text()) if Path("runs/selection/paired/infnull_registry.json").exists() else {}
Path("runs/selection/paired/infnull_registry.json").write_text(json.dumps({**old, **reg}, indent=1) + "\n")
EOF
st done "6 jobs: see BATCH2_check.txt and BATCH2_lanes.log" none
