#!/bin/bash
# Selection setups on HotpotQA / Natural Questions at the flex price, and
# B4_200_rev on MultiHop-RAG (runs/selection/SPEC.md, "the selection setups on
# HotpotQA and Natural Questions, at the flex price", 25 Nov 2026).
#
# Usage: bash runs/selection/scripts/vm_datasets.sh DATASET [LIMIT]
#   DATASET: hotpotqa | natural_questions | multihop_rev
#   LIMIT:   optional, passed to run.py (trial runs)
# One repeat, no reruns: a variant whose log already has RESULTS is skipped;
# a failed variant is recorded in the status file and the script goes on.
# At most 3 lanes in parallel. Writes jobs/STATUS_DS_<DATASET>.md and
# jobs/DS_<DATASET>_check.txt.
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=. RECRAG_QUESTION_SERVICE_TIER=flex HF_HOME=$HOME/.cache/huggingface
DS=$1; LIMIT=${2:-}
J=runs/selection/jobs; mkdir -p $J runs/selection/paired
S=$J/STATUS_DS_${DS}.md
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $S; }

case $DS in
  hotpotqa|natural_questions)
    VARIANTS="B4_20 B4_30 B4 B4_100 B4_200 B2 B2_100 B1"; QS=500 ;;
  multihop_rev)
    VARIANTS="B4_200_rev"; QS=250 ;;
  *) echo "unknown dataset $DS" >&2; exit 1 ;;
esac

runit() {
  v=$1
  log=$J/${DS}_${v}.log
  grep -q '^RESULTS ' $log 2>/dev/null && return 0
  echo "$(date -u +%FT%TZ) start $DS $v" >> $J/DS_${DS}_lanes.log
  if [ "$DS" = multihop_rev ]; then
    RECRAG_DATASET=multihop_rag .venv/bin/python runs/selection/scripts/run.py $v 1 0 1 $LIMIT > $log 2>&1
  else
    RECRAG_DATASET=$DS .venv/bin/python runs/selection/scripts/run.py $v 1 0 1 $LIMIT > $log 2>&1
  fi \
    && echo "$(date -u +%FT%TZ) done $DS $v" >> $J/DS_${DS}_lanes.log \
    || echo "$(date -u +%FT%TZ) FAILED $DS $v" >> $J/DS_${DS}_lanes.log
}

st running "$DS: $VARIANTS" none
n=0
for v in $VARIANTS; do
  runit $v &
  n=$((n + 1))
  [ $((n % 3)) = 0 ] && wait
done
wait

.venv/bin/python - "$DS" "$LIMIT" $VARIANTS > $J/DS_${DS}_check.txt 2>&1 <<'EOF'
import json, re, sys
from pathlib import Path
J = Path("runs/selection/jobs")
ds, limit, variants = sys.argv[1], sys.argv[2], sys.argv[3:]
qs = 250 if ds == "multihop_rev" else 500
want = int(limit) if limit else qs
K = {"B4_20": 20, "B4_30": 30, "B4": 50, "B4_100": 100, "B4_200": 200,
     "B4_200_rev": 200, "B2": 10, "B2_100": 10, "B1": 10}
for v in variants:
    log = J / f"{ds}_{v}.log"
    txt = log.read_text() if log.exists() else ""
    m = re.search(r"^RESULTS (.*)$", txt, re.M)
    if not m:
        print(v, "FAILED (no RESULTS; last line:", txt.strip().splitlines()[-1][:160] if txt.strip() else "no log", ")")
        continue
    d = Path(m.group(1)).parent if Path(m.group(1)).is_file() else Path(m.group(1))
    res = [json.loads(l) for l in open(d / "results.jsonl")]
    ret = [json.loads(l) for l in open(d / "retrieval.jsonl")]
    bad = [v, len(res), len(ret)]
    assert len(res) == len(ret) == want, bad
    if v.startswith("B4"):
        assert all(len(r["retrieval"]["passages"]) == K[v] for r in ret), v
    else:
        assert all(len(r["retrieval"]["passages"]) <= 10 for r in ret), v
    types = {}
    for r in res:
        types[r["question_type"]] = types.get(r["question_type"], 0) + 1
    print(v, d.name, "rows", len(res), "passages<=", K[v],
          "types", types, "empty answers", sum(not r["answer"] for r in res),
          "failed", sum(1 for r in res if r.get("failed_attempts")))
EOF
fails=$(grep -c FAILED $J/DS_${DS}_lanes.log 2>/dev/null || echo 0)
if [ "$fails" -gt 0 ]; then
  st partial "$(grep -c ' done ' $J/DS_${DS}_lanes.log) variants done, $fails failed" "$(grep FAILED $J/DS_${DS}_lanes.log | tail -n 1)"
else
  st done "$DS: all variants" none
fi
