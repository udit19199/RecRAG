#!/bin/bash
# Matched-budget agent + forced-answer controls (runs/selection/runs/matched/SPEC.md, 27 Nov 2026).
# Answer step only, on retrieval files built on the Mac by matched_prep.py. 9 jobs, 3 lanes.
# A job whose results.jsonl already has 250 rows is skipped, so the script can be rerun.
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=.
M=runs/selection/runs/matched; S=$M/STATUS.md
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $S; }
job() { # $1 run dir
  d=$1; n=$(basename $d)
  [ "$(wc -l < $d/results.jsonl 2>/dev/null)" = "250" ] && return 0
  rm -f $d/results.jsonl
  forced=0; case $n in FORCED*) forced=1 ;; esac
  RECRAG_FORCED_ANSWER=$forced .venv/bin/python -c "from pathlib import Path; from benchmark import answer_saved; print('RESULTS', answer_saved(Path('$d/retrieval.jsonl')))" > $d/answer.log 2>&1 \
    || { echo "FAILED $n" >> $M/lanes.log; return 1; }
  echo "$(date -u +%FT%TZ) done $n" >> $M/lanes.log
}
st running "9 jobs, 3 lanes" none
fail=0
for d in $M/*_r*; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n || fail=1; done
  job $d &
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || fail=1; done
[ $fail = 0 ] || { st failed "a job failed" "$(grep FAILED $M/lanes.log | tail -1)"; exit 1; }
.venv/bin/python - > $M/check.txt 2>&1 <<'PY' || { st failed "output check" "$(tail -n 1 $M/check.txt)"; exit 1; }
import json, glob
for d in sorted(glob.glob("runs/selection/runs/matched/*_r*")):
    res = [json.loads(l) for l in open(d + "/results.jsonl")]
    v = json.load(open(d + "/variant.json"))
    k = {len(json.loads(l)["retrieval"]["passages"]) for l in open(d + "/retrieval.jsonl")}
    assert len(res) == 250 and k == {v["passages_to_answer"]}, (d, len(res), k)
    ins = sum("insufficient information" in (r["answer"] or "").lower() for r in res)
    print(d.split("/")[-1], "rows", len(res), "passages", k, "empty", sum(not r["answer"] for r in res), "insufficient", ins)
PY
st done "9 x 250 answers, checked" none
