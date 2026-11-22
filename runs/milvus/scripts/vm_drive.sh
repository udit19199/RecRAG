#!/bin/bash
# Milvus runs on the Instabox VM (runs/milvus/SPEC.md), strictly one job at a time.
# 1. wait until B1 (runs/selection/scripts/vm_b1.sh) reports done; 2. delete the
# reranker model to free disk; 3. load Milvus; 4. smoke-test the agent on 2
# questions; 5. agent, B0, B2 on Milvus: repeats 1-3, 2 shards of 125 questions.
# A shard whose log already has its RESULTS line is skipped, so the script can be
# rerun after a pause. Writes runs/milvus/jobs/STATUS.md.
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=.
J=runs/milvus/jobs; mkdir -p $J
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $J/STATUS.md; }
fail() { st failed "$1" "$(tail -n 1 "$2" 2>/dev/null | cut -c1-200)"; exit 1; }

st waiting "for B1 to finish" none
until grep -q '^stage: done' runs/selection/jobs/STATUS_B1.md 2>/dev/null; do
  grep -q '^stage: failed' runs/selection/jobs/STATUS_B1.md 2>/dev/null && fail "B1 failed; Milvus not started" runs/selection/jobs/STATUS_B1.md
  sleep 60
done
rm -rf "$HOME/.cache/huggingface/hub/models--BAAI--bge-reranker-large" "$HOME"/.cache/huggingface/hub/blobs/*

if ! grep -q 'rag_chunks rows 7728' $J/load.log 2>/dev/null; then
  st running "loading Milvus" none
  .venv/bin/python runs/milvus/scripts/load.py > $J/load.log 2>&1 || fail "load" $J/load.log
fi
if ! grep -q '^RESULTS ' $J/smoke.log 2>/dev/null; then
  st running "agent smoke test, 2 questions" none
  .venv/bin/python runs/milvus/scripts/agent.py smoke 0 1 2 > $J/smoke.log 2>&1 || fail "agent smoke" $J/smoke.log
  # Both rows answered, each with at least one vector_search that returned 10
  # results, no other tool, and 10 items sent to the answer model.
  .venv/bin/python - "$(grep -oP '^RESULTS \K.*' $J/smoke.log)" > $J/smoke_check.txt 2>&1 <<'EOF' || fail "agent smoke check" $J/smoke_check.txt
import json, sys
from pathlib import Path
d = Path(sys.argv[1]).parent
ret = [json.loads(l) for l in open(d / "retrieval.jsonl")]
res = [json.loads(l) for l in open(d / "results.jsonl")]
assert len(ret) == len(res) == 2, (len(ret), len(res))
for r in ret:
    tools = [t["tool"] for t in r["retrieval"]["trace"]]
    assert set(tools) <= {"vector_search", "select_evidence"}, tools
    assert any(t["tool"] == "vector_search" and t["results"] == 10 for t in r["retrieval"]["trace"]), tools
    assert len(r["retrieval"]["passages"]) == 10 and not r["retrieval_error"]
print("smoke ok", [t["tool"] for t in ret[0]["retrieval"]["trace"]], [x["answer"][:30] for x in res])
EOF
  mkdir -p /tmp/milvus_smoke && mv "$(dirname "$(grep -oP '^RESULTS \K.*' $J/smoke.log)")" /tmp/milvus_smoke/
fi
for job in agent B0 B2; do
  for rep in 1 2 3; do
    for s in 0 1; do
      log=$J/${job}_r${rep}_s${s}.log
      grep -q '^RESULTS ' $log 2>/dev/null && continue
      st running "$job repeat $rep shard $s of 2" none
      if [ $job = agent ]; then
        .venv/bin/python runs/milvus/scripts/agent.py $rep $s 2 > $log 2>&1 || fail "$job r$rep s$s" $log
      else
        RECRAG_VECTOR_DB=milvus .venv/bin/python runs/selection/scripts/run.py $job $rep $s 2 > $log 2>&1 || fail "$job r$rep s$s" $log
      fi
    done
  done
done
st done "agent, B0, B2 on Milvus: 3 repeats x 2 shards each" none
