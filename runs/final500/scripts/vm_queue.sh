#!/bin/bash
# Remaining Natural Questions work of the final 500 study, on the Instabox VM
# (4 vCPU / 8 GB, Neo4j in Docker), one heavy process at a time so it fits in
# memory. Each step is skipped when its output is already complete, so after a
# stop the script is simply started again.
# Writes runs/final500/jobs/vm_queue/STATUS.md (stage: running | done | failed).
cd /data/workspace/RecRAG || exit 1
R=runs/final500
Q=$R/jobs/vm_queue
mkdir -p $Q
export PYTHONPATH=.
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $Q/STATUS.md; }
fail() { st failed "$1" "$2"; exit 1; }

build() {  # METHOD PYTHON [env...]
  m=$1; py=$2; shift 2
  [ -f $R/natural_questions/$m/build/build.json ] && return
  st running "NQ $m build" none
  env "$@" $py $R/scripts/build.py natural_questions $m > $R/natural_questions/$m/build/build.log 2>&1 \
    || fail "NQ $m build" "$(tail -n 1 $R/natural_questions/$m/build/build.log | cut -c1-300)"
  [ -f $R/natural_questions/$m/build/build.json ] || fail "NQ $m build" "no build.json"
}

answer() {  # METHOD SHARDS REP
  name=nq_$1_r$3
  grep -q '^stage: done' $R/jobs/$name/STATUS.md 2>/dev/null && return
  st running "NQ $1 answers, repeat $3" none
  .venv/bin/python $R/scripts/drive.py $name $2 natural_questions:$1:$3 > $R/jobs/$name.log 2>&1
  grep -q '^stage: done' $R/jobs/$name/STATUS.md \
    || fail "NQ $1 answers, repeat $3" "$(grep '^last error' $R/jobs/$name/STATUS.md | cut -c1-300)"
}

mkdir -p $R/natural_questions/graphrag/indexes $R/natural_questions/raptor/build
build graphrag .venv/bin/python RECRAG_CONFIG=$R/config.build.toml
build raptor .venv/bin/python RECRAG_RAPTOR_WORKERS=8 RECRAG_RAPTOR_CHECKPOINT_EVERY=500
for rep in 1 2 3; do answer rag 2 $rep; done
for rep in 1 2 3; do answer raptor 1 $rep; done
for rep in 1 2 3; do answer graphrag 3 $rep; done
for rep in 1 2 3; do answer hipporag 1 $rep; done
st done "NQ graph and RAPTOR builds, 4 methods x 3 repeats answered, row counts checked by drive.py" none
