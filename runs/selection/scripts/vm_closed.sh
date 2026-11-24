#!/bin/bash
# CLOSED: closed book, no chunks at all, same answer prompt (runs/selection/SPEC.md, 25 Nov 2026). 3 repeats, one after another.
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=. CLOSED=1
J=runs/selection/jobs; S=$J/STATUS_CLOSED.md
st() { printf "stage: %s\nprogress: %s\nlast error: %s\nupdated: %s\n" "$1" "$2" "$3" "$(date -u +%FT%TZ)" > $S; }
st running "CLOSED r1-r3" none
for rep in 1 2 3; do
  log=$J/CLOSED_r${rep}.log
  grep -q '^RESULTS ' $log 2>/dev/null && continue
  .venv/bin/python runs/selection/scripts/wrong10.py $rep > $log 2>&1 || { st failed "r$rep" "$(tail -n 1 $log)"; exit 1; }
done
.venv/bin/python - > $J/CLOSED_check.txt 2>&1 <<'EOF' || { st failed "output check" "$(tail -n 1 $J/CLOSED_check.txt)"; exit 1; }
import json, re
from pathlib import Path
J = Path("runs/selection/jobs"); reg = {"CLOSED": {}}
for rep in "123":
    path = re.search(r"^RESULTS (.*)$", (J / f"CLOSED_r{rep}.log").read_text(), re.M).group(1)
    d = Path(path).parent if Path(path).is_file() else Path(path)
    res = [json.loads(l) for l in open(d / "results.jsonl")]; ret = [json.loads(l) for l in open(d / "retrieval.jsonl")]
    assert len(res) == len(ret) == 250 and all(not r["retrieval"]["passages"] for r in ret), rep
    print("CLOSED", rep, d.name, "rows 250, no passages, empty answers", sum(not r["answer"] for r in res))
    reg["CLOSED"][rep] = [str(d.resolve())]
Path("runs/selection/paired/CLOSED_registry.json").write_text(json.dumps(reg, indent=1) + "\n")
EOF
st done "CLOSED: 3 repeats x 250 rows" none
