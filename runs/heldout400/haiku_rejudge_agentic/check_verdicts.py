"""Check Haiku verdict files of the final500 judge against their item files.

Usage: PYTHONPATH=. .venv/bin/python runs/final500/scripts/check_verdicts.py STAGE FIRST LAST
STAGE is claims, judge or audit (the Opus audit, checked like judge). Prints one line per file: "ok N" or "BAD N: reason".
A file is ok when it parses, has one entry per item id of its items file (no
unknown or repeated ids), and (claims) one fact list per sentence, none empty,
or (judge) exactly one entry per fact label with a valid status.
Exit code 1 if any file is bad or missing.
"""

import json
import sys
from pathlib import Path

from evals.haiku_fact_judge import ClaimsVerdict, JudgeVerdict, merged_single_passage

stage, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
folder = Path("runs/heldout400/haiku_rejudge_agentic") / stage
bad = 0
for n in range(first, last + 1):
    tag = "audit-" if stage == "audit" else ""
    items_path = folder / f"items-{tag}{n:03d}.json"
    verdict_path = folder / f"verdict-{tag}{n:03d}.json"
    if not items_path.exists():
        continue
    try:
        items = {i["id"]: i for i in json.loads(items_path.read_text())["items"]}
        rows = json.loads(verdict_path.read_text())
        ids = [row["id"] for row in rows]
        if sorted(ids) != sorted(items):
            raise ValueError(
                f"ids differ: missing {sorted(set(items) - set(ids))[:3]}, "
                f"unknown {sorted(set(ids) - set(items))[:3]}, repeated {len(ids) - len(set(ids))}"
            )
        for row in rows:
            if stage == "claims":
                v = merged_single_passage(
                    ClaimsVerdict.model_validate(row), len(items[row["id"]]["sentences"])
                )
                if len(v.sentences) != len(items[v.id]["sentences"]) or not all(v.sentences):
                    raise ValueError(f"{v.id}: sentence count or empty fact list")
            else:
                v = JudgeVerdict.model_validate(row)
                labels = sorted(l for g in items[v.id]["gold"] for l in g["facts"])
                if sorted(f.fact for f in v.facts) != labels:
                    raise ValueError(f"{v.id}: fact labels differ")
        print(f"ok {n}")
    except Exception as exc:  # noqa: BLE001 - report and continue
        bad += 1
        print(f"BAD {n}: {repr(exc)[:300]}")
sys.exit(1 if bad else 0)
