"""Gold facts found: question as search text vs agent's first search text.
Per repeat, as both / only agent / only question / neither, with exact McNemar.
Run from the repo root: python3 runs/heldout400/scripts/search_text_compare.py"""
import json
from math import comb
from pathlib import Path

J = Path("runs/agentic_tune/search_text/judge/per_question.jsonl")
share = {}
for line in open(J):
    r = json.loads(line)
    share[(Path(r["run"]).name, r["record_id"])] = r["share"]
qs = sorted({rid for (_, rid) in share})


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n)


out = {}
for th, label in [(0.5, ">=50%"), (0.75, ">=75%"), (1.0, "100%")]:
    for tag in ("r1", "r2", "r3"):
        a = [share[(f"agent_first_{tag}", q)] >= th - 1e-9 for q in qs]
        b = [share[("question", q)] >= th - 1e-9 for q in qs]
        cells = [sum(x and y for x, y in zip(a, b)), sum(x and not y for x, y in zip(a, b)),
                 sum(y and not x for x, y in zip(a, b)), sum(not x and not y for x, y in zip(a, b))]
        p = mcnemar(cells[1], cells[2])
        out[f"{label} {tag}"] = {"agent": sum(a), "question": sum(b), "cells": cells, "p": round(p, 3)}
        print(f"{label} {tag}: agent {sum(a)} / question {sum(b)}  [both, only agent, only question, neither] {cells}  p={p:.3f}")
Path("runs/agentic_tune/search_text/comparison.json").write_text(json.dumps(out, indent=1))
