"""Answered right on NQ 100: Normal RAG vs Agentic GraphRAG, one repeat.

Rules fixed before running (runs/nq100/README.md): exact match, else the Haiku
answer judge; gold "No answer": "Insufficient Information" is right, anything
else wrong. Cells: [both, only Agentic, only Normal RAG, neither], exact McNemar.
Run from the repo root: PYTHONPATH=. uv run python runs/nq100/scripts/answered_right.py"""

import json
from math import comb
from pathlib import Path

from evals.qa_metrics import normalize_answer
from evals.subagent_judge import ABSTENTION, answer_key

RUNS = {
    "rag": Path("runs/nq100/rag/20261110T200339107433Z"),
    "graphrag": Path("runs/nq100/graphrag/run1"),
}
J = Path("runs/nq100/judge")
verdicts = {
    v["id"]: v["correct"]
    for path in sorted(J.glob("verdicts-answers-*.json"))
    for v in json.loads(path.read_text())
}


def right(r: dict) -> bool:
    abstained = normalize_answer(r["answer"]) == ABSTENTION
    if r["expected_answers"] == ["No answer"]:
        return abstained
    if r["exact_match"]:
        return True
    if abstained:
        return False
    return bool(verdicts[answer_key(r["question"], r["answer"])])


res = {}
for name, d in RUNS.items():
    for line in (d / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        res[(name, r["record_id"])] = (right(r), r["expected_answers"] == ["No answer"])
qs = sorted({q for (_, q) in res})


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n)


out = {}
for label, keep in [("all 100", lambda q: True), ("gold not 'No answer'", lambda q: not res[("rag", q)][1]), ("gold 'No answer'", lambda q: res[("rag", q)][1])]:
    sel = [q for q in qs if keep(q)]
    a = [res[("graphrag", q)][0] for q in sel]
    b = [res[("rag", q)][0] for q in sel]
    cells = [sum(x and y for x, y in zip(a, b)), sum(x and not y for x, y in zip(a, b)),
             sum(y and not x for x, y in zip(a, b)), sum(not x and not y for x, y in zip(a, b))]
    p = mcnemar(cells[1], cells[2])
    out[label] = {"n": len(sel), "graphrag": sum(a), "rag": sum(b), "cells": cells, "p": round(p, 3)}
    print(f"{label} (n={len(sel)}): Agentic {sum(a)} / Normal RAG {sum(b)}  [both, only Agentic, only RAG, neither] {cells}  p={p:.3f}")
(Path("runs/nq100") / "answered_right.json").write_text(json.dumps(out, indent=1))
