"""Gold facts found on NQ (68 questions with evidence), Normal RAG vs Agentic GraphRAG.

Cells [both, only Agentic, only Normal RAG, neither], exact McNemar, and how many
found facts Agentic GraphRAG got only from cypher_search rows (items that are
graph rows, not passages). Run from the repo root:
PYTHONPATH=. uv run python runs/nq100/scripts/gold_facts.py"""

import json
from math import comb
from pathlib import Path

from evals.haiku_fact_judge import JudgeTask, JudgeVerdict, _verified

F = Path("runs/nq100/fact_judge")
share = {}
links = {}
for line in (F / "per_question.jsonl").read_text().splitlines():
    r = json.loads(line)
    name = "graphrag" if "graphrag" in r["run"] else "rag"
    share[(name, r["record_id"])] = r["share"]
    links[(name, r["record_id"])] = r["task"]
qs = sorted({q for (_, q) in share})


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n)


out = {}
for th, label in [(0.5, ">=50%"), (0.75, ">=75%"), (1.0, "100%")]:
    a = [share[("graphrag", q)] >= th - 1e-9 for q in qs]
    b = [share[("rag", q)] >= th - 1e-9 for q in qs]
    cells = [sum(x and y for x, y in zip(a, b)), sum(x and not y for x, y in zip(a, b)),
             sum(y and not x for x, y in zip(a, b)), sum(not x and not y for x, y in zip(a, b))]
    p = mcnemar(cells[1], cells[2])
    out[label] = {"graphrag": sum(a), "rag": sum(b), "cells": cells, "p": round(p, 3)}
    print(f"{label}: Agentic {sum(a)} / Normal RAG {sum(b)}  [both, only Agentic, only RAG, neither] {cells}  p={p:.3f}")

# Facts Agentic GraphRAG found only through graph rows.
tasks = {t["id"]: JudgeTask.model_validate(t) for p in (F / "judge").glob("items-*.json") for t in json.loads(p.read_text())["items"]}
verdicts = {v["id"]: JudgeVerdict.model_validate(v) for p in (F / "judge").glob("verdict-*.json") for v in json.loads(p.read_text())}
found = rows_only = 0
questions_helped = set()
for q in qs:
    task, verdict = tasks[links[("graphrag", q)]], verdicts[links[("graphrag", q)]]
    for fact in verdict.facts:
        ok = [e for e in fact.evidence if _verified(e, task.items)]
        if fact.status != "stated" or not ok:
            continue
        found += 1
        if all(task.items[e.item - 1].lstrip().startswith("{") for e in ok):
            rows_only += 1
            questions_helped.add(q)
print(f"Agentic found facts: {found}; found only in graph rows: {rows_only} (in {len(questions_helped)} questions)")
out["graph_rows_only"] = {"facts": rows_only, "of_found": found, "questions": len(questions_helped)}
(Path("runs/nq100") / "gold_facts.json").write_text(json.dumps(out, indent=1))
