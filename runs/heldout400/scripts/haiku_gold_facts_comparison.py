"""Gold facts found (Claude Haiku judge) on the 100 comparison questions.

Reads runs/agentic_tune/fact_judge/per_question.jsonl. Per repeat: questions with
>=50% / >=75% / 100% of gold facts found, and pair cells
[both, Agentic GraphRAG only, Normal RAG only, neither] against Normal RAG of the
same repeat. Run from the repo root."""
import json
from pathlib import Path

H = Path("runs/heldout400")
dirs = {}
for line in open(H / "manifest.jsonl"):
    m = json.loads(line)
    if m["method"] in ("rag", "graphrag") and m["tag"] in ("r1", "r2", "r3") and m["exit"] == 0 and m["run_dir"]:
        d = m["run_dir"]
        dirs[str(Path(d) if d.startswith("/") else H / d)] = (m["method"], m["tag"])
dirs["runs/agentic_tune/graphrag/comparison100"] = ("graphrag", "new")
share = {}
for line in open("runs/agentic_tune/fact_judge/per_question.jsonl"):
    r = json.loads(line)
    share[dirs[r["run"]] + (r["record_id"],)] = r["share"]
TH = [("0.5", ">=50%"), ("0.75", ">=75%"), ("1.0", "100%")]
qs = sorted({k[2] for k in share})
def ok(method, tag, q, t): return share[(method, tag, q)] >= float(t) - 1e-9
out = {}
for tag in ("r1", "r2", "r3"):
    for t, label in TH:
        rag = sum(ok("rag", tag, q, t) for q in qs)
        for ag in (tag, "new"):
            a = [ok("graphrag", ag, q, t) for q in qs]; b = [ok("rag", tag, q, t) for q in qs]
            cells = [sum(x and y for x, y in zip(a, b)), sum(x and not y for x, y in zip(a, b)),
                     sum(y and not x for x, y in zip(a, b)), sum(not x and not y for x, y in zip(a, b))]
            out[f"{tag}|{label}|agentic-{'old' if ag == tag else 'new'}"] = {"agentic": sum(a), "rag": rag, "cells": cells}
            print(f"{tag} {label:6s} Agentic({'old '+tag if ag == tag else 'new'}) {sum(a):3d}  Normal RAG {rag:3d}  [both, agentic only, rag only, neither] {cells}")
print("Normal RAG identical across repeats:", all(share[("rag","r1",q)] == share[("rag",t,q)] for q in qs for t in ("r2","r3")))
Path("runs/agentic_tune/fact_judge/comparison_tables.json").write_text(json.dumps(out, indent=1))
