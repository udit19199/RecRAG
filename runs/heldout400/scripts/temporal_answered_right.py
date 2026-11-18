"""Answered right on the 100 temporal questions: every method vs Normal RAG, per repeat.
Cells [both, only method, only Normal RAG, neither]. Run from the repo root:
PYTHONPATH=. uv run python runs/heldout400/scripts/temporal_answered_right.py"""
import json, glob
from pathlib import Path
from evals.subagent_judge import answer_key
H = Path("runs/heldout400")
verd = {}
for f in sorted(glob.glob(str(H / "judge/verdicts-*.json"))) + sorted(glob.glob("runs/agentic_tune/judge/verdicts-*.json")):
    for v in json.load(open(f)):
        verd[v["id"]] = v
def ok(r):
    return bool(r["exact_match"]) or bool(verd.get(answer_key(r["question"], r["answer"]), {}).get("correct"))
res = {}
for line in open(H / "manifest.jsonl"):
    m = json.loads(line)
    if m["tag"] in ("r1", "r2", "r3") and m["exit"] == 0 and m["run_dir"]:
        d = m["run_dir"]; d = d if d.startswith("/") else str(H / d)
        for l in open(Path(d) / "results.jsonl"):
            r = json.loads(l)
            if r["question_type"] == "temporal_query":
                res.setdefault((m["method"], m["tag"]), {})[r["record_id"]] = ok(r)
for tag, name in zip(("r1", "r2", "r3"), ("temporal100", "temporal100_rep2", "temporal100_rep3")):
    for l in open(f"runs/agentic_tune/graphrag/{name}/results.jsonl"):
        r = json.loads(l); res.setdefault(("agentic_10nov", tag), {})[r["record_id"]] = ok(r)
out = {}
for (meth, tag), v in sorted(res.items()):
    rag = res[("rag", tag)]; qs = sorted(rag); assert len(v) == 100 and set(v) == set(qs), (meth, tag, len(v))
    c = [sum(v[q] and rag[q] for q in qs), sum(v[q] and not rag[q] for q in qs), sum(rag[q] and not v[q] for q in qs), sum(not v[q] and not rag[q] for q in qs)]
    out[f"{meth}|{tag}"] = {"right": sum(v.values()), "cells": c}
    print(meth, tag, sum(v.values()), c)
Path("runs/agentic_tune/temporal_answered_right.json").write_text(json.dumps(out, indent=1))
