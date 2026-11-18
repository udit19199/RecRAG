"""Checks behind the findings slide (comparison and temporal, Haiku judges only).

1. Answered right by gold facts found, per method and repeat (comparison).
2. "Insufficient Information" refusals per method and repeat (temporal and comparison).
Run from the repo root: PYTHONPATH=. uv run python runs/heldout400/scripts/reasons_check.py"""

import glob
import json
from pathlib import Path

from evals.qa_metrics import normalize_answer
from evals.subagent_judge import answer_key

H = Path("runs/heldout400")
verd = {}
for f in sorted(glob.glob(str(H / "judge/verdicts-*.json"))) + sorted(glob.glob("runs/agentic_tune/judge/verdicts-*.json")):
    for v in json.load(open(f)):
        verd[v["id"]] = v
REFUSE = normalize_answer("Insufficient Information")


def right(r):
    return bool(r["exact_match"]) or bool(verd.get(answer_key(r["question"], r["answer"]), {}).get("correct"))


def runs(kind):
    """{(method, tag): {record_id: (result, run_dir)}}"""
    out = {}
    for line in open(H / "manifest.jsonl"):
        m = json.loads(line)
        if m["tag"] in ("r1", "r2", "r3") and m["exit"] == 0 and m["run_dir"]:
            d = m["run_dir"]; d = d if d.startswith("/") else str(H / d)
            for x in open(Path(d) / "results.jsonl"):
                r = json.loads(x)
                if r["question_type"] == kind:
                    out.setdefault((m["method"], m["tag"]), {})[r["record_id"]] = (r, d)
    names = {"comparison_query": ("comparison100", "comparison100_rep2", "comparison100_rep3"),
             "temporal_query": ("temporal100", "temporal100_rep2", "temporal100_rep3")}[kind]
    for tag, n in zip(("r1", "r2", "r3"), names):
        d = f"runs/agentic_tune/graphrag/{n}"
        for x in open(Path(d) / "results.jsonl"):
            r = json.loads(x)
            out.setdefault(("agentic_10nov", tag), {})[r["record_id"]] = (r, d)
    return out


out = {}
share = {}
for line in open("runs/agentic_tune/fact_judge/per_question.jsonl"):
    r = json.loads(line)
    share[(str(Path(r["run"]).resolve()), r["record_id"])] = r["share"]
comp = runs("comparison_query")
print("== Comparison: Answered right among questions by gold facts found")
for (meth, tag) in [("rag", t) for t in ("r1", "r2", "r3")] + [("agentic_10nov", t) for t in ("r1", "r2", "r3")] + [("graphrag", t) for t in ("r1", "r2", "r3")]:
    rows = comp[(meth, tag)]
    b = {"0%": [0, 0], "some, <100%": [0, 0], "100%": [0, 0]}
    for q, (r, d) in rows.items():
        s = share.get((str(Path(d).resolve()), q))
        if s is None:
            continue
        k = "0%" if s == 0 else "100%" if s >= 1 - 1e-9 else "some, <100%"
        b[k][0] += 1; b[k][1] += right(r)
    out[f"comparison|{meth}|{tag}"] = b
    print(meth, tag, {k: f"{v[1]}/{v[0]} right" for k, v in b.items()})
for kind in ("comparison_query", "temporal_query"):
    print(f"== {kind}: 'Insufficient Information' answers per repeat")
    res = runs(kind) if kind == "temporal_query" else comp
    for meth in ("rag", "graphrag", "agentic_10nov", "hipporag", "raptor"):
        vals = [sum(normalize_answer(r["answer"]) == REFUSE for r, _ in res[(meth, t)].values()) for t in ("r1", "r2", "r3")]
        out[f"refusals|{kind}|{meth}"] = vals
        print(meth, vals)
Path("runs/agentic_tune/reasons_check.json").write_text(json.dumps(out, indent=1))

print("== Temporal: questions only Agentic 10 Nov got right; how many Normal RAG refused")
tem = runs("temporal_query")
for t in ("r1", "r2", "r3"):
    a, b = tem[("agentic_10nov", t)], tem[("rag", t)]
    only = [q for q in a if right(a[q][0]) and not right(b[q][0])]
    ref = sum(normalize_answer(b[q][0]["answer"]) == REFUSE for q in only)
    out[f"temporal_only_agentic_rag_refused|{t}"] = [ref, len(only)]
    print(t, f"Normal RAG refused {ref} of the {len(only)} questions only Agentic got right")
Path("runs/agentic_tune/reasons_check.json").write_text(json.dumps(out, indent=1))
