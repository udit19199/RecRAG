"""Per question: when Agentic GraphRAG finds more (or fewer) gold facts than Normal RAG, who answers right.
Run from the repo root: PYTHONPATH=. uv run python runs/heldout400/scripts/agentic_more_facts.py
"""
import glob, json
from collections import defaultdict
from pathlib import Path
from evals.subagent_judge import answer_key
H=Path("runs/heldout400"); Q=json.load(open(H/"questions.json"))
qtype={f"multihop-rag-{i}":t for t,ids in Q["by_type"].items() for i in ids}
REPS=["r1","r2","r3"]
ver={}
for f in sorted(glob.glob(str(H/"judge"/"verdicts-*.json"))):
    for v in json.load(open(f)): ver[v["id"]]=v
right=defaultdict(dict); facts=defaultdict(dict)
for m in map(json.loads, open(H/"manifest.jsonl")):
    if m["tag"] in REPS and m["run_dir"] and m["method"] in ("rag","graphrag"):
        for r in map(json.loads, open(Path(m["run_dir"])/"results.jsonl")):
            v=ver.get(answer_key(r["question"],r["answer"]))
            right[(m["tag"],m["method"])][r["record_id"]]=bool(r["exact_match"]) or bool(v and v.get("correct"))
for t in REPS:
    for s in range(4):
        for r in map(json.loads, open(H/"judge_support"/f"{t}-s{s}"/"per_question.jsonl")):
            if r["method"] not in ("rag","graphrag"): continue
            c=[x for g in r["gold"] for x in g["claims"]]
            facts[(t,r["method"])][r["record_id"]]=sum(x["full_status"]=="stated" for x in c)/len(c) if c else None
for qt in ["comparison_query","temporal_query"]:
    qs=[q for q in qtype if qtype[q]==qt]
    for t in REPS:
        more=[q for q in qs if facts[(t,"graphrag")][q]>facts[(t,"rag")][q]]
        less=[q for q in qs if facts[(t,"graphrag")][q]<facts[(t,"rag")][q]]
        full=[q for q in more if facts[(t,"graphrag")][q]==1]
        g=lambda L,m: sum(right[(t,m)][q] for q in L)
        print(f"{qt[:4]} {t}: Agentic finds more facts on {len(more)} (to 100% on {len(full)}); right there: Agentic {g(more,'graphrag')}, Normal RAG {g(more,'rag')} | Agentic finds fewer on {len(less)}; right there: Agentic {g(less,'graphrag')}, Normal RAG {g(less,'rag')}")
