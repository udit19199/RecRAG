"""How many of Agentic GraphRAG's 10 items are the same chunks Normal RAG retrieves, per type and repeat.
Run from the repo root: PYTHONPATH=. uv run python runs/heldout400/scripts/agentic_vs_rag_overlap.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
from evals.retrieval_quality import normalize

H = Path("runs/heldout400")
Q = json.load(open(H / "questions.json"))
qtype = {f"multihop-rag-{i}": t for t, ids in Q["by_type"].items() for i in ids}
ret = defaultdict(dict)
for m in map(json.loads, open(H / "manifest.jsonl")):
    if m["tag"] in ("r1", "r2", "r3") and m["run_dir"] and m["method"] in ("rag", "graphrag"):
        for r in map(json.loads, open(Path(m["run_dir"]) / "retrieval.jsonl")):
            ret[(m["tag"], m["method"])][r["record_id"]] = r["retrieval"]["passages"]
def split(p):
    try:
        d = json.loads(p)
        if isinstance(d, dict) and set(d) == {"title", "text", "entities", "facts"}:
            return "chunk", normalize(d["text"])
        return "graph", None
    except ValueError:
        return "chunk", normalize(p)
for qt in ["comparison_query", "temporal_query", "inference_query", "null_query"]:
    for t in ("r1", "r2", "r3"):
        same = Counter(); graph = 0; n = 0
        for q in (q for q in qtype if qtype[q] == qt):
            rag = {normalize(p) for p in ret[(t, "rag")][q]}
            items = [split(p) for p in ret[(t, "graphrag")][q]]
            same[sum(k == "chunk" and x in rag for k, x in items)] += 1
            graph += sum(k == "graph" for k, _ in items); n += len(items)
        print(f"{qt[:5]} {t}: graph-query rows {graph} of {n} items | questions by how many of the 10 are Normal RAG's chunks:",
              dict(sorted(same.items(), reverse=True)))
