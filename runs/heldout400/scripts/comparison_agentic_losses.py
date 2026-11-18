"""Comparison questions: where Agentic GraphRAG finds fewer gold facts than Normal RAG, which Normal RAG
chunk held the missed facts, at what Normal RAG rank, and what took its place in Agentic GraphRAG's 10.
Run from the repo root: PYTHONPATH=. uv run python runs/heldout400/scripts/comparison_agentic_losses.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
from evals.retrieval_quality import normalize

H = Path("runs/heldout400")
Q = json.load(open(H / "questions.json"))
qtype = {f"multihop-rag-{i}": t for t, ids in Q["by_type"].items() for i in ids}
QT = "comparison_query"
qs = sorted(q for q in qtype if qtype[q] == QT)
ret = defaultdict(dict)
for m in map(json.loads, open(H / "manifest.jsonl")):
    if m["tag"] in ("r1", "r2", "r3") and m["run_dir"] and m["method"] in ("rag", "graphrag"):
        for r in map(json.loads, open(Path(m["run_dir"]) / "retrieval.jsonl")):
            ret[(m["tag"], m["method"])][r["record_id"]] = (r["retrieval"]["passages"], r["retrieval"].get("scores"))
claims = defaultdict(dict)
for t in ("r1", "r2", "r3"):
    for s in range(4):
        for r in map(json.loads, open(H / "judge_support" / f"{t}-s{s}" / "per_question.jsonl")):
            if r["record_id"] in qtype and qtype[r["record_id"]] == QT and r["method"] in ("rag", "graphrag"):
                claims[(t, r["method"])][r["record_id"]] = [c for g in r["gold"] for c in g["claims"]]
def chunk_text(p):
    try:
        d = json.loads(p)
        return normalize(d["text"]) if isinstance(d, dict) and "text" in d and "facts" in d else None
    except ValueError:
        return normalize(p)
share = lambda cl: sum(c["full_status"] == "stated" for c in cl) / len(cl)

for t in ("r1", "r2", "r3"):
    lost_rank = Counter(); in_agentic = Counter(); losses = 0; why = Counter()
    for q in qs:
        rp, rs = ret[(t, "rag")][q]; gp, gs = ret[(t, "graphrag")][q]
        rc, gc = claims[(t, "rag")][q], claims[(t, "graphrag")][q]
        if share(gc) >= share(rc):
            continue
        losses += 1
        g_chunks = [chunk_text(p) for p in gp]
        g_set = {c for c in g_chunks if c}
        ncy = sum(c is None for c in g_chunks)
        # claims Normal RAG found that Agentic did not: which Normal RAG chunk holds the quote
        missed_ranks = set()
        for a, b in zip(rc, gc):
            if a["full_status"] == "stated" and b["full_status"] != "stated":
                for k, p in enumerate(rp):
                    if any(normalize(x) and normalize(x) in normalize(p) for x in a["quotes"]):
                        missed_ranks.add(k + 1); break
        for k in missed_ranks:
            lost_rank[k] += 1
            in_agentic["chunk was in Agentic's 10" if normalize(rp[k - 1]) in g_set else "chunk not in Agentic's 10"] += 1
        why[f"graph rows in Agentic's 10: {ncy}"] += 1
    print(f"{t}: Agentic finds fewer facts on {losses}")
    print("   Normal RAG rank of the chunk holding the missed facts:", dict(sorted(lost_rank.items())))
    print("  ", dict(in_agentic), "|", dict(sorted(why.items())))

# first chunk identity and how far Normal RAG's top 10 survive in Agentic's list
print()
for t in ("r1", "r2", "r3"):
    first_same = 0; kept_by_rank = Counter()
    for q in qs:
        rp, rs = ret[(t, "rag")][q]; gp, gs = ret[(t, "graphrag")][q]
        g = [chunk_text(p) for p in gp]
        first_same += bool(g and g[0] == normalize(rp[0]))
        gset = set(g)
        for k, p in enumerate(rp):
            kept_by_rank[k + 1] += normalize(p) in gset
    print(f"{t}: Agentic's first item is Normal RAG's first chunk on {first_same}/100; "
          f"Normal RAG chunk at rank 1..10 also in Agentic's 10:", [kept_by_rank[k] for k in range(1, 11)])
q = qs[0]; print("score scales, example:", ret[("r1", "rag")][q][1][:3], ret[("r1", "graphrag")][q][1][:3])

