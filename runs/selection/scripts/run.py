"""One shard of the no-agent selection baselines (runs/selection/SPEC.md).

Usage (repo root):
  PYTHONPATH=. .venv/bin/python runs/selection/scripts/run.py VARIANT REP SHARD NSHARDS [LIMIT]

VARIANT:
  B1  Normal RAG, top 50 by vector score, cross-encoder (BAAI/bge-reranker-large) keeps 10
      (benchmark.py's existing --rerank path, candidates 50).
  B2  Normal RAG, top 50 by vector score, GPT-6 Luna picks up to 10 by ID.
  B3  GPT-6 Luna writes up to 3 search queries; the question's top 20 plus each
      query's top 10 are pooled (duplicates dropped, at most 50); GPT-6 Luna picks
      up to 10 by ID, as in B2.

Same 250 MultiHop-RAG comparison and temporal questions as runs/nograph
(runs/nograph/questions.json), same Normal RAG index (never rebuilt), same
answer model, prompt and settings. Picks fewer than 10 are filled in retrieval
order; no valid ID means the first 10. Every model call is metered into the
question's cost. Writes variant.json into the run folder; prints "RESULTS <path>".
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
variant, rep = sys.argv[1], sys.argv[2]
shard, shards = int(sys.argv[3]), int(sys.argv[4])
limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
assert variant in ("B1", "B2", "B3")
os.environ["RECRAG_RUN_NAMESPACE"] = "selection/runs"  # run folders go under runs/selection/runs/
indices = json.loads((ROOT / "runs/nograph/questions.json").read_text())["indices"][shard::shards]
if limit:
    indices = indices[:limit]
INDEX = ROOT / "runs/rag/indexes/07062033a2e929ec10405ef3a38d541e51b907cbb7041de545f72ab1e7b71799"
if not (INDEX / "index_store.json").exists():
    raise SystemExit(f"No Normal RAG index at {INDEX}; refusing to build.")

import benchmark  # noqa: E402
import rag_retrieval  # noqa: E402
from benchmark import Message, ReasoningStep, RetrievalOptions  # noqa: E402
from graphrag_retrieval import EVIDENCE_ID, evidence_id  # noqa: E402

# The run namespace moves run folders under runs/selection/runs/, which would
# also move the index path; point the index back at the existing one.
_retrieve = rag_retrieval.retrieve


def retrieve_existing(*, index_dir, **kwargs):
    if not (INDEX / "index_store.json").exists():
        raise RuntimeError("Normal RAG index missing: refusing to build.")
    return _retrieve(index_dir=INDEX, **kwargs)


rag_retrieval.retrieve = retrieve_existing

TOP = 10
# The agent's own picking sentences (graphrag_retrieval.AGENT_INSTRUCTION), word for word.
SELECT_INSTRUCTION = (
    "Every result line starts with an ID such as [E1a2b3c]. Do not answer the "
    "question. Reply only with the IDs of the results the answer needs, at most "
    f"{TOP}, most important first, one per line. Pick evidence for every "
    "article, person or part the question asks about."
)
QUERY_INSTRUCTION = (
    "Write search queries that find source passages for the question: one query "
    "per article, person or part the question asks about, at most 3. Do not answer "
    "the question. Reply only with the queries, one per line."
)


def pick(question: str, passages: list[str], scores: list[float], llm) -> tuple[list[str], list[float], str]:
    ids = [evidence_id(p) for p in passages]
    lines = "\n".join(f"[{i}] {' '.join(p.split())}" for i, p in zip(ids, passages))
    reply = llm.infer(
        [
            Message(role="system", content=SELECT_INSTRUCTION).model_dump(),
            Message(role="user", content=f"Question: {question}\n\nResults:\n{lines}").model_dump(),
        ]
    )[0]
    order: list[int] = []
    for found in EVIDENCE_ID.findall(reply):
        if found in ids and ids.index(found) not in order:
            order.append(ids.index(found))
    order = order[:TOP]
    order += [i for i in range(len(passages)) if i not in order][: TOP - len(order)]
    return [passages[i] for i in order], [scores[i] for i in order], reply


def select(question, initial, search, llm, options, reranker):
    trace = []
    if variant == "B3":
        reply = llm.infer(
            [
                Message(role="system", content=QUERY_INSTRUCTION).model_dump(),
                Message(role="user", content=question).model_dump(),
            ]
        )[0]
        queries = [q.strip(" -*\t") for q in reply.splitlines() if q.strip(" -*\t")][:3]
        trace.append({"tool": "write_queries", "output": reply, "queries": queries})
        passages, scores, seen = [], [], set()
        for query, k in [(question, 20)] + [(q, 10) for q in queries]:
            found = search(query)
            trace.append({"tool": "search", "query": query, "results": len(found.passages[:k])})
            for p, s in zip(found.passages[:k], found.scores[:k]):
                key = " ".join(p.lower().split())
                if key not in seen and len(passages) < 50:
                    seen.add(key)
                    passages.append(p)
                    scores.append(s)
    else:
        passages, scores = initial.passages, initial.scores
    kept, kept_scores, reply = pick(question, passages, scores, llm)
    trace.append({"tool": "select_evidence", "candidates": len(passages), "output": reply})
    initial.passages, initial.scores, initial.trace = kept, kept_scores, trace
    return [ReasoningStep(retrieval_query=t.get("query", t["tool"]), text=t.get("output", "")) for t in trace]


if variant == "B1":
    options = RetrievalOptions(top_k=TOP, rerank=True, rerank_candidate_k=50, rerank_top_k=TOP)
else:
    benchmark.refine_retrieval = select
    # B3's first search is the question's top 20; B2's is the question's top 50.
    options = RetrievalOptions(top_k=50 if variant == "B2" else 20)
os.environ["RECRAG_CONCURRENCY"] = "1"
path = benchmark.run("rag", "multihop_rag", 0, len(indices), options=options, record_indices=indices)
run_dir = Path(path).parent if Path(path).is_file() else Path(path)
(run_dir / "variant.json").write_text(
    json.dumps({"variant": variant, "rep": rep, "shard": shard, "shards": shards, "limit": limit,
                "select_instruction": SELECT_INSTRUCTION if variant != "B1" else None,
                "query_instruction": QUERY_INSTRUCTION if variant == "B3" else None}) + "\n"
)
print("RESULTS", path, flush=True)
