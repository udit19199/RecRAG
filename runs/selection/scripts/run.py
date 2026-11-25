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
  B4      Normal RAG, top 50 by vector score (B2's candidates), all 50 sent to the
          answer model in retrieval order. No picking call. (Bhanu, 24 Nov 2026)
  B2_100  as B2, candidates top 100 instead of top 50 (25 Nov 2026).
  B4_20   as B4, top 20.      B4_30   as B4, top 30.      B4_100  as B4, top 100 (25 Nov 2026).
  B4_200  as B4, top 200 (25 Nov 2026).
  B4_rev  as B4, the 50 in reversed order (best-scoring last).
  B4_shuf as B4, the 50 shuffled; the order is fixed per question (seeded by the
          question text), so the 3 repeats differ only in the answer call.

Same 250 MultiHop-RAG comparison and temporal questions as runs/nograph
(runs/nograph/questions.json), same Normal RAG index (never rebuilt), same
answer model, prompt and settings. Picks fewer than 10 are filled in retrieval
order; no valid ID means the first 10. Every model call is metered into the
question's cost. Writes variant.json into the run folder; prints "RESULTS <path>".

Milvus (runs/milvus/SPEC.md): with RECRAG_VECTOR_DB=milvus, every vector search
goes to the rag_chunks collection in self-hosted Milvus (same 7,728 chunks, same
saved llama-index vectors, HNSW) instead of the llama-index store; the query is
embedded the same way (text-embedding-3-small via the pipeline's embedder). Run
folders go under runs/milvus/runs/. Variant B0 is plain Normal RAG, top 10
(it works without Milvus too: top 10 straight through the same code path, no
picking; used for the inference/null question set, 25 Nov 2026).

RECRAG_QUESTIONS (path to a questions json with "indices") overrides the
question set; default runs/nograph/questions.json.
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
MILVUS = os.environ.get("RECRAG_VECTOR_DB") == "milvus"
B4 = {"B4": 50, "B4_20": 20, "B4_30": 30, "B4_100": 100, "B4_200": 200, "B4_rev": 50, "B4_shuf": 50}
B2 = {"B2": 50, "B2_100": 100}
assert variant in ("B0", "B1", "B3", *B2, *B4)
os.environ["RECRAG_RUN_NAMESPACE"] = "milvus/runs" if MILVUS else "selection/runs"
questions_path = Path(os.environ.get("RECRAG_QUESTIONS", ROOT / "runs/nograph/questions.json"))
indices = json.loads((ROOT / questions_path if not questions_path.is_absolute() else questions_path).read_text())["indices"][shard::shards]
if limit:
    indices = indices[:limit]
INDEX = ROOT / "runs/rag/indexes/07062033a2e929ec10405ef3a38d541e51b907cbb7041de545f72ab1e7b71799"
if not MILVUS and not (INDEX / "index_store.json").exists():
    raise SystemExit(f"No Normal RAG index at {INDEX}; refusing to build.")

import benchmark  # noqa: E402
import rag_retrieval  # noqa: E402
from benchmark import Message, ReasoningStep, RetrievalOptions  # noqa: E402

if variant == "B1":
    try:
        import sentence_transformers  # noqa: F401
    except ModuleNotFoundError:
        # B1 runs in .venv-hipporag (torch, transformers; no sentence_transformers).
        # Same model and scoring as sentence_transformers.CrossEncoder.predict for a
        # one-label reranker: tokenized (query, passage) pairs, truncated to the
        # model's 512 tokens, sigmoid of the single logit. Order is what matters.
        import types

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # The cross-encoder is deterministic, so each (question, passage) pair is
        # scored once and the score saved (runs/selection/cache/rerank_scores.jsonl);
        # repeats 2 and 3 reuse it and differ only in the answer call.
        from hashlib import sha256

        CACHE = ROOT / "runs/selection/cache/rerank_scores.jsonl"

        class CrossEncoder:
            def __init__(self, name):
                self.tok = AutoTokenizer.from_pretrained(name)
                self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                self.saved = {}
                if CACHE.exists():
                    for line in CACHE.read_text().splitlines():
                        row = json.loads(line)
                        self.saved[row["key"]] = row["score"]

            @staticmethod
            def key(question, passage):
                return sha256(f"BAAI/bge-reranker-large\0{question}\0{passage}".encode()).hexdigest()

            @torch.no_grad()
            def predict(self, pairs, batch_size=16):
                keys = [self.key(a, b) for a, b in pairs]
                todo = [i for i, k in enumerate(keys) if k not in self.saved]
                for i in range(0, len(todo), batch_size):
                    batch = [pairs[j] for j in todo[i : i + batch_size]]
                    enc = self.tok([a for a, _ in batch], [b for _, b in batch], padding=True,
                                   truncation=True, max_length=512, return_tensors="pt")
                    scores = torch.sigmoid(self.model(**enc).logits.view(-1)).tolist()
                    with CACHE.open("a") as f:
                        for j, s in zip(todo[i : i + batch_size], scores):
                            self.saved[keys[j]] = s
                            f.write(json.dumps({"key": keys[j], "score": s}) + "\n")
                return [self.saved[k] for k in keys]

        sys.modules["sentence_transformers"] = types.SimpleNamespace(CrossEncoder=CrossEncoder)
else:
    from graphrag_retrieval import EVIDENCE_ID, evidence_id  # noqa: E402

# The run namespace moves run folders under runs/selection/runs/, which would
# also move the index path; point the index back at the existing one.
_retrieve = rag_retrieval.retrieve


def retrieve_existing(*, index_dir, **kwargs):
    if not (INDEX / "index_store.json").exists():
        raise RuntimeError("Normal RAG index missing: refusing to build.")
    return _retrieve(index_dir=INDEX, **kwargs)


def retrieve_milvus(*, documents, question, index_dir, embeddings, top_k):
    from time import perf_counter

    from runs.milvus.scripts.milvus_store import search

    started = perf_counter()
    hits = search("rag_chunks", embeddings.get_query_embedding(question), top_k, ["text"])
    return benchmark.RetrievedContext(
        passages=[h["text"] for h in hits], scores=[h["score"] for h in hits],
        construction_seconds=0.0, retrieval_seconds=perf_counter() - started,
    )


rag_retrieval.retrieve = retrieve_milvus if MILVUS else retrieve_existing

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
    try:
        reply = llm.infer(
            [
                Message(role="system", content=SELECT_INSTRUCTION).model_dump(),
                Message(role="user", content=f"Question: {question}\n\nResults:\n{lines}").model_dump(),
            ]
        )[0]
    except RuntimeError as e:
        # On unanswerable (null) questions the model often returns an empty
        # response instead of naming IDs; treat that as no picks, so the fill
        # rule below (first 10 by score) applies instead of a failed question.
        if "did not complete" not in str(e):
            raise
        reply = ""
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


def reorder(question, initial, search, llm, options, reranker):
    """B4_rev / B4_shuf: same top 50, different order; no model call."""
    import random
    from hashlib import sha256

    order = list(range(len(initial.passages)))
    if variant == "B4_rev":
        order.reverse()
    else:
        random.Random(sha256(question.encode()).hexdigest()).shuffle(order)
    initial.passages = [initial.passages[i] for i in order]
    initial.scores = [initial.scores[i] for i in order]
    initial.trace = [{"tool": "reorder", "order": order}]
    return [ReasoningStep(retrieval_query="reorder", text=" ".join(map(str, order)))]


if variant == "B0":
    options = RetrievalOptions(top_k=TOP)
elif variant in B4:
    # Same retrieval call as B2 (Normal RAG index, question top k by vector
    # score), but the picking call is skipped: every candidate goes to the answer.
    options = RetrievalOptions(top_k=B4[variant])
    if variant in ("B4_rev", "B4_shuf"):
        benchmark.refine_retrieval = reorder
elif variant == "B1":
    options = RetrievalOptions(top_k=TOP, rerank=True, rerank_candidate_k=50, rerank_top_k=TOP)
else:
    benchmark.refine_retrieval = select
    # B3's first search is the question's top 20; B2's is the question's top 50,
    # B2_100's the top 100.
    options = RetrievalOptions(top_k=B2.get(variant, 20))
os.environ["RECRAG_CONCURRENCY"] = "1"
path = benchmark.run("rag", "multihop_rag", 0, len(indices), options=options, record_indices=indices)
run_dir = Path(path).parent if Path(path).is_file() else Path(path)
(run_dir / "variant.json").write_text(
    json.dumps({"variant": variant, "vector_db": "milvus 2.5.10 rag_chunks" if MILVUS else "llama-index", "rep": rep, "shard": shard, "shards": shards, "limit": limit,
                "select_instruction": SELECT_INSTRUCTION if variant in ("B2", "B2_100", "B3") else None,
                "query_instruction": QUERY_INSTRUCTION if variant == "B3" else None,
                "passages_to_answer": B4.get(variant, TOP),
                "order": {"B4_rev": "reversed", "B4_shuf": "shuffled, seed sha256(question)"}.get(variant, "retrieval")}) + "\n"
)
print("RESULTS", path, flush=True)
