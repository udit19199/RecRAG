"""One shard of the no-graph agent on Milvus (runs/milvus/SPEC.md), on the VM.

Usage (repo root): PYTHONPATH=. .venv/bin/python runs/milvus/scripts/agent.py REP SHARD NSHARDS [LIMIT]

The same agent as runs/nograph (RECRAG_AGENT_NO_GRAPH=1): one tool, vector_search,
with the same description and the same instruction (NO_GRAPH_AGENT_INSTRUCTION),
the same model-call limit, evidence picking (up to 10 by ID, rest filled by best
score), answer prompt and settings. The one difference: vector_search queries the
agent_chunks collection in Milvus (same 7,728 chunks, same saved vectors, top 10)
instead of Neo4j's vector index. Results carry the passage and its page title,
with no entities or graph facts, exactly as in the no-graph run. Scores are
reported as (1 + cos) / 2 like Neo4j's cosine index. Neo4j is never contacted.
Same 250 questions (runs/nograph/questions.json); shard s of n takes s, s+n, ...
"""

import json
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
rep, shard, shards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
os.environ["RECRAG_AGENT_NO_GRAPH"] = "1"
os.environ["RECRAG_RUN_NAMESPACE"] = "milvus/runs"
os.environ["RECRAG_CONCURRENCY"] = "1"
os.environ.setdefault("NEO4J_URI", "bolt://127.0.0.1:9")  # lazy driver, never opened
os.environ.setdefault("NEO4J_AUTH", "neo4j/unused")
indices = json.loads((ROOT / "runs/nograph/questions.json").read_text())["indices"][shard::shards]
if limit:
    indices = indices[:limit]

import neo4j  # noqa: E402
from neo4j_graphrag.retrievers.base import Retriever  # noqa: E402
from neo4j_graphrag.types import RawSearchResult  # noqa: E402

import benchmark  # noqa: E402
import graphrag_retrieval as g  # noqa: E402
from runs.milvus.scripts.milvus_store import search  # noqa: E402

assert g.no_graph_agent()


class MilvusChunkRetriever(Retriever):
    VERIFY_NEO4J_VERSION = False

    def __init__(self, driver, embedder) -> None:
        super().__init__(driver)
        self.embedder = embedder
        self.result_formatter = g.format_vector_record

    def get_search_results(self, query_text: str) -> RawSearchResult:
        """Find source passages similar in meaning to the query text."""
        hits = search("agent_chunks", self.embedder.embed_query(query_text), g.RETRIEVAL_TOP_K, ["title", "text"])
        return RawSearchResult(
            records=[
                neo4j.Record({"title": h["title"], "text": h["text"], "entities": [], "facts": [],
                              "score": (1 + h["score"]) / 2})
                for h in hits
            ]
        )


g.AgenticToolsRetriever.VERIFY_NEO4J_VERSION = False
_agent = None


def milvus_retrieve(*, rag, pages, question, index_dir, top_k):
    global _agent
    if _agent is None:
        tool = MilvusChunkRetriever(rag._driver, rag._embedder).convert_to_tool(
            name="vector_search",
            description=(
                "Find source text passages similar in meaning to the "
                "query. Each result has the passage and its page title."
            ),
            parameter_descriptions={"query_text": "What to search for, in plain words."},
        )
        _agent = g.AgenticToolsRetriever(
            driver=rag._driver, llm=rag._cypher_llm, agent_llm=rag._agent_llm, tools=[tool],
            neo4j_database="milvus-agent_chunks", system_instruction=g.NO_GRAPH_AGENT_INSTRUCTION,
        )
    started = perf_counter()
    result = _agent.search(question, top_k=top_k)
    meta = result.metadata or {}
    return benchmark.RetrievedContext(
        passages=[g.answer_text(str(i.content)) for i in result.items],
        evidence=[str(i.content) for i in result.items],
        trace=meta.get("trace", []),
        scores=[float(i.metadata["score"]) if i.metadata and i.metadata.get("score") is not None else g.NO_SCORE
                for i in result.items],
        construction_seconds=0.0,
        retrieval_seconds=perf_counter() - started,
    )


g.retrieve = milvus_retrieve
path = benchmark.run("graphrag", "multihop_rag", 0, len(indices), record_indices=indices)
run_dir = Path(path).parent if Path(path).is_file() else Path(path)
(run_dir / "milvus.json").write_text(json.dumps(
    {"vector_db": "milvus 2.5.10", "collection": "agent_chunks", "RECRAG_AGENT_NO_GRAPH": "1",
     "rep": rep, "shard": shard, "shards": shards, "limit": limit}) + "\n")
print("RESULTS", path, flush=True)
