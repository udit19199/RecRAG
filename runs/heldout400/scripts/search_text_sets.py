"""Which search text finds more gold facts: the question as written, or the
agent's own first vector_search text? Comparison questions, 10 Nov 2026 runs.

Writes one folder per set under runs/agentic_tune/search_text/, each with a
retrieval.jsonl in the usual format so evals/haiku_fact_judge.py can judge it:
- question/      the question searched as written on the graph's chunk index
                 (top 10, same index and code as the agent's vector_search;
                 needs Neo4j; 100 embedding calls, no LLM)
- agent_first_rN/  the 10 passages the agent's first vector_search returned in
                 repeat N (from the saved trace, no new calls)
Passages are the plain chunk text. Run from the repo root:
PYTHONPATH=. uv run python runs/heldout400/scripts/search_text_sets.py"""

import json
from pathlib import Path

from graphrag_retrieval import GraphRAG, _ChunkVectorRetriever

NEW = {
    "r1": "runs/agentic_tune/graphrag/comparison100",
    "r2": "runs/agentic_tune/graphrag/comparison100_rep2",
    "r3": "runs/agentic_tune/graphrag/comparison100_rep3",
}
OUT = Path("runs/agentic_tune/search_text")
DATABASE = "recrag-benchmark-e5315c2b5c9f57e2595cc51c"


def texts(output: str) -> list[str]:
    rows = []
    for line in output.split("\n"):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "text" in row:
            rows.append(row["text"])
    return rows


def write(name: str, rows: list[dict]) -> None:
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "retrieval.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    print(name, len(rows))


base = [json.loads(line) for line in open(Path(NEW["r1"]) / "retrieval.jsonl")]
for tag, run in NEW.items():
    rows = []
    for line in open(Path(run) / "retrieval.jsonl"):
        saved = json.loads(line)
        first = next(
            s for s in saved["retrieval"]["trace"] if s["tool"] == "vector_search"
        )
        saved["retrieval"]["passages"] = texts(first["output"])
        saved["retrieval"]["first_search_text"] = first["args"].get("query_text", "")
        rows.append(saved)
    write(f"agent_first_{tag}", rows)

rag = GraphRAG.from_config(dataset="multihop_rag")
try:
    vector = _ChunkVectorRetriever(
        driver=rag._driver, database=DATABASE, embedder=rag._embedder
    )
    rows = []
    for saved in base:
        result = vector.get_search_results(saved["question"])
        saved["retrieval"]["passages"] = [r.data()["text"] for r in result.records]
        saved["retrieval"]["first_search_text"] = saved["question"]
        rows.append(saved)
    write("question", rows)
finally:
    rag.close()
