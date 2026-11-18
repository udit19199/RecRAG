"""Natural Questions, first 100 validation rows (indices 0-99), one method.

Usage: PYTHONPATH=. uv run python runs/nq100/scripts/run.py METHOD [DONE_RETRIEVAL_JSONL ...]
METHOD is rag or graphrag. Results go to runs/nq100/<method>/<timestamp>/.
Each question has its own Wikipedia page, so each gets its own index (for
graphrag: its own Neo4j database, built on first use and reused afterwards).
Rows already saved in any DONE_RETRIEVAL_JSONL are skipped (resume).
NQ_LIMIT=N runs only the first N questions still to do (cost check pilot).

Decisions (10-11 Nov 2026): first 100 rows in order, no filtering; Normal RAG
and Agentic GraphRAG only; general ontology for NQ
(graphrag_construction.ontology_for); warm-up off (see runs/nq100/README.md)."""

import json
import os
import sys
from pathlib import Path

method = sys.argv[1]
assert method in ("rag", "graphrag"), method
# runs/nq100/questions.json: the frozen 100 (validation index, NQ id, page title).
frozen = json.loads((Path(__file__).parents[1] / "questions.json").read_text())
rows = frozen["questions"]
assert [row["index"] for row in rows] == list(range(100))
done = {
    json.loads(line)["record_id"]
    for path in sys.argv[2:]
    for line in Path(path).read_text().splitlines()
    if line.strip()
}
indices = [row["index"] for row in rows if row["id"] not in done]
if "NQ_LIMIT" in os.environ:
    indices = indices[: int(os.environ["NQ_LIMIT"])]
if not indices:
    sys.exit("Nothing left to run.")

os.environ["RECRAG_RUN_NAMESPACE"] = "nq100"
# benchmark.run answers one question at a time; this only labels how many
# runs share the machine. Set NQ_PROCESSES when running both methods together.
os.environ["RECRAG_CONCURRENCY"] = os.environ.get("NQ_PROCESSES", "1")
# The warm-up question exists so the first question's time does not include
# start-up. Here every question opens a new index, so it would run 100 times,
# unmetered, with a news question. Latency is not reported for this run.
os.environ["RECRAG_WARMUP"] = "0"

import benchmark

print("questions to run:", len(indices), "first index", indices[0])
path = benchmark.run(
    method, "natural_questions", 0, len(indices), record_indices=indices
)
print("RESULTS", path)
