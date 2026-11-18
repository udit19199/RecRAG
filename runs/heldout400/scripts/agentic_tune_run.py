"""Run Agentic GraphRAG on a few held-out questions to check the 10 Nov fixes.

Usage: PYTHONPATH=. uv run python runs/heldout400/scripts/agentic_tune_run.py TYPE N [DONE_RETRIEVAL_JSONL]
Runs the first N questions of TYPE (e.g. comparison_query) from the frozen
held-out set, one at a time, skipping any already saved in DONE_RETRIEVAL_JSONL. Results go to runs/agentic_tune/graphrag/<timestamp>/. Refuses to
build a graph: if the existing graph is not found, it stops instead of paying
for a new build."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
kind, count = sys.argv[1], int(sys.argv[2])
by_type = json.loads((ROOT / "runs/heldout400/questions.json").read_text())["by_type"]
indices = by_type[kind][:count]
if len(sys.argv) > 3:
    done = {
        int(json.loads(line)["record_id"].rsplit("-", 1)[1])
        for line in Path(sys.argv[3]).open()
        if line.strip()
    }
    indices = [index for index in indices if index not in done]

os.environ["RECRAG_RUN_NAMESPACE"] = "agentic_tune"
# benchmark.run answers one question at a time; RECRAG_CONCURRENCY only labels
# how many runs share the machine at once. Set TUNE_PROCESSES when starting
# several of these scripts together, so the label stays true.
os.environ["RECRAG_CONCURRENCY"] = os.environ.get("TUNE_PROCESSES", "1")

import graphrag_retrieval  # noqa: E402


def refuse_build(*args, **kwargs):
    raise RuntimeError("Graph not found: refusing to build a new one.")


graphrag_retrieval.GraphRAG.construct = refuse_build

import benchmark  # noqa: E402

path = benchmark.run(
    "graphrag", "multihop_rag", 0, len(indices), record_indices=indices
)
print("RESULTS", path)
