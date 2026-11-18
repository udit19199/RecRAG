"""Run Agentic GraphRAG (13 Nov agent-picks change) on the swinging questions.

Usage: PYTHONPATH=. uv run python runs/agentic_tune/agent_picks/run.py REP [LIMIT]
Writes a timestamped folder under runs/agentic_tune/graphrag/ and symlinks it
as runs/agentic_tune/agent_picks/repREP. Refuses to build a graph.
"""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
rep = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
ids = json.loads((HERE / "questions.json").read_text())["record_ids"][:limit]
indices = [int(q.rsplit("-", 1)[1]) for q in ids]
os.environ["RECRAG_RUN_NAMESPACE"] = "agentic_tune"
os.environ["RECRAG_CONCURRENCY"] = os.environ.get("TUNE_PROCESSES", "1")
import graphrag_retrieval  # noqa: E402
def refuse_build(*a, **k):
    raise RuntimeError("Graph not found: refusing to build a new one.")
graphrag_retrieval.GraphRAG.construct = refuse_build
import benchmark  # noqa: E402
path = benchmark.run("graphrag", "multihop_rag", 0, len(indices), record_indices=indices)
link = HERE / f"rep{rep}"
if link.is_symlink(): link.unlink()
link.symlink_to(Path(path).resolve().parent)
print("RESULTS", path)
