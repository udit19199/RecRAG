"""One answer shard of the graph ablation (runs/nograph/SPEC.md).

Usage (repo root): PYTHONPATH=. .venv/bin/python runs/nograph/scripts/answer.py REP SHARD NSHARDS [LIMIT]

Agentic Hybrid RAG with RECRAG_AGENT_NO_GRAPH=1 on the 250 MultiHop-RAG
comparison and temporal questions of runs/nograph/questions.json. Shard s of n
takes questions s, s+n, ... (LIMIT keeps only the first LIMIT, for a smoke test).
Same index checks as runs/final500/scripts/answer.py: never builds a graph.
Writes nograph.json into the run folder and prints "RESULTS <path>".
"""

import json
import os
import sys
import tomllib
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
rep, shard, shards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
os.environ["RECRAG_AGENT_NO_GRAPH"] = "1"
os.environ.pop("RECRAG_RUN_NAMESPACE", None)  # MultiHop-RAG indexes live under runs/<method>/indexes
os.environ["RECRAG_CONCURRENCY"] = str(shards)
os.environ.pop("RECRAG_BUILD_SERVICE_TIER", None)
indices = json.loads((ROOT / "runs/nograph/questions.json").read_text())["indices"][shard::shards]
if limit:
    indices = indices[:limit]

import benchmark  # noqa: E402
import graphrag_retrieval  # noqa: E402
from dataset import get_source  # noqa: E402

assert graphrag_retrieval.no_graph_agent()
settings = benchmark.Settings.model_validate(tomllib.loads((ROOT / "config.toml").read_text()))
pages = get_source("multihop_rag").load_record(indices[0]).pages
corpus_key = sha256("\n\n".join(benchmark.split_pages(pages)).encode()).hexdigest()
index_dir = ROOT / "runs/graphrag/indexes" / benchmark.index_key("graphrag", "multihop_rag", corpus_key, settings)  # type: ignore[arg-type]
if not (index_dir / "graph.complete").exists():
    raise SystemExit(f"No complete graph at {index_dir}; refusing to build.")


def refuse_build(*args, **kwargs):
    raise RuntimeError("Graph not found: refusing to build a new one.")


graphrag_retrieval.GraphRAG.construct = refuse_build  # type: ignore[method-assign]

path = benchmark.run("graphrag", "multihop_rag", 0, len(indices), record_indices=indices)
run_dir = Path(path).parent if Path(path).is_file() else Path(path)
(run_dir / "nograph.json").write_text(
    json.dumps({"RECRAG_AGENT_NO_GRAPH": "1", "rep": rep, "shard": shard, "shards": shards, "limit": limit}) + "\n"
)
print("RESULTS", path, flush=True)
