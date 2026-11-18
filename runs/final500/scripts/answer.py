"""One answer shard of the final 500-question study, on already-built indexes.

Usage (repo root; HippoRAG with .venv-hipporag/bin/python):
  PYTHONPATH=. python runs/final500/scripts/answer.py DATASET METHOD REP SHARD NSHARDS

Questions come from runs/final500/questions.json:
  multihop_rag: Agentic Hybrid RAG answers all 500; Normal RAG, RAPTOR and
    HippoRAG answer the 100 new questions (their 400 are reused from
    runs/heldout400).
  hotpotqa, natural_questions: all 500.
Shard s of n takes questions s, s+n, s+2n, ... of that list. The run refuses
to start if the method's index for this corpus does not exist, and Agentic
GraphRAG refuses to build a graph, so nothing is ever rebuilt by an answer run.
Prints "RESULTS <path>"; the driver links the run folder under
runs/final500/DATASET/METHOD/.
"""

import json
import os
import sys
import tomllib
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
dataset, method, rep = sys.argv[1], sys.argv[2], sys.argv[3]
shard, shards = int(sys.argv[4]), int(sys.argv[5])
questions = json.loads((ROOT / "runs/final500/questions.json").read_text())
if dataset == "multihop_rag":
    # The existing MultiHop-RAG indexes live under runs/<method>/indexes.
    os.environ.pop("RECRAG_RUN_NAMESPACE", None)
    q = questions["multihop_rag"]
    indices = q["all500"] if method == "graphrag" else q["new100"]
elif dataset == "hotpotqa":
    os.environ["RECRAG_RUN_NAMESPACE"] = "final500/hotpotqa"
    os.environ["RECRAG_HOTPOTQA_ROWS"] = str(
        ROOT / "runs/final500/hotpotqa/questions_rows.json"
    )
    indices = questions["hotpotqa"]["record_indices"]
elif dataset == "natural_questions":
    os.environ["RECRAG_RUN_NAMESPACE"] = "final500/natural_questions"
    os.environ["RECRAG_NQ_CORPUS_ROWS"] = "0-499"
    os.environ["RECRAG_NQ_RECORDS"] = str(
        ROOT / "runs/final500/natural_questions/records.json"
    )
    indices = questions["natural_questions"]["record_indices"]
else:
    raise SystemExit(f"unknown dataset {dataset}")
indices = indices[shard::shards]
os.environ["RECRAG_CONCURRENCY"] = str(shards)
# Build calls never happen here; the tier setting would only apply to builds.
os.environ.pop("RECRAG_BUILD_SERVICE_TIER", None)

import benchmark  # noqa: E402
from dataset import (  # noqa: E402
    get_source,
    load_hotpotqa_corpus,
    load_natural_questions_corpus,
    natural_questions_corpus_rows,
)

settings = benchmark.Settings.model_validate(
    tomllib.loads((ROOT / "config.toml").read_text())
)
if dataset == "hotpotqa":
    pages = load_hotpotqa_corpus()
elif dataset == "natural_questions":
    pages = load_natural_questions_corpus(natural_questions_corpus_rows())
else:
    pages = get_source(dataset).load_record(indices[0]).pages
corpus_key = sha256("\n\n".join(benchmark.split_pages(pages)).encode()).hexdigest()
namespace = os.environ.get("RECRAG_RUN_NAMESPACE")
run_root = ROOT / "runs" / namespace if namespace else ROOT / "runs"
index_dir = (
    run_root
    / method
    / "indexes"
    / benchmark.index_key(method, dataset, corpus_key, settings)  # type: ignore[arg-type]
)
marker = {
    "graphrag": "graph.complete",
    "hipporag": "index.complete",
}.get(method)
if not index_dir.is_dir() or not any(index_dir.iterdir()):
    raise SystemExit(f"No built index at {index_dir}; refusing to build one.")
if marker and not (index_dir / marker).exists():
    raise SystemExit(f"Index at {index_dir} has no {marker}; refusing to build.")
if method == "graphrag":
    import graphrag_retrieval

    def refuse_build(*args, **kwargs):
        raise RuntimeError("Graph not found: refusing to build a new one.")

    graphrag_retrieval.GraphRAG.construct = refuse_build  # type: ignore[method-assign]

path = benchmark.run(
    method,  # type: ignore[arg-type]
    dataset,
    0,
    len(indices),
    record_indices=indices,
)
print("RESULTS", path, flush=True)
