"""Write runs/final500/FROZEN.md: the frozen Agentic Hybrid RAG version, verbatim."""
import inspect, subprocess
from hashlib import sha256
from pathlib import Path
import benchmark, graphrag_retrieval as g
from evals import qa_metrics
from graphrag_construction import ONTOLOGY_EXTRACTION_PROMPT

def git(*a): return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
files = ["graphrag_retrieval.py", "graphrag_construction.py", "benchmark.py", "cost.py", "dataset.py",
         "rag_retrieval.py", "raptor_retrieval.py", "hipporag_retrieval.py", "evals/qa_metrics.py", "config.toml"]
source = inspect.getsource(g.build_agentic_retriever)
tools = source[source.index("tools=["):source.index("neo4j_database=database")]
fence = "````"
md = f"""# Frozen: Agentic Hybrid RAG final version (13 Nov 2026)

Frozen before any question of the final 500-question study ran. No change after that.

- Code commit (Agentic Hybrid RAG): `{git('rev-parse', '16cb0ff')}`
- Study plumbing commit (flex builds, question files, NQ shared corpus, judge code): `{git('rev-parse', 'b058b3d')}`
- Working tree code files at freeze (sha256), all equal to commit `b058b3d`:
""" + "\n".join(f"  - `{f}` `{sha256(Path(f).read_bytes()).hexdigest()}`" + ("" if not git('diff', '--stat', 'b058b3d', '--', f) else " (DIFFERS FROM COMMIT)") for f in files) + f"""
- Diffs against the previous commit: `frozen_agentic.diff`, `frozen_plumbing.diff`.

## What changed in this version (and why)

1. The agent picks the evidence. Every search result line carries a short ID (`E` + 6 hex characters of the sha256 of the result text). When it stops searching, the agent replies with the IDs the answer needs, at most {g.RETRIEVAL_TOP_K}, from either tool. Those items go to the answer model first, and leftover slots are filled by the old rule: vector passages by score, then graph rows. Before this, code kept the 10 highest-scoring vector passages, and graph rows reached the answer model in 0 of 300 comparison runs.
2. Graph rows carry graph facts only. `Chunk.text` is removed from the schema the Cypher writer sees. A generated query that returns `.text` is refused, and the agent sees the refusal. `text` and `embedding` keys are dropped from rows, and lists in a row keep at most {g.CYPHER_LIST_LIMIT} items, plus a "(+N more)" marker.

Evidence: 38 comparison questions that swung most in the 10 Nov repeats, 3 repeats each, Haiku judges (`runs/agentic_tune/graph_facts/compare.json`).

## Settings

- Answer and agent model: gpt-6-luna, reasoning effort medium (`config.toml` `[llm]`)
- Embeddings: text-embedding-3-small, 1536 dimensions
- Items to the answer model: {g.RETRIEVAL_TOP_K}; vector_search returns {g.RETRIEVAL_TOP_K} passages; Cypher rows at most {g.CYPHER_ROW_LIMIT}; entities per passage {g.ENTITY_LIMIT}; list items per graph row {g.CYPHER_LIST_LIMIT}
- Agent model calls per question: at most 5 (`ModelCallLimitMiddleware(run_limit=5)`)
- Chunks: {benchmark.CHUNK_SIZE} tokens, overlap {benchmark.CHUNK_OVERLAP}
- Ontology: news (MultiHop-RAG), biography (HotpotQA), general (Natural Questions)

## config.toml
{fence}toml
{Path('config.toml').read_text()}{fence}

## Agent instruction (verbatim)
{fence}text
{g.AGENT_INSTRUCTION}
{fence}

## Tools given to the agent (verbatim source)
{fence}python
{tools}{fence}

## Cypher writer prompt (verbatim template)
{fence}text
{g.CYPHER_PROMPT}{fence}

## Cypher examples (verbatim)
{fence}text
""" + "\n".join(g.CYPHER_EXAMPLES) + f"""
{fence}

## Answer prompts (verbatim)
System, MultiHop-RAG:
{fence}text
{benchmark.NEWS_QA_INSTRUCTION}
{fence}
System, HotpotQA and Natural Questions:
{fence}text
{benchmark.QA_INSTRUCTION}
{fence}
User message (`evals.qa_metrics.benchmark_prompt`):
{fence}python
{inspect.getsource(qa_metrics.benchmark_prompt)}{fence}

## Graph extraction prompt (construction, verbatim)
{fence}text
{ONTOLOGY_EXTRACTION_PROMPT}
{fence}
"""
Path("runs/final500/FROZEN.md").write_text(md)
print("ok", len(md))
