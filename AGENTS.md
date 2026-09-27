# AGENTS.md - RecRAG Operating Guide

This file is the primary guide for agentic coding work in this repo. Keep it practical: only facts an agent would miss or guess wrong.

## Project Context

RecRAG is a research project between **Hewlett Packard Enterprise (HPE)** and
**JECRC University**. It compares retrieval-augmented generation approaches.
The FiNER-139 extraction-method benchmark implementation was removed after its
latest result was preserved under `runs/finer139/`. Its dataset files remain
under `datasets/finer139/` as archived inputs; do not use them in
GraphRAG work unless explicitly asked.

RecRAG previously included a candidate-pipeline recommendation/benchmark
product (`app/orchestrator/`, `src/orchestration/`, `docs/recommendation.md`).
That product and its orchestrator service have been cut as part of an ongoing
remodel — do not resurrect that pattern (per-candidate indexing, benchmark
suites, blueprint export) without confirming it's back in scope.

This is **research-first**: the value is in what we learn and document, not a
scalable product. People are curious about findings, so keep experiment runs
and results local-only (`runs/` and `results/` are gitignored), and record
learnings where they can be found (docs, benchmark artifacts). Keep
architecture work simple and reversible.

## Setup

There is no Makefile. Run from repository root:

- `uv sync --extra experiments` — install Python and research dependencies
- `cp .env.example .env` — create `.env`
- Start Neo4j Desktop Enterprise with APOC. The app creates one database per
  record and construction method.
- `uv run streamlit run streamlit_app.py` — run the app against Desktop.

## Quality Checks

No tests are allowed in this repo. Do not add test files, test suites, or test
frameworks. Use direct inspection for verification.
Run `uv run ruff format --check .` and `uv run ruff check .` for Python checks.

## Key Service Ports

| Store | Port | Use |
|-------|------|-----|
| Neo4j | `:7687` | Graph storage |

## Critical Constraints

- **Do not move API keys into `config.toml`** — `OPENAI_API_KEY` and `LLAMA_CLOUD_API_KEY` live in `.env`. Pass LangChain OpenAI chat and embedding models directly; `config.toml` `[llm]`/`[embedding]` `model` is a plain OpenAI model id (for chat, use `gpt-6-luna`). There is no provider switch. Future PDF approaches must use LlamaCloud's LlamaParse exclusively.
- **Only `gpt-6-luna` may be used as the GPT model** — use it for extraction, retrieval, answers, evaluation judges, fallbacks, CLI flags, and benchmark scripts, with the configured `medium` reasoning effort. Do not use any other GPT model. Use the Responses API for every OpenAI generation call.
- Adapter timeouts come from `config.toml`, not hardcoded.
- Dicts, `TypedDict`s, and tuples are strongly prohibited.
- Retries use `urllib3.util.Retry` with idempotent-method-only.
- GraphRAG writes chunk embeddings and the graph to Neo4j.
- Neo4j vector retrieval uses `neo4j-graphrag`'s `VectorCypherRetriever`.
  Agentic retrieval uses Neo4j vector and text-to-Cypher tools.
- GraphRAG uses the official `neo4j-graphrag` package for the retrieve-to-answer
  flow and passes its `GraphRAGChatLLM` adapter around the configured LangChain
  chat model to that package.

## GraphRAG (active)

The active GraphRAG path uses ontology-guided construction (`ontology_guided`)
and agentic retrieval (`agentic`). The ontology suggests top-level node and
relationship types and allows additional types. Vector and text-to-Cypher
search remain internal tools for agentic retrieval. Construction writes chunk
vectors to Neo4j; agentic retrieval searches Neo4j.
Benchmark scoring is optional and sits outside the normal GraphRAG path.

HippoRAG is an independent upstream checkout under `hipporag/`, ignored by this
repository. `baselines/hipporag.py` runs its native retrieval and
`baselines/rag.py` runs dense Normal RAG. Both share the benchmark loop in
`baselines/runner.py` and dataset adapters in `dataset.py`. Use
`.venv-hipporag`, because upstream dependency pins conflict with Neo4j GraphRAG.
Keep upstream code unchanged.

Use `GraphRAG` from `graphrag/graph_rag.py` directly from Python.
`graphrag/construction.py` builds the graph using `graphrag/ontology.py`.
`graphrag/retrieval.py` provides the agentic retrieval tools and runs retrieval
and answering. Dataset inputs shared by all methods live in `dataset.py`;
evaluation helpers live in `evals/`. Save experiment results under local-only
`runs/` (gitignored).

## Older datasets (archived)

Hannon, REFinD, and FiNER-139 files remain for historical runs only. They are
not part of the active GraphRAG benchmark path.

## Agent Checklist (before finishing)

- types are updated where contracts changed
- GraphRAG result shapes still match
- meaningful changes document **why** (comment or commit message)
- smallest relevant verification commands have run
- local-only files were not staged by accident
- do not dump file names, paths, or raw internal info in user-facing output — explain findings in plain words and keep the detail focused

## Decision References

- See `README.md` for the current package layout and verification commands.
