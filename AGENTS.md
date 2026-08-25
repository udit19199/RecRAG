# AGENTS.md - RecRAG Operating Guide

This file is the primary guide for agentic coding work in this repo. Keep it practical: only facts an agent would miss or guess wrong.

## Project Context

RecRAG is a research project between **Hewlett Packard Enterprise (HPE)** and
**JECRC University**. It compares retrieval-augmented generation approaches.
The FiNER-139 extraction-method benchmark implementation was removed after its
latest result was preserved under `runs/finer139/`. Its dataset files remain
under `datasets/finer139/` as archived inputs; do not use them in
HybridRAG work unless explicitly asked.

RecRAG previously included a candidate-pipeline recommendation/benchmark
product (`app/orchestrator/`, `src/orchestration/`, `docs/recommendation.md`).
That product and its orchestrator service have been cut as part of an ongoing
remodel — do not resurrect that pattern (per-candidate indexing, benchmark
suites, blueprint export) without confirming it's back in scope.

This is **research-first**: the value is in what we learn and document, not a
scalable product. People are curious about findings, so preserve experiment runs
and results, and record learnings where they can be found (docs, benchmark
artifacts). Keep architecture work simple and reversible.

## Setup

There is no Makefile. Run from repository root:

- `uv sync --extra experiments` — install Python and research dependencies
- `cp .env.example .env` — create `.env`
- Start Neo4j Desktop Enterprise with APOC and the database named in `.env`.
- `uv run streamlit run streamlit_app.py` — run the app against Desktop.

## Quality Checks

No tests are allowed in this repo. Do not add test files, test suites, or test
frameworks. Use direct inspection for verification.
This repo does not need lint or formatting checks.

## Key Service Ports

| Store | Port | Use |
|-------|------|-----|
| Neo4j | `:7687` | Graph storage |

## Critical Constraints

- **Do not move API keys into `config.toml`** — `OPENAI_API_KEY` and `LLAMA_CLOUD_API_KEY` live in `.env`. Pass LangChain OpenAI chat and embedding models directly; `config.toml` `[llm]`/`[embedding]` `model` is a plain OpenAI model id (e.g. `gpt-5.6-terra`). There is no provider switch. Future PDF approaches must use LlamaCloud's LlamaParse exclusively.
- **Default chat model is `gpt-5.6-terra`** — for any LLM-related work (config defaults, adapter fallbacks, CLI flags, benchmark scripts), use `gpt-5.6-terra` with the configured `xhigh` reasoning effort. Use the Responses API for every OpenAI generation call.
- Adapter timeouts come from `config.toml`, not hardcoded.
- Dicts, `TypedDict`s, and tuples are strongly prohibited.
- Retries use `urllib3.util.Retry` with idempotent-method-only.
- HybridRAG uses LangChain's `Neo4jVector`, so its record vectors and graph stay in Neo4j.
- Neo4j is the only store. Use LangChain's Neo4j integration directly.
  Do not hand-roll database clients when those integrations support the need.
- HybridRAG uses the official `neo4j-graphrag` package for the retrieve-to-answer
  flow and passes its LangChain chat model directly to that package.

## HybridRAG (active)

The active HybridRAG module supports **2 graph-construction approaches** and
**4 retrieval approaches**. Both construction methods use the same input,
Neo4j database, chunk indexes, retrieval, and answering flow. Only extraction
or retrieval behavior differs. Benchmark scoring is optional and sits outside
the normal HybridRAG path.

| Construction approach | Method ID | Notes |
|-----------------------|-----------|-------|
| Standard | `standard` | Open-ended LLM extraction |
| Ontology-guided | `ontology_guided` | Fixed top-level node and relationship types, with additional types allowed |

| Retrieval approach | Method ID | Notes |
|-------------------|-----------|-------|
| Text to Cypher | `text2cypher` | Generates and runs a read-only Cypher query |
| Agentic | `agentic` | Chooses between vector and Cypher search tools |
| Vector | `vector` | Vector search with graph context |
| Hybrid | `hybrid` | Vector and full-text search with graph context |

The deep HybridRAG interface is `hybridrag/hybrid_rag.py`. Construction code
lives in `hybridrag/construction/`, retrieval code in
`hybridrag/retrieval/`. Shared models, validation, Neo4j storage, and saved-run
handling live directly under `hybridrag/`. There is no separate research layer;
the entire package exists for research and experimentation.
Use `HybridRAG` directly from Python. Construction execution lives in
`hybridrag/construction/construction.py`; retrieval and answering execution
lives in `hybridrag/retrieval/answering.py`. Evaluation helpers live in
`hybridrag/evals/`. Preserve saved experiment results under `runs/`.

## Older datasets (archived)

Hannon, REFinD, and FiNER-139 files remain for historical runs only. They are
not part of the active HybridRAG benchmark path.

## Agent Checklist (before finishing)

- types are updated where contracts changed
- HybridRAG result shapes still match
- meaningful changes document **why** (comment or commit message)
- smallest relevant verification commands have run
- local-only files were not staged by accident
- do not dump file names, paths, or raw internal info in user-facing output — explain findings in plain words and keep the detail focused

## Decision References

- See `README.md` for the current package layout and verification commands.
