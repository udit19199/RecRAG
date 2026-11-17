# AGENTS.md - RecRAG Operating Guide

This file is the primary guide for agentic coding work in this repo. Keep it practical: only facts an agent would miss or guess wrong.

## Project Context

RecRAG is a research project between **Hewlett Packard Enterprise (HPE)** and
**JECRC University**. It compares retrieval-augmented generation approaches.
The FiNER-139 extraction-method work was removed, and its runs and datasets
were deleted. The August result (Ontology 49.0% vs Hybrid 36.4% strict F1,
n=200, 328 gold spans, +12.6 F1, CI 10.9–14.4) survives only in the archived
August decks and the 2026-07-23 FiNER-139 experiment map, both archived on
2026-11-10 under `~/Google Drive/My Drive/07_Archive/HPE/`, and in Udit's
memory. FiNER-139 tests extraction, not GraphRAG; do not present it as a
benchmark.

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

## Experiment results: the Excalidraw scene is the source of truth

The **Experiments** scene in Excalidraw+ (HPE collection,
https://app.excalidraw.com/s/9pwrOc81bDT/5KJVp4wI32J) is the source of truth
for every experiment's results. `runs/` and `results/` are local, gitignored
and can be deleted, so a result that only lives there can be lost.

- Every experiment you run gets its results added to that scene before the work
  counts as done: dataset, question count, repeats, the judges used, and the
  results reported as below.

### Reporting results

- Use these measures and these exact labels. Do not rename them or invent
  new measures or wording.
  - **Answered right**: exact match, then the Claude Haiku judge.
  - Retrieval: **gold facts found** at three thresholds, **≥50%**, **≥75%**
    and **100%** of a question's gold facts. The facts are the claims the
    Claude Haiku judge extracts from the gold sentences (`evals/haiku_fact_judge.py`,
    `claims`). A fact counts as found only when `judge` marks it `stated` with a
    quote that appears in the retrieved text (`full_status`). Per-question data:
    `gold[].claims[].full_status` in `runs/heldout400/judge_support/*/per_question.jsonl`.
  - Do not report gold-sentence recall or sentence-level matching; the gold
    facts replace them. Null questions have no gold facts, so they have no
    retrieval numbers.
- Always split by question type. Never report pooled or "overall" numbers.
- Report each of the 3 repeats separately, and say which questions fail in all
  3 repeats.
- The main comparison is Agentic Hybrid RAG vs Normal RAG on temporal and
  comparison questions, shown as confusion matrices (both, Agentic only,
  Normal RAG only, neither) per repeat.
- No cost or latency in results.
- When you need a past result, look in the scene first, then in local `runs/`.
- Before deleting local run folders, check that their results are already in
  the scene.

### Deck layout

- **8 slides, never more than 13.** Do not add a slide for a new run; add the
  run to the existing slides. If something new must go in, merge or drop a
  slide first. Do not repeat a number or a point on two slides.
- The current eight: 1 run map, 2 methods and prompts, 3 comparison, 4
  temporal, 5 inference and null (HotpotQA is in the run map only), 6 found facts and
  right answers, 7 using the results and likely questions,
  8 backup. Each type slide is tall: dataset facts, then all 4 methods, then
  Agentic Hybrid RAG vs Normal RAG.
- Tables hold one number per cell, one table per measure. Never pack several
  numbers into a cell with slashes; use more space or another slide instead.
- Results are organised by MultiHop-RAG's four question types (comparison,
  temporal, inference, null), at most two types per slide. Each type's block
  carries its own dataset facts: the questions per run and in the dataset, one
  example question with its gold answer, and the answer formats. Then comes one
  table per run that has the type. There is no separate dataset slide. Label
  other datasets' questions with these four types; say so where a type has few
  or no questions.
- Slide 1 is the run map: one row per run with the data (collection size and
  questions), repeats, how it was scored, status and the slides that show it.
  Every new run gets a row.
- Slides carry no dates or timeline. Name runs as the previous run and the
  latest run, and describe what changed, not when.
- A superseded run keeps its row in the run map, marked older, and its results
  leave the other slides.
- One backup slide at the end holds the scoring rules and every judge prompt
  word for word. Update it when a judge or prompt changes.
- Keep the frame name number, the canvas position (left to right) and the
  presentation order (`slidesOrder`) in step, and fix slide-number references
  (run map, "see slide N") after any reorder.
- Before changing the scene, export a backup of it.

## Setup

There is no Makefile. Run from repository root:

- `uv sync` — install Python dependencies
- `cp .env.example .env` — create `.env`
- Start Neo4j Desktop Enterprise with APOC. For each HotpotQA run, the app
  creates one database per construction method from the shared validation
  corpus. Other datasets use one database per record and construction method.
- `uv run streamlit run streamlit_app.py` — run the app against Desktop.

## Quality Checks

No tests are allowed in this repo. Do not add test files, test suites, or test
frameworks. Use direct inspection for verification.

## Key Service Ports

| Store | Port | Use |
|-------|------|-----|
| Neo4j | `:7687` | Graph storage |

## Critical Constraints

- **Do not move API keys into `config.toml`** — `OPENAI_API_KEY` and `LLAMA_CLOUD_API_KEY` live in `.env`. Pass LangChain OpenAI chat and embedding models directly; `config.toml` `[llm]`/`[embedding]` `model` is a plain OpenAI model id (for chat, use `gpt-6-luna`). There is no provider switch. Future PDF approaches must use LlamaCloud's LlamaParse exclusively.
- **Only `gpt-6-luna` may be used as the GPT model** — use it for extraction, retrieval, answers, fallbacks, CLI flags, and benchmark scripts, with the configured `medium` reasoning effort. Do not use any other GPT model. Use the Responses API for every OpenAI generation call.
- Adapter timeouts come from `config.toml`, not hardcoded.
- Dicts, `TypedDict`s, and tuples are strongly prohibited.
- Retries use `urllib3.util.Retry` with idempotent-method-only.
- Agentic Hybrid RAG writes chunk embeddings and the graph to Neo4j.
- Neo4j vector retrieval uses `neo4j-graphrag`'s `VectorCypherRetriever`.
  Agentic retrieval uses Neo4j vector and text-to-Cypher tools.
- Agentic Hybrid RAG uses the official `neo4j-graphrag` package for the retrieve-to-answer
  flow and passes its `GraphRAGChatLLM` adapter around the configured LangChain
  chat model to that package.

## Agentic Hybrid RAG (active)

Call this method **Agentic Hybrid RAG** in docs, slides, tables and messages —
never "GraphRAG" alone or "Hybrid RAG". The method id `graphrag`, the
`graphrag_*.py` modules and the `GraphRAG` class keep their code names because
saved runs and imports depend on them.

The active Agentic Hybrid RAG path uses ontology-guided construction (`ontology_guided`)
and agentic retrieval (`agentic`). The ontology suggests top-level node and
relationship types and allows additional types. Vector and text-to-Cypher
search remain internal tools for agentic retrieval. Construction writes chunk
vectors to Neo4j; agentic retrieval searches Neo4j.
Benchmark scoring is optional and sits outside the normal Agentic Hybrid RAG path.

HippoRAG installs from a pinned upstream Git commit into `.venv-hipporag`; its
source files are not in this working tree. `hipporag_retrieval.py` runs its
native retrieval and `rag_retrieval.py` runs dense Normal RAG. Both share the
benchmark loop in `benchmark.py` and dataset adapters in `dataset.py`.
Keep using the separate environment because HippoRAG's dependency pins conflict
with the main Agentic Hybrid RAG environment. See `README.md` for the install command.

Use `GraphRAG` from `graphrag_retrieval.py` directly from Python.
`graphrag_construction.py` holds the ontology and builds the graph.
`graphrag_retrieval.py` provides the agentic retrieval tools and runs retrieval
and answering. Dataset inputs shared by all methods live in `dataset.py`;
evaluation helpers live in `evals/`. Save experiment results under local-only
`runs/` (gitignored).

## Older datasets (archived)

Hannon, REFinD, and FiNER-139 dataset files were deleted; `datasets/` now holds
only `hotpotqa` and `multihop_rag`. They are not part of the active Agentic
GraphRAG benchmark path.

## Agent Checklist (before finishing)

- types are updated where contracts changed
- Agentic Hybrid RAG result shapes still match
- meaningful changes document **why** (comment or commit message)
- smallest relevant verification commands have run
- local-only files were not staged by accident
- do not dump file names, paths, or raw internal info in user-facing output — explain findings in plain words and keep the detail focused

## Decision References

- See `README.md` for the current package layout and verification commands.
