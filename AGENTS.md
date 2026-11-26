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
scalable product. People are curious about findings, so run results in `runs/`
are committed, but indexes and caches under it are gitignored, and record
learnings where they can be found (docs, benchmark artifacts). Keep
architecture work simple and reversible.

## Experiment results: the Excalidraw scene is the source of truth

The **Experiments** scene in Excalidraw+ (HPE collection,
https://app.excalidraw.com/s/9pwrOc81bDT/5KJVp4wI32J) is the source of truth
for every experiment's results. Indexes and caches under `runs/` are local and
can be deleted; run results in `runs/` are committed.

- Every experiment you run gets its results added to that scene before the work
  counts as done: dataset, question count, repeats, the judges used, and the
  results reported as below.

### Reporting results

- Use these measures and these exact labels. Do not rename them or invent
  new measures or wording.
  - Claude Haiku is the only judge, for answers and facts.
  - **Answered right**: exact match, then the Claude Haiku judge.
  - Retrieval: **gold facts found** at three thresholds, **≥50%**, **≥75%**
    and **100%** of a question's gold facts. The facts are the claims the
    Claude Haiku judge extracts from the gold sentences (`evals/haiku_fact_judge.py`,
    `claims`). A fact counts as found only when `judge` marks it `stated` with a
    quote that appears in the retrieved text (`full_status`). Per-question data:
    latest run `facts[].found` in `runs/final500/judge/per_question.jsonl`. Previous
    run, also judged by Claude Haiku on the same frozen facts: counts in
    `runs/heldout400/gold_facts_by_run_haiku.json` (Agentic Hybrid RAG re-judged in
    `runs/heldout400/haiku_rejudge_agentic/`; baselines are final500's rows for the
    heldout400 folders). The old GPT-6 Luna counts (`gold_facts_by_run.json`,
    `judge_support/`) are kept for provenance only; never chart them.
  - Do not report gold-sentence recall or sentence-level matching; the gold
    facts replace them. Null questions have no gold facts, so they have no
    retrieval numbers.
- Always split by question type. Never report pooled or "overall" numbers.
- Show the average of the 3 repeats with a lowest-to-highest repeat line on the
  same bar; keep per-repeat numbers in the run data.
- The main comparison is Agentic Hybrid RAG vs Normal RAG on temporal and
  comparison questions: Normal RAG sits next to Agentic on every chart, so no
  separate confusion-matrix slide.
- No cost or latency slides: Udit removed them on 2026-11-25 because the runs'
  timings came from different days, machines and concurrency, so they did not
  compare fairly. Cost and latency stay in the run data
  (`runs/selection/paired/cost.json`).
- When you need a past result, look in the scene first, then in `runs/`.
- Before deleting run folders, check that their results are already in the
  scene.

### Findings from the latest run

Analyses live under `runs/final500/`, each a script plus a plain-words
`SUMMARY.md`. Read the summary before re-deriving a number, and add a new
analysis the same way:
- `two_run_analysis/`: previous vs latest run per question type, wrong
  "Insufficient Information" answers, latency, cost.
- `repeat_variation/`: why answers varied between repeats (no summary yet;
  numbers in its JSON).
- `cypher_failures/`: graph queries written by the model fail 7.4% of the time
  and come back empty another 24.5%; reach for it before blaming retrieval for a
  missing fact.
- `runs/agentic_tune/agent_picks/README.md`: the change that lets the agent pick
  its evidence, and why.

### Deck

The deck is the Experiments scene (`5KJVp4wI32J`), the only RecRAG scene. Each
slide is a frame named `NN Title`, laid out left to right in number order;
keep the number, canvas position and `slidesOrder` in step. Every slide is
3000 x 1688 (16:9): a taller slide shrinks to unreadable text when presented,
so split content across slides instead of growing one.

Content:
- Both runs share every chart and table: the previous run (held-out 400, 100
  questions per type) and the latest run (final500, 125 per type). Only
  HotpotQA and Natural Questions, which the latest run alone has, get
  latest-only charts.
- MultiHop-RAG appears per question type (comparison, temporal, inference,
  null), never as totals across types. Two types share a slide, one row of four
  charts each: comparison + temporal, inference + null.
- HotpotQA is shown as one dataset (bridge and comparison pooled), in a row
  above Natural Questions.
- Method order on every chart and table: Normal RAG, Agentic Hybrid RAG,
  RAPTOR, HippoRAG. Normal RAG sits beside Agentic Hybrid RAG, so that
  comparison needs no slide of its own.
- Each number lives on one slide. Fold new content into a related slide; the
  deck has no backup slides.
- Repeat spread sits on the result charts: a thin line from lowest to highest
  repeat, and the lowest and highest repeat written in small type under each
  average ("68.8–72.0") only when they are 2 or more points apart. There is
  no stability slide.
- Slides carry no takeaway subtitles, footnotes or commentary and no "What we
  tested" overview: a title, the setup legend and the charts. Takeaways and
  chart keys go in the frame's presenter notes.
- A new run joins the slides beside the older runs, whose results stay.
- Wording: "changes", not "fixes"; "previous run" and "latest run", never
  dates or a timeline. Plain words a non-specialist can follow; explain any
  technical term you keep. Exact prompts stay word for word.
- Order (2026-11-25): 01-02 how Agentic Hybrid RAG works, 03-04 its prompts,
  05 how answers are judged, 06-07 judge prompts. Then what was already
  discussed: 08 what changed between the runs, 09 question 747, 10 why the
  previous run's repeats were far apart (`runs/final500/repeat_variation/`),
  11 the 100 questions per type both runs answered and the 25 new ones (only
  the 100 compare the runs), 12 inference + null, 13 HotpotQA + Natural
  Questions, 14 HotpotQA + Natural Questions: more chunks. The main flow
  starts at 15 comparison + temporal, then 16 is it
  the graph (`runs/nograph/`), 17 is it the picking, 18 more chunks, 19 LLM
  picks vs top 50/100 question by question, 20 chunk order, 21 answer
  changes, 22 guessing, 23 made-up answers, 24 Neo4j vs Milvus, 25 what we
  found (the conclusion), 26 next phase.
- Method slides are for Agentic Hybrid RAG only; no architecture slides for
  Normal RAG, RAPTOR or HippoRAG. Each setup is explained once, in the legend
  of the first slide it appears on; Normal RAG and Milvus need no legend.
- "Agent picks, no graph rows" is not shown: Udit found it confusing
  (2026-11-25). Its data stays in `runs/nograph/`.
- Prompts are shown in Comic Shanns (`fontFamily` 8), as bands: grey same in
  both runs, blue added in the latest run, orange the previous run's replaced
  text. Keep the previous run's judge prompts: its numbers still use them.

Charts:
- Stacked, never split: a solid base to the lower run's average, its value just
  inside the top; a hatched cap to the higher run's value, its value above. On
  a tie, the previous run is the base and there is no cap.
- Previous run orange (`#ffd8a8` fill, `#e8590c` stroke); latest run blue
  (`#a5d8ff`, `#1971c2`). Square corners, no arrows, no "45 → 46" labels.
- No gridlines or tick labels; every bar shows its figure.
- Tables: one number per cell, one table per measure.
- The deck is presented in dark mode: check contrast in the screenshots.
- Text: every label centred horizontally and vertically (use the label of an
  invisible box). Break lines on purpose: no orphan words, no accidental third
  line, no overlap with other elements.

Helpers in `~/Workspace/local/hpe-diagrams/` (outside the repo):
- `deckops.py`: MCP client (`call`, `send`), `scene()` (live elements),
  `moves()` (moves elements, skipping bound text), `place(order)` (lays frames
  out left to right and renames them), `restore_offsets(backup)` (puts
  elements back at their in-frame offsets from a backup).
- `results_v2.py types|other|changed|q747`: rebuilds slides 08, 12-13 (two types
  per slide; HotpotQA pooled) and 15-16 from run data. Its `order` predates
  slides 09-10; renumber with `nograph_slides.py order` instead.
  `nograph_slides.py s09|s10` builds slides 09-10. It reuses
  `chart2.py` (the reference chart style: `results_slides.py`'s `chart()` plus
  the repeat range labels) and `results_slides.py`'s `reset(fid)`.
- `prompts_pack.py dry|build`: the prompt slides (03-04, 06-07). Prompts are
  packed at size 26 into as few slides as fit, each under a numbered heading
  that says what the prompt is for. Prompt text comes from `split_slides.py`
  (`PROMPT_SLIDES`); its `check` confirms it still matches the sources.
- `fill_slides.py s01|s06|s16`: slide 01's previous-run row, the judging rows
  on 05 and the next-phase table. `repeat_slide.py build`: slide 11.
- Fill each slide: no large empty band at the bottom or beside a column.
- Text slides: headings do the work, one short line per point, no filled
  coloured cards (neutral greys; orange/blue only as thin run markers), each
  fact once in the deck; detail goes in the frame's presenter notes.
- `declutter_a.py s02|s05|s09` and `declutter_13_14.py` build slides 02, 05,
  11, 15 and 16 in that style (script names keep the old numbers).
- `newslides.py` (`stacked_chart`, `new_frame`, `delete`, `by_name`) and
  `two_run_slides.py cost`: other builders. `two_run_slides.py` uses the right
  method order; `newslides.py` still has Agentic last, and its `stab` builds the
  retired stability slide; fix the order before reusing it.
- `method_slides.py`: prompt sources (`split_agent`, `cypher_bands`, ...),
  drawing helpers (`S`, `table`) and `clear <frame prefix>`. Its `s1|s2|s3|s10`
  build the old tall slides; do not use them.
- `xmcp.py` reads the CLI's saved login: an API key or, since CLI v1, an OAuth
  access token that expires after about an hour; run `excalidraw whoami` first
  to refresh it.

Excalidraw MCP quirks:
1. Moving a bound text makes the server re-derive its container's position.
   Move boxes only; bound text follows.
2. Left/right- or top-aligned labels become grouped text with no `frameId`.
   Prefer centred labels or plain text.
3. Changing text that has `autoResize: false` needs `autoResize: true` plus a
   width.
4. `take_screenshot` needs `maxWidth` ≤ 1920.
5. Multi-line or top-aligned labels can lose their frameId (and survive a frame reset). Use `deckops.unframed_in` when clearing a frame and `deckops.adopt(fid)` after adding labels.

Steps for any deck change:
1. Back up: `excalidraw scenes content get 5KJVp4wI32J --out
   ~/Workspace/local/hpe-backups/Experiments-<what>-<timestamp>.json`.
2. In that backup, note frames with `isDeleted: true`: the user deleted those
   slides, and they stay deleted.
3. Rebuild each changed slide from a script (`reset`, then draw), not by
   patching elements by hand.
4. After any layout move, run `restore_offsets` against the backup, and fix
   slide-number references after a reorder.
5. Screenshot every changed slide and look at it.
6. One Opus review subagent checks the numbers and layout. Done when it passes
   and the screenshots are clean; only then show the user.

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
evaluation helpers live in `evals/`. Save experiment results under `runs/` and
commit them; indexes and caches there stay gitignored.

The agent ends by listing the tags of the results the answer needs; code sends
those first, then the best-scoring passages, 10 items in all (the app sends 5).
Graph queries return facts only, never article text.

## Older datasets (archived)

Hannon, REFinD, and FiNER-139 dataset files were deleted; `datasets/` now holds
only `hotpotqa` and `multihop_rag`; Natural Questions and NovelQA load from the
Hugging Face cache. The REFinD and 2WikiMultiHopQA write-ups were removed from
`docs/` in the same commit as this line; read them from git history if needed.

## Agent Checklist (before finishing)

- types are updated where contracts changed
- Agentic Hybrid RAG result shapes still match
- meaningful changes document **why** (comment or commit message)
- smallest relevant verification commands have run
- local-only files were not staged by accident
- do not dump file names, paths, or raw internal info in user-facing output — explain findings in plain words and keep the detail focused

## Decision References

- See `README.md` for the current package layout and verification commands.
