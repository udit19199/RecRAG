# FiNER-139 Graph-Construction Entity Recognition

Research finding for action item #2 from the GraphRAG analysis: compare the five
graph-construction methods on how well they **recognise entities** in the
FiNER-139 dataset.

## Dataset

- **Source:** [nlpaueb/finer-139](https://huggingface.co/datasets/nlpaueb/finer-139)
- **Split used:** `validation` (validation role in the analysis doc)
- **Task:** IOB2 token classification where gold "entities" are **numeric tokens**
  tagged with one of **139 XBRL concept types** (e.g.
  `DebtInstrumentFaceAmount`). The correct tag depends on context, not the token
  alone.

## Fairness protocol

1. **Type-agnostic detection** — we score whether each method finds the correct
   entity *spans*, not the 139-way XBRL type.
2. **Numeric-only evaluation universe** — before matching, every method's
   predictions are filtered to numeric expressions. FiNER gold is numeric-only,
   so recall is unchanged; precision becomes comparable across open-ended methods
   (LLM/spaCy) that would otherwise extract non-numeric entities FiNER never
   annotates.
3. **Each method in its ideal config** (see below). Dynamic runs as a
   memory-augmented incremental extractor (its intended operating mode).

## Methods (ideal configs)

| Method | Implementation | Notes |
|--------|----------------|-------|
| **LLM-Based** | Open-ended prompt → JSON entity list → span alignment | Defaults to OpenAI `gpt-4o-mini` (override in tab) |
| **NLP / OpenIE** | spaCy `en_core_web_sm`, labels MONEY/PERCENT/CARDINAL/QUANTITY/DATE/ORDINAL | Deterministic, offline |
| **Ontology / Schema** | XBRL keyword gazetteer (CamelCase-split concept names) + numeric regex when keywords co-occur in sentence | Deterministic |
| **Hybrid** | LLM prompted with full 139 concept list + numeric post-filter | Schema-guided LLM |
| **Dynamic / Incremental** | Memory-augmented wrapper over Hybrid; learns context words from base hits and flags nearby numerics | Reuses Hybrid LLM calls when both selected |

## Metrics

- Micro **precision**, **recall**, **F1** over the sample
- **Strict** match: exact token span
- **Relaxed** match: any token overlap
- Per-method **latency** and **LLM call count**

## How to run

1. Install optional deps (orchestrator only pulls them when a run starts):

   ```bash
   uv sync --extra experiments
   python -m spacy download en_core_web_sm
   ```

2. Set an LLM key in `.env` for LLM/Hybrid/Dynamic (e.g. `OPENAI_API_KEY`).

3. Start the stack (`make dev` or `make orchestrator` + `make frontend`).

4. Open the **FiNER-139** tab, configure sample size / methods / LLM, click
   **Run benchmark**. Results appear when the background job completes (poll
   every ~1.5s).

## Architecture

- **Engine:** `src/experiments/finer139/` (pure functions, lazy heavy imports)
- **API:** `POST/GET /experiments/finer139/runs` on orchestrator `:8002`
- **UI:** `frontend/app/(main)/finer139/` + `features/finer139/`
- **Persistence:** in-memory only (results lost on orchestrator restart)

## Related work

- **Action item #1 (retrieval):** [graphrag-retrieval-patterns.md](./graphrag-retrieval-patterns.md) — hybrid, community graph, PathRAG vs neighborhood
- **Action item #2 (ranking):** [finer139-method-ranking.md](./finer139-method-ranking.md) — which construction method wins on FiNER-139
- Index: [README.md](./README.md)

## Caveats

- Results measure **numeric financial entity detection**, not general NER or
  faithful 139-way XBRL typing.
- First run downloads the HF dataset (~103 MB).
- Dynamic's incremental memory is an interpretation for a per-sentence detection
  benchmark; its production value is graph maintenance, not raw detection F1.
- Research-only — archive before prod per `AGENTS.md`; not wired into
  recommendation/orchestration product paths.

## E2E verification (2026-07-04)

Offline smoke (`nlp` + `ontology`, sample_size=100, seed=42) via live tab:

| Method | P (strict) | R (strict) | F1 (strict) | F1 (relaxed) | Latency |
|--------|------------|------------|-------------|--------------|---------|
| NLP / OpenIE (spaCy) | 7.1% | 15.6% | 9.8% | 55.3% | 0.9s |
| Ontology / Schema-Driven | 32.0% | 91.9% | 47.5% | 47.5% | 0.0s |

Screenshots (see `screenshots/` in this folder):

- `01-finer139-config.png` — tab config panel
- `02-finer139-results.png` — results table after live run
- `03-finer139-examples.png` — gold vs predicted span examples
- `04-finer139-nav.png` — sidebar with FiNER-139 nav item

Capture locally: `node scripts/capture-finer139-screenshots.mjs` (requires orchestrator + frontend running).

## Results

Live benchmark on validation sample (n=100, seed=42, 160 gold entities) — 2026-07-06:

| Method | P (strict) | R (strict) | F1 (strict) | F1 (relaxed) | Latency |
|--------|------------|------------|-------------|--------------|---------|
| Ontology / Schema-Driven | 32.0% | 91.9% | **47.5%** | 47.5% | 0.0s |
| NLP / OpenIE (spaCy) | 7.1% | 15.6% | 9.8% | 55.3% | 0.9s |
| LLM-Based | — | — | — | — | requires `OPENAI_API_KEY` |
| Hybrid | — | — | — | — | requires `OPENAI_API_KEY` |
| Dynamic | — | — | — | — | requires `OPENAI_API_KEY` |

Stakeholder summary: [finer139-showcase-results.md](./finer139-showcase-results.md). Raw JSON: `finer139-e2e-offline.json`, `finer139-e2e-full.json`.
