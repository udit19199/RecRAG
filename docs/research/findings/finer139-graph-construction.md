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
| **LLM-Based** | Open-ended prompt → JSON entity list → span alignment | Uses configured LLM (`config.toml` or tab override) |
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

2. Set an LLM key in `.env` for LLM/Hybrid/Dynamic (e.g. `GEMINI_API_KEY`).

3. Start the stack (`make dev` or `make orchestrator` + `make frontend`).

4. Open the **FiNER-139** tab, configure sample size / methods / LLM, click
   **Run benchmark**. Results appear when the background job completes (poll
   every ~1.5s).

## Architecture

- **Engine:** `src/experiments/finer139/` (pure functions, lazy heavy imports)
- **API:** `POST/GET /experiments/finer139/runs` on orchestrator `:8002`
- **UI:** `frontend/app/(main)/finer139/` + `features/finer139/`
- **Persistence:** in-memory only (results lost on orchestrator restart)

## Caveats

- Results measure **numeric financial entity detection**, not general NER or
  faithful 139-way XBRL typing.
- First run downloads the HF dataset (~103 MB).
- Dynamic's incremental memory is an interpretation for a per-sentence detection
  benchmark; its production value is graph maintenance, not raw detection F1.
- Research-only — archive before prod per `AGENTS.md`; not wired into
  recommendation/orchestration product paths.

## E2E verification (2026-07-04)

Offline smoke (`nlp` + `ontology`, sample_size=5, seed=42):

| Method | F1 (strict) | Notes |
|--------|-------------|-------|
| NLP / OpenIE | 0.000 | spaCy general NER misses FiNER numeric XBRL spans on small sample |
| Ontology / Schema | 0.429 | Keyword + numeric regex baseline |

API smoke: `POST /experiments/finer139/runs` → poll `GET .../runs/{id}` until
`complete`; results JSON includes per-method metrics and example spans.

## Results

*(Fill in after running the tab with your chosen sample size and LLM.)*

| Method | P (strict) | R (strict) | F1 (strict) | F1 (relaxed) | Latency |
|--------|------------|------------|-------------|--------------|---------|
| | | | | | |
