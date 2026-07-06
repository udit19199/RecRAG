# FiNER-139: Which Graph-Construction Method Recognizes Entities Best?

Research finding for **action item #2** from the GraphRAG analysis.

We benchmark **five graph-construction entity extractors** on the
[FiNER-139](https://huggingface.co/datasets/nlpaueb/finer-139) validation split.
Gold labels are **numeric tokens** tagged with XBRL concept types; we score
**type-agnostic span detection** (fair comparison across methods).

---

## Benchmark setup

| Parameter | Value |
|-----------|-------|
| Dataset | `nlpaueb/finer-139`, validation split |
| Sample | 100 sentences (seed 42), 160 gold numeric entities |
| Primary metric | **Strict micro-F1** (exact token-span match) |
| Extended (v2) | Partial F1 (IoU≥0.5), macro F1, bootstrap CI, error taxonomy, head-to-head |
| Secondary | Relaxed F1, concept-stratified recall, latency, LLM calls |
| Engine | `src/experiments/finer139/runner.py` |
| UI | `/finer139` tab on orchestrator `:8002` |
| Latest artifact | `finer139-benchmark-latest.json` |
| HTML report | [finer139-benchmark-report.html](./finer139-benchmark-report.html) |

---

## Method ranking (2026-07-06 run)

### Among methods with scores

| Rank | Method | F1 (strict) | Recall | Precision | Verdict |
|------|--------|-------------|--------|-----------|---------|
| **1** | **Ontology / Schema-Driven** | **47.5%** | **91.9%** | 32.0% | **Best overall** — highest strict F1 and recall |
| 2 | NLP / OpenIE (spaCy) | 9.8% | 15.6% | 7.1% | Partial F1 20.4%; high spurious FP count |

### Head-to-head (sentence-level strict F1)

| Method | Sentence wins | Ties |
|--------|---------------|------|
| **Ontology** | **74** | (15 total ties) |
| NLP | 11 | |

Ontology wins on **74%** of gold sentences outright; NLP only leads on 11.

### LLM methods (pending `OPENAI_API_KEY`)

| Method | Status | Expected role |
|--------|--------|---------------|
| LLM-Based (open-ended) | Not scored | Flexible but may over-extract non-numeric entities |
| Hybrid (Schema-Guided LLM) | Not scored | Combines 139-concept schema + LLM — candidate to beat ontology on precision |
| Dynamic / Incremental | Not scored | Memory-augmented Hybrid — may improve recall on streaming filings |

**Provisional winner:** **Ontology / Schema-Driven** is the best method **among all
methods run to date**. Final ranking across all five requires re-running with
`OPENAI_API_KEY` set (default: OpenAI `gpt-4o-mini`).

---

## Why ontology wins (so far)

1. **FiNER gold is numeric + XBRL-contextual** — ontology fires a gazetteer built from
   139 CamelCase-split concept names, then takes numeric expressions when keywords
   co-occur. That matches the evaluation universe.
2. **spaCy is generic** — MONEY/CARDINAL labels catch many numerics but with wrong
   boundaries vs FiNER tokenization → low strict precision (7.1%).
3. **LLM methods untested** — Hybrid was designed specifically for this schema; it may
   close the precision gap while keeping recall — **must be measured**.

---

## Recommendations for graph construction

| Use case | Recommended method | Rationale |
|----------|-------------------|-----------|
| **Production graph build (financial filings)** | Start with **Ontology** | Best F1/recall without API cost |
| **Exploration / non-numeric entities** | Add **LLM-Based** after key is set | Broader entity types (not scored on FiNER strict) |
| **Max precision on XBRL numerics** | Benchmark **Hybrid** vs Ontology | Schema-guided LLM may reduce false positives |
| **Streaming / incremental graphs** | Evaluate **Dynamic** after Hybrid baseline | Memory layer only helps if Hybrid is strong |

---

## How to complete the full 5-way comparison

```bash
uv sync --extra experiments
python -m spacy download en_core_web_sm
echo 'OPENAI_API_KEY=sk-...' >> .env
make orchestrator && make frontend
```

1. Open **FiNER-139** → select all 5 methods → Run benchmark  
2. Regenerate report: `uv run python scripts/generate-finer139-report.py`  
3. Update this file with final ranking

---

## Related docs

- Methodology: [finer139-graph-construction.md](./finer139-graph-construction.md)
- Stakeholder summary: [finer139-showcase-results.md](./finer139-showcase-results.md)
- Retrieval patterns (action item #1): [graphrag-retrieval-patterns.md](./graphrag-retrieval-patterns.md)
