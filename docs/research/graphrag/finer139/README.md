# FiNER-139 graph-construction benchmark

**Action item #2:** compare five entity extractors on [FiNER-139](https://huggingface.co/datasets/nlpaueb/finer-139) (numeric tokens in SEC-style filings tagged with 139 XBRL concept types). We score **type-agnostic span detection**, not the 139-way label.

**UI:** `/graphrag?tab=benchmark` · **Engine:** `src/experiments/finer139/runner.py`

Full protocol and method definitions: [methodology.md](./methodology.md).

---

## Current results (2026-07-07, validation, n=100, seed=42, all five methods)

| Rank | Method | F1 strict | Recall | Precision | Notes |
|------|--------|-----------|--------|-----------|-------|
| **1** | **Hybrid (Schema-Guided LLM)** | **66.9%** | **73.8%** | 61.1% | Best strict F1; 100 LLM calls |
| 2 | Dynamic / Incremental | 48.0% | 91.9% | 32.5% | Same recall as ontology on this run |
| 3 | Ontology / Schema-Driven | 47.5% | 91.9% | 32.0% | Strong offline baseline |
| 4 | NLP / OpenIE (spaCy) | 9.8% | 15.6% | 7.1% | High relaxed F1 (55%); poor span boundaries |
| 5 | LLM-Based (open-ended) | 3.0% | 4.4% | 2.3% | Scored with `gpt-4o-mini`; weak strict spans |

**Head-to-head (strict, per sentence):** Hybrid wins 49 sentences; Ontology 12; spaCy 4; LLM 1; 34 ties.

**Leader:** Hybrid on strict micro-F1. Ontology/Dynamic remain strong on recall without LLM cost; Hybrid trades cost for precision and overall F1.

---

## Recommendations

| Use case | Start with | Why |
|----------|------------|-----|
| Production graph build on filings | **Hybrid** | Best strict F1 (66.9%); schema-guided LLM cuts false positives vs ontology |
| Offline / no API cost | **Ontology** | 47.5% F1, 91.9% recall; no LLM calls |
| Broader entity types | **LLM-Based** | Weak on FiNER strict spans; use only if non-numeric entities matter |
| Streaming graphs | **Dynamic** after Hybrid baseline | Memory layer; recall matches ontology on this run |

---

## Artifacts

| File | Purpose |
|------|---------|
| [report.html](./report.html) | Meeting-ready HTML report (regenerate after new runs) |
| [latest.json](./latest.json) | Canonical run — all five methods, full metrics (2026-07-07) |
| [screenshots/](./screenshots/) | UI captures from live benchmark tab |

---

## Run and regenerate

```bash
uv sync --extra experiments
python -m spacy download en_core_web_sm
# OPENAI_API_KEY in .env for LLM methods
make orchestrator && make frontend
```

1. Open **GraphRAG → Benchmark** tab, select methods, run.
2. Save completed run JSON from `GET /experiments/finer139/runs/{id}` as `latest.json` (orchestrator is in-memory only).
3. Regenerate report: `uv run python scripts/generate-finer139-report.py`
4. Update the results table in this file if rankings change.

Screenshots: `node scripts/capture-finer139-screenshots.mjs`

---

## Related

- Retrieval patterns (action item #1): [../retrieval-patterns.md](../retrieval-patterns.md)
- GraphRAG index: [../README.md](../README.md)
