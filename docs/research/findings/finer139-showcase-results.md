# FiNER-139 Benchmark Results (Showcase)

**Date:** 2026-07-06  
**Dataset:** [nlpaueb/finer-139](https://huggingface.co/datasets/nlpaueb/finer-139) validation split  
**Sample:** 100 sentences (seed 42), 160 gold numeric entities  
**Evaluation:** Type-agnostic numeric span detection (strict token-span match)

## Executive summary

We compared five graph-construction entity-recognition approaches on real financial filing sentences.

**Winner (among methods scored): Ontology / Schema-Driven** — strict F1 **47.5%**, recall **91.9%**.

Full ranking and recommendations: [finer139-method-ranking.md](./finer139-method-ranking.md).
HTML report: [finer139-benchmark-report.html](./finer139-benchmark-report.html).

**NLP / OpenIE (spaCy)** finds many overlapping spans (relaxed F1 **55.3%**) but poor strict precision (**7.1%**). LLM-based methods require `OPENAI_API_KEY` to complete the 5-way comparison (default: OpenAI `gpt-4o-mini`).

## Results table (live run, 2026-07-06)

| Method | P (strict) | R (strict) | F1 (strict) | F1 (relaxed) | Latency | LLM calls |
|--------|------------|------------|-------------|--------------|---------|-----------|
| **Ontology / Schema-Driven** | **32.0%** | **91.9%** | **47.5%** | 47.5% | 0.0s | — |
| NLP / OpenIE (spaCy) | 7.1% | 15.6% | 9.8% | **55.3%** | 0.9s | — |
| LLM-Based (open-ended) | — | — | — | — | — | needs `OPENAI_API_KEY` |
| Hybrid (Schema-Guided LLM) | — | — | — | — | — | needs `OPENAI_API_KEY` |
| Dynamic / Incremental | — | — | — | — | — | needs `OPENAI_API_KEY` |

### How to read this

- **Strict F1** — exact token-span match against FiNER-139 gold numeric entities (primary metric).
- **Relaxed F1** — any token overlap; spaCy over-predicts non-gold spans, inflating relaxed scores.
- **Ontology** — keyword gazetteer + numeric regex when XBRL concepts co-occur; fast and high-recall but moderate precision.
- **LLM methods** — blocked in this environment without `OPENAI_API_KEY`; re-run the tab with the key set to complete the comparison.

## Key findings for stakeholders

1. **Schema-driven rules are strong baselines** for numeric financial entity detection in filings — 92% recall without any LLM cost.
2. **Generic NLP (spaCy) is not sufficient alone** for precise span detection (9.8% strict F1) though it surfaces many candidate numerics.
3. **LLM comparison is the open question** — Hybrid and Dynamic are designed to combine schema guidance + LLM flexibility; results pending API key.
4. **Reproducible** — same seed/sample yields identical offline numbers; artifacts in `finer139-e2e-offline.json`.

## How to reproduce

```bash
uv sync --extra experiments
python -m spacy download en_core_web_sm
mkdir -p state   # orchestrator SQLite
# Set OPENAI_API_KEY in .env for LLM methods
make orchestrator   # :8002
make retrieval      # :8000 (provider picker)
make frontend       # :3000
```

Open **FiNER-139** tab → Run benchmark. Screenshots: `docs/research/findings/screenshots/`.

## Artifacts

| File | Description |
|------|-------------|
| `finer139-e2e-offline.json` | API run: nlp + ontology, n=100 |
| `finer139-e2e-full.json` | API run: all 5 methods (LLM blocked without key) |
| `screenshots/01-finer139-config.png` | UI config panel |
| `screenshots/02-finer139-results.png` | Results table after live UI run |
| `screenshots/03-finer139-examples.png` | Gold vs predicted span examples |
| `screenshots/04-finer139-nav.png` | Sidebar navigation |
