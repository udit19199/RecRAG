# Research Findings — GraphRAG Analysis Action Items

Artifacts from the GraphRAG analysis follow-up work in RecRAG.

## Action items

| # | Item | Status | Document |
|---|------|--------|----------|
| 1 | Retrieval details: hybrid + community graph; PathRAG vs neighborhood | **Done** | [graphrag-retrieval-patterns.md](./graphrag-retrieval-patterns.md) |
| 2 | FiNER-139 dataset: best graph-construction method for entity recognition | **Done** (2/5 methods scored; LLM pending API key) | [finer139-method-ranking.md](./finer139-method-ranking.md) |

## FiNER-139 experiment

| Artifact | Description |
|----------|-------------|
| [finer139-benchmark-report.html](./finer139-benchmark-report.html) | Visual benchmark report |
| [finer139-benchmark-latest.json](./finer139-benchmark-latest.json) | Latest run JSON |
| [finer139-graph-construction.md](./finer139-graph-construction.md) | Methodology |
| [finer139-showcase-results.md](./finer139-showcase-results.md) | Executive summary |

## Code & UI

- Engine: `src/experiments/finer139/`
- API: `app/orchestrator/experiments/router.py`
- UI: `frontend/app/(main)/finer139/`
