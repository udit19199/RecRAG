# Research Findings — GraphRAG Analysis Action Items

Artifacts from the GraphRAG analysis follow-up work in RecRAG.

## Action items

| # | Item | Status | Document / UI |
|---|------|--------|---------------|
| 1 | Retrieval details: hybrid + community graph; PathRAG vs neighborhood | **Done** | [graphrag-retrieval-patterns.md](./graphrag-retrieval-patterns.md) · UI: `/graphrag` |
| 2 | FiNER-139 dataset: best graph-construction method for entity recognition | **Done** (offline methods scored; LLM methods runnable with provider key) | [finer139-method-ranking.md](./finer139-method-ranking.md) · UI: `/graphrag?tab=benchmark` |

## FiNER-139 experiment

| Artifact | Description |
|----------|-------------|
| [finer139-benchmark-report.html](./finer139-benchmark-report.html) | Visual benchmark report |
| [finer139-benchmark-latest.json](./finer139-benchmark-latest.json) | Latest run JSON |
| [finer139-graph-construction.md](./finer139-graph-construction.md) | Methodology |
| [finer139-showcase-results.md](./finer139-showcase-results.md) | Executive summary |

## Code & UI

- GraphRAG research hub: `frontend/app/(main)/graphrag/` (`/graphrag` — retrieval tab; `/graphrag?tab=benchmark` — FiNER-139)
- `/finer139` redirects to the benchmark tab
- Engine: `src/experiments/finer139/`
- API: `app/orchestrator/experiments/router.py`
