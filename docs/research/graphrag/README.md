# GraphRAG research

Follow-up work from the GraphRAG analysis. Two action items, each with one primary doc and a UI surface.

| # | Question | Doc | UI |
|---|----------|-----|-----|
| 1 | How does graph retrieval differ from flat vector RAG? | [retrieval-patterns.md](./retrieval-patterns.md) | `/graphrag` |
| 2 | Which graph-construction method finds entities best on financial text? | [finer139/README.md](./finer139/README.md) | `/graphrag?tab=benchmark` |

## Code

| Component | Path |
|-----------|------|
| FiNER-139 benchmark engine | `src/experiments/finer139/` |
| Orchestrator experiments API | `app/orchestrator/experiments/router.py` |
| GraphRAG + FiNER UI | `frontend/app/(main)/graphrag/`, `frontend/src/features/finer139/` |

Research-only. Not wired into recommendation or production orchestration paths.
