# FiNER-139 methodology

How the graph-construction benchmark is designed, scored, and reproduced.

## Dataset

- **Source:** [nlpaueb/finer-139](https://huggingface.co/datasets/nlpaueb/finer-139)
- **Task:** IOB2 token classification; gold entities are **numeric tokens** with one of **139 XBRL concept types** (e.g. `DebtInstrumentFaceAmount`). The correct tag depends on context.
- **Loader:** `src/experiments/finer139/dataset.py` (lazy HuggingFace download, ~103 MB)

## Fair comparison protocol

1. **Type-agnostic detection** — score span location, not the 139-way XBRL type.
2. **Numeric-only evaluation universe** — filter all predictions to numeric expressions before matching. FiNER gold is numeric-only, so recall is unchanged; open-ended methods are not penalized for non-numeric extractions.
3. **Each method in its ideal config** (see table below). Dynamic runs as memory-augmented incremental extraction.

## Methods

| Method | Implementation | LLM |
|--------|----------------|-----|
| **LLM-Based** | Open-ended prompt → JSON entity list → span alignment | Yes (`gpt-4o-mini` default) |
| **NLP / OpenIE** | spaCy `en_core_web_sm`, MONEY/PERCENT/CARDINAL/QUANTITY/DATE/ORDINAL | No |
| **Ontology / Schema** | XBRL keyword gazetteer + numeric regex when keywords co-occur | No |
| **Hybrid Construction** | LLM with full 139-concept schema + numeric post-filter | Yes |
| **Dynamic / Incremental** | Memory-augmented wrapper over Hybrid | Yes |

Hybrid here means **schema-guided graph construction**, not hybrid retrieval (vector + graph at query time). See [../retrieval-patterns.md](../retrieval-patterns.md).

## Metrics

**Primary:** strict micro-F1 (exact token span, numeric-only universe).

**Extended (v2/v3)** via `src/experiments/finer139/analysis.py`:

| Metric | Purpose |
|--------|---------|
| Partial micro F1 (IoU ≥ 0.5) | Nearly-correct spans |
| Macro strict F1 | Per-sentence average |
| Sentence hit rate | % of gold sentences with ≥1 strict TP |
| Bootstrap 95% CI | Uncertainty from sentence resampling |
| Error taxonomy | `boundary_fp` vs `spurious_fp` |
| Head-to-head sentence wins | Per-sentence strict-F1 winner |
| Paired bootstrap delta (v3) | CI for F1 difference between two methods |
| McNemar sentence hits (v3) | Discordant counts on strict-hit |
| Multi-seed aggregate (v3) | Mean / std / min / max across seeds |

Also: relaxed micro-F1, latency, LLM call counts.

## Sampling controls (protocol v3)

| Control | Purpose |
|---------|---------|
| `split` | `train` / `validation` / `test` — prefer **test** for final reports |
| `stratified` | Proportional sample by primary gold XBRL concept |
| `seed` / `seeds[]` | Reproducible draws; multi-seed → aggregate stats |
| `suite_path` | Frozen suite (`data/finer139_frozen_suite_v1.json`) or CI fixture |

```python
from experiments.finer139.runner import RunParams, run_benchmark

run_benchmark(RunParams(sample_size=100, seed=42, split="test", stratified=True))
run_benchmark(RunParams(seeds=[42, 43, 44], methods=["nlp", "ontology"]))
run_benchmark(RunParams(suite_path="tests/fixtures/finer139_mini_corpus.json", methods=["ontology"]))
```

## Architecture

| Layer | Path |
|-------|------|
| Engine | `src/experiments/finer139/` |
| API | `POST/GET /experiments/finer139/runs` on orchestrator `:8002` |
| UI | `frontend/app/(main)/graphrag/` + `features/finer139/` |
| Persistence | In-memory only (save JSON manually after runs) |

## Caveats

- Measures numeric financial entity **detection**, not general NER or faithful XBRL typing.
- Dynamic's memory layer is interpreted for per-sentence benchmarking; production value is graph maintenance.
- Research-only — archive before prod.
