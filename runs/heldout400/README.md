# Held-out 400 study (MultiHop-RAG, 2026-11-02 to 2026-11-07)

The latest RecRAG experiment and its judge runs. Everything needed to read or
rescore it is in this folder. Older studies were removed on 2026-11-08.

400 MultiHop-RAG questions, 100 per type (comparison, temporal, inference,
null), never used in earlier runs. All 4 methods answered every question in
3 repeats (r1, r2, r3). Each repeat ran in 4 shards of 100. A separate
40-question, one-worker pass measured latency.

## Headline results (`analysis.json`)

| Method | Accuracy (EM + Haiku judge) | Gold-sentence recall | USD per 1,000 questions (incl. build) | Median latency (s) |
|---|---|---|---|---|
| RAG | 0.726 | 0.437 | 0.31 | 2.2 |
| RAPTOR | 0.727 | 0.442 | 0.42 | 2.1 |
| HippoRAG | 0.746 | 0.406 | 1.96 | 4.8 |
| Agentic GraphRAG | 0.748 | 0.466 | 7.00 | 32.5 |

The per-type and per-repeat view, which includes answer support, is in
`reports/email-heldout-short.md`.

## Layout

| Path | What it holds |
|---|---|
| `questions.json` | The frozen 400-question set |
| `manifest.jsonl` | One row per (pass, method, shard), pointing to its folder under `runs/`. `manifest.original-paths.jsonl` has the paths from before the move |
| `runs/<method>/<timestamp>/` | The 56 run folders (`results.jsonl`, `retrieval.jsonl`, `settings.json`, `summary.json`) |
| `logs/`, `driver_*.log` | Run logs |
| `judge/` | Haiku answer and evidence judge items and verdicts. `-strict-` verdicts replace the earlier lenient ones |
| `rejudge/` | Strict re-judge of the 194 batch-2 evidence items and the 38 items that were never judged |
| `judge_support/` | gpt-6-luna answer-support judge, one folder per (repeat, shard). `merged/` joins the two halves of r2 Agentic GraphRAG |
| `build_calibration/` | Metered one-time build cost for the graph and RAPTOR tree |
| `analysis.json`, `analysis_by_type.json`, `confusion.json`, `confusion_by_run.json`, `insights.json`, `hand_label_sample.json`, `by_type_chart.html` | Analysis outputs |
| `reports/` | The 3 Nov and 7 Nov emails |
| `scripts/` | Scripts that picked, ran, judged and analysed the study. Run them from the repo root |

r2 Agentic GraphRAG crashed partway. Each shard was finished in two folders:
`*-r2resume` and the `20261101T231404*` resume run.

## Shared inputs outside this folder

- Indexes used by these runs, all still in place:
  - `runs/rag/indexes/07062033…`
  - `runs/raptor/indexes/be25c875…`
  - `runs/hipporag/indexes/06a6bf29…`
  - `runs/graphrag/indexes/e5315c2b…`
  - the matching `runs/graphrag/construction-cache/recrag-benchmark-e5315c2b….sqlite`
  - Neo4j database `recrag-benchmark-e5315c2b5c9f57e2595cc51c`
- `runs/retrieval_judge/cache.jsonl`: the answer-support judge call cache. A rerun of the judge reads from it instead of paying again.

## Scripts that will not rerun as-is

- `pick_heldout.py` excluded every question seen in earlier runs, and those runs were deleted. Rerunning it would pick a different set. Use `questions.json`.
- `raptor_cost_calibration.py` read a RAPTOR run that was deleted. Its output is saved in `build_calibration/`.
