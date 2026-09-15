# Experiment registry

This registry is the human index for the artifacts in `results/`. Dataset names
are run metadata. The registry is the place to learn what was run, how much of
it completed, and what the result can support.

## Status definitions

- **Complete** means every intended comparison for the run finished.
- **Incomplete** means the artifact is useful, but some planned combinations,
  questions, or evaluation stages are missing.
- **Historical** means the run belongs to an older path and does not define the
  active GraphRAG comparison.
- **Exploratory** means the artifact helped investigate a method but was not a
  controlled configuration decision.

## Saved runs

| Date and run | Dataset | Scale | Methods covered | Status | What it supports | Artifact |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-26, first three reports | Hannon | 39 records, 7 questions | LLM construction, node retrieval | Incomplete | Early retrieval and construction observations | [11:59](../results/graphrag/20260826_115937_77c0/report.json), [12:11](../results/graphrag/20260826_121112_fddd/report.json), [12:28](../results/graphrag/20260826_122839_9f6b/report.json) |
| 2026-08-26, `125648_431a` | Hannon | 39 records, 7 questions | LLM construction, node and hybrid retrieval | Incomplete | Hybrid versus node comparison on the early path | [report](../results/graphrag/20260826_125648_431a/report.json) |
| 2026-08-27, `132202_1451` | Hannon | 39 records, 7 questions | LLM and NLP construction, node retrieval | Incomplete | Construction comparison and partial retrieval evidence | [report](../results/graphrag/20260827_132202_1451/report.json) |
| 2026-09-02, `head2head_000347_f793` | HotpotQA | 20 configured examples | Seven earlier retrieval methods | Exploratory | Earlier retrieval-method investigation | [report](../results/graphrag/head2head_20260902_000347_f793/report.json) |
| 2026-09-02, `123642_a3a9` | HotpotQA | One reported example | Seven earlier retrieval methods | Exploratory | Earlier construction and retrieval investigation | [report](../results/graphrag/20260902_123642_a3a9/report.json) |
| 2026-09-02, `125115_9d39` | HotpotQA | One reported example | Seven earlier retrieval methods, with LLM judge | Exploratory | Earlier judged retrieval investigation | [report](../results/graphrag/20260902_125115_9d39/report_with_llm_judge.json) |
| 2026-09-09, `2026090905` | 2WikiMultiHopQA | 5 records | Full 2 × 4 matrix | Complete, small | Current provisional configuration decision | [report](../results/recrag/2026090905_report.pdf) |
| 2026-08-14, `seed42_n10` | FiNER-139 | 10 records | Extraction benchmark | Historical | Preserved older benchmark result | [report](../results/finer139/20260814_120902_seed42_n10.json) |

The raw Hannon reports are in
[`results/graphrag`](../results/graphrag). The current 2Wiki report is
[`results/recrag/2026090905_report.pdf`](../results/recrag/2026090905_report.pdf).

## Reading a run correctly

Do not compare numbers across rows unless the dataset, sample, model, method
definitions, and evaluation path are the same. A report with many records can
still be incomplete if it covers only one retrieval method or a small set of
questions.

The five-record 2Wiki report is the only saved artifact currently used for the
provisional configuration decision. The larger Hannon reports explain the
project's earlier choices but do not replace the larger 2Wiki study.
