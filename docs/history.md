# Research history

The project has changed its datasets, extraction methods, retrieval shapes, and
evaluation paths over time. This history keeps those experiments visible while
separating them from the active 2WikiMultiHopQA study.

## Hannon

The early Hannon study used 39 records and 7 synthesis questions. It compared
LLM and NLP construction and explored node and hybrid retrieval. The LLM path
produced a richer graph. The retrieval comparison was incomplete, so the study
did not select a final configuration.

Its main value was diagnostic. It showed that construction and retrieval need to
be evaluated separately and that a large record count does not make a run a
complete configuration study.

See the detailed interpretation in [Findings](findings.md) and the saved reports
in the [experiment registry](experiment-registry.md).

## REFinD

REFinD is an older graph-extraction experiment. It tested entity and
relationship extraction from records and used the extracted facts to support
multi-record synthesis questions. The retrieval path was not fully preserved in
the supplied material, so the experiment is evidence about an earlier idea,
not a description of the active system.

The original detailed note is [REFinD experiment](refind-experiment.md).

## FiNER-139

The FiNER-139 extraction benchmark was removed from the active GraphRAG path.
Its latest result remains preserved under `results/finer139/` as historical
evidence.

## Retired product direction

The former candidate-pipeline recommendation product and its orchestrator are
not part of the current research plan. The active work compares GraphRAG
construction and retrieval choices directly.
