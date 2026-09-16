# Research roadmap

## Next study

The next study is a larger 2WikiMultiHopQA run. It will repeat the complete
2 × 4 matrix:

- `standard` and `ontology_guided` construction;
- `text2cypher`, `agentic`, `vector`, and `entity_vector` retrieval;
- the same answer model, embedding model, Neo4j setup, and evaluation rules.

The sample must be larger than the completed five-record run. The exact sample
size can be chosen when the run is prepared, but it must be recorded with the
split and selection rule.

## Decision after the run

The project will first compare correct, evidence-grounded final answers. It
will then use faithfulness, retrieval metrics, and graph metrics to explain the
winner or separate a tradeoff.

The project will not lock a configuration from a run that is missing a matrix
cell, has uneven question coverage, or mixes evaluation protocols.

## Later studies

HotpotQA, FinQA, TAT-QA, LegalBench-RAG, and BioASQ remain candidates. No order
has been chosen. Their purpose is to test whether a configuration selected on
multi-hop Wikipedia questions transfers to different document structures and
answer types.

The project should choose the next dataset after the larger 2Wiki result, based
on what the result leaves unclear. Dataset selection is a research decision,
not a fixed implementation backlog.

## Open questions

- Does `standard + vector` remain the best answer configuration at larger scale?
- Does the graph-quality advantage of `ontology_guided` improve final answers
  when the sample is larger?
- Is `text2cypher` failing because of graph construction, query generation, or
  the evaluation setup?
- Which later dataset will best test the next unresolved question?
