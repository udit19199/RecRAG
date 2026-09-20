# RecRAG documentation

RecRAG compares retrieval designs on multi-hop questions. The active app runs
GraphRAG on 2WikiMultiHopQA. The router study uses those measurements to
compare candidate designs.

Use this page to choose where to read.

## Current GraphRAG

| If you want to know... | Read... |
| --- | --- |
| How source pages become a Neo4j graph | [Graph construction](graphrag/construction.md) |
| How the graph returns context and an answer | [Graph retrieval](graphrag/retrieval.md) |
| How the graph, context, and answer are scored | [GraphRAG evaluation](graphrag/evaluation.md) |
| What the first five-record run found | [2WikiMultiHopQA results](graphrag/2wikimultihopqa-results.md) |

## Benchmark and router research

| If you want to know... | Read... |
| --- | --- |
| Which datasets are available and how much a run may cost | [Benchmark plan](benchmark.md) |
| What the router chooses | [Router decision](router/decision.md) |
| How candidates and routers are measured | [Router measurement](router/measurement.md) |
| How Jev fits into the router | [Jev in the router](router/jev.md) |

## Historical research

- [REFinD experiment](refind/refind-experiment.md) contains preserved notes and
  figures. It does not describe the active implementation.
