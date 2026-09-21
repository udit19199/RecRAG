# RecRAG documentation

RecRAG compares graph construction and retrieval methods for multi-hop questions.
The active app runs the same GraphRAG pipeline on HotpotQA, TriviaQA,
2WikiMultiHopQA, or Natural Questions. The router study uses the run results to
compare candidate designs.

Read this page first. It records the facts shared by the other guides.

## Current scope

The app supports these construction methods:

| Method | Meaning |
| --- | --- |
| `standard` | The model infers graph types from the source text. |
| `ontology_guided` | The model starts with fixed top-level graph types and may add more. |

It supports these retrieval methods:

| Method | Meaning |
| --- | --- |
| `agentic` | An agent chooses between vector and Cypher search. |
| `vector` | Vector search finds chunks, then a Cypher query adds graph context. |
| `entity_vector` | Vector search finds entity-linked chunks, then returns graph context. |

The active app reads one selected dataset through its adapter and accepts up to
20 records. Natural Questions expects a dev JSONL file under
`datasets/natural_questions/`.

Every record and construction method uses its own Neo4j database. Neo4j stores
the graph, chunks, embeddings, and vector indexes. The public interface is
`GraphRAG`.

Use `gpt-5.6-luna` for extraction, retrieval, answers, and evaluation judges.
OpenAI generation uses the Responses API with medium reasoning effort. API keys
stay in `.env`, and adapter timeouts come from `config.toml`.

## Find a guide

| If you want to know... | Read... |
| --- | --- |
| How a source record becomes graph data | [Graph construction](graphrag/construction.md) |
| How the graph returns context and an answer | [Graph retrieval](graphrag/retrieval.md) |
| How to score construction, retrieval, and answers | [GraphRAG evaluation](graphrag/evaluation.md) |
| What the first five-record run found | [2WikiMultiHopQA results](graphrag/2wikimultihopqa-results.md) |
| Which datasets are available and how a run reports cost | [API cost estimate](api-cost-estimate.md) |
| How the router chooses a candidate | [Router guide](router.md#choose-a-candidate) |
| How to measure candidates and router choices | [Router guide](router.md#measure-candidates) |
| How Jev fits into the router | [Router guide](router.md#use-jev) |

## Terms

- **Record:** One dataset question, its source pages, and its evaluation fields.
- **Construction:** The process that turns source pages into a graph and search indexes.
- **Retrieval:** The process that selects context for a question.
- **Candidate:** One architecture, construction method, and retrieval method.

## Historical research

[REFinD experiment](refind/refind-experiment.md) preserves old notes and figures.
It does not describe the active implementation.
