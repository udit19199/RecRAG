# Research agenda

## The decision

RecRAG is selecting a GraphRAG configuration for questions that require facts
from more than one piece of source text. A configuration is the combination of
one graph-construction method and one retrieval method.

The decision is practical. We want to know which configuration gives the most
correct answers while staying grounded in the evidence returned from the
source material.

## Why the graph matters

The source documents contain both text and relationships between named things.
RecRAG uses Neo4j to keep those two forms of information together:

- the source is split into searchable chunks;
- an LLM extracts entities and relationships into a graph;
- vector and full-text indexes support different search behaviours;
- retrieval can add graph entities and paths to the source context;
- the answer model receives the retrieved context and produces the answer.

The experiment is not asking whether Neo4j can store a graph. It is asking which
choices on top of that platform make the final answer better.

## Working hypotheses

These are hypotheses to test, not conclusions:

1. The way the graph is constructed changes how much useful evidence the graph
   preserves.
2. Retrieval methods that combine semantic search with graph context may work
   better for multi-hop questions than a single retrieval strategy.
3. A construction method that produces a better graph may not produce the best
   final answer when paired with every retriever.

## Active scope

The active study uses 2WikiMultiHopQA because each example includes a question,
source pages, a known answer, and supporting passages. It lets the project
compare both graph quality and answer quality on a shared reference.

The study compares:

| Dimension | Choices |
| --- | --- |
| Construction | `standard`, `ontology_guided` |
| Retrieval | `text2cypher`, `agentic`, `vector`, `hybrid` |
| Answering | The configured `gpt-5.6-luna` answer model |
| Storage and graph operations | Neo4j |

The larger 2WikiMultiHopQA run must complete all eight construction and
retrieval combinations before the project locks a winner.

## Boundaries

The project is research-first. It is not currently trying to build a
production-scale service, prove that GraphRAG always beats plain RAG, or choose
the order of all future datasets. Older experiments remain useful when they
explain why the current study is shaped this way.
