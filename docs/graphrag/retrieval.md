# Graph retrieval

GraphRAG supports agentic retrieval only. `GraphRAG.answer()` delegates to
`answer_question()`, which reads the database schema and builds the agent.

The agent chooses between two tools:

| Tool | Implementation | Evidence |
| --- | --- | --- |
| `vector_search` | Neo4j `VectorCypherRetriever` over `chunk_embeddings` | Chunk text, linked entities, and up to 25 nearby graph paths spanning one or two relationships. |
| `cypher_search` | Neo4j `Text2CypherRetriever` | Records returned by a generated Cypher query using the database schema. |

These are internal tools, not standalone retrieval methods.

## Agent and answer flow

1. Read the target Neo4j database schema.
2. Run the LangChain agent with both tools and a five-model-call limit.
3. Collect tool result artifacts and keep at most five evidence items.
4. Pass those items to the official `neo4j-graphrag` answer flow.
5. Return the answer and retrieved context for display and optional evaluation.

The agent is instructed to call one tool before answering, refine its search
when evidence is insufficient, and avoid repeating searches with no new evidence.
The Cypher tool is described as read-only; use database permissions to enforce
that restriction.

The configured LangChain chat model drives the agent. `GraphRAGChatLLM` adapts
that model for text-to-Cypher generation and the final answer. Generation uses
`gpt-6-luna` with medium reasoning through the Responses API.

## Inputs and results

`GraphRAG.answer(question, database=..., retrieval_methods=["agentic"])`
returns a list containing one `RagResultModel`. The list result shape is kept
for existing callers. An empty method list returns no results; any method
other than `agentic` raises `ValueError` before retrieval starts.

Each result contains `answer` and `retriever_result`. Evidence items contain
serialized Neo4j records and tool metadata. Answer generation and retrieval
evaluation use the same five-item context limit.

When no context is found, the answer fallback is:

> I could not find supporting context for this question.

## Code and related docs

- [Agent and tools](../../graphrag/agentic.py)
- [Vector context query and record formatting](../../graphrag/retrievers.py)
- [Answer flow](../../graphrag/answering.py)
- [Graph construction](construction.md)
- [Evaluation](evaluation.md)
