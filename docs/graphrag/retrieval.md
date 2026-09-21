# Graph retrieval

Retrieval selects Neo4j records for one question, then passes those records to
the answer model. The public entry point is `GraphRAG.answer()` in
[`graphrag/graph_rag.py`](../../graphrag/graph_rag.py#L256-L271). It delegates to
`answer_question()` in
[`graphrag/retrieval/answering.py`](../../graphrag/retrieval/answering.py#L26-L65).

The active implementation has three methods. All three read one Neo4j database
created for the record and construction method. The methods differ in the
index or query used to find evidence.

```mermaid
flowchart TD
    A[Question] --> B[GraphRAG.answer]
    B --> C[answer_question]
    C --> D{Selected methods, in input order}
    D --> E[agentic]
    D --> F[vector]
    D --> G[entity_vector]
    E --> H[RetrieverResult]
    F --> H
    G --> H
    H --> I[neo4j-graphrag GraphRAG]
    I --> J[Answer and retriever context]
```

## Inputs and outputs

`GraphRAG.answer()` receives the question and two keyword arguments:

| Input | Source | Use |
| --- | --- | --- |
| `question` | The dataset record | The vector query, Cypher prompt, and answer prompt. |
| `database` | `ConstructionMethod.database_name(record.id)` | The Neo4j database read by every retriever in this call. |
| `retrieval_methods` | The Streamlit method selection | The methods to build and run, in the given order. |
The `GraphRAG` instance keeps the configured models and passes them to the
retrievers. `GraphRAG.from_config()` creates these model objects:

| Object | Used by |
| --- | --- |
| `GraphRAGChatLLM` around the configured `ChatOpenAI` | The Neo4j GraphRAG answer call and `Text2CypherRetriever`. |
| The configured `ChatOpenAI` instance | The LangChain agent in `agentic`. |
| `OpenAIEmbeddings` | The `vector` and `entity_vector` searches. |

`answer_question()` returns one `RagResultModel` for each selected method. The
list order matches `retrieval_methods`. Each result has:

| Field | Meaning |
| --- | --- |
| `answer` | The answer model's text, or the no-context fallback. |
| `retriever_result.items` | The evidence items used to build the answer prompt. Each item has `content` and optional `metadata`. |
| `retriever_result.metadata` | Retriever-specific metadata when the retriever supplies it. |

The call always sets `return_context=True`, so the app can display the items and
the evaluators can score them. The vector retrievers use their default `top_k`
of 5 because `answer_question()` does not pass a `retriever_config` value.

## The shared answer flow

`answer_question()` follows the selected method list rather than a fixed method
order:

1. If `agentic` is selected, call `get_schema()` once for the target database.
2. Build the retriever for the current method.
3. Call `neo4j_graphrag.generation.GraphRAG.search()` with the question.
4. Return the answer and retriever context before building the next result.

The call uses this code:

```python
Neo4jGraphRAG(retriever, llm).search(
    question,
    return_context=True,
    response_fallback=NO_CONTEXT,
)
```

`llm` is the `GraphRAGChatLLM` adapter. It turns the LangChain
`ChatOpenAI` instance into the `neo4j-graphrag` interface and generates the
final answer. The answer prompt joins each retrieved item's `content` with a
newline. If the retriever returns no items, Neo4j GraphRAG skips the answer
model and returns this text:

```text
I could not find supporting context for this question.
```

The [documentation index](../index.md) lists the shared model, API, and
configuration rules.

## What each retriever reads

Construction creates two vector indexes in the target database:

| Index | Label | Property | Used by |
| --- | --- | --- | --- |
| `chunk_embeddings` | `Chunk` | `embedding` | `vector` and the vector tool in `agentic` |
| `entity_embeddings` | `EntityEmbedding` | `embedding` | `entity_vector` |

The embedding model turns the question into a vector before a vector search.
The construction step copies each chunk embedding to an `EntityEmbedding` node
linked from its entity. See [Graph construction](construction.md#build-the-graph)
for the write path and [storage option 1](construction.md#storage-option-1-neo4j-for-the-graph-and-embeddings)
for the resulting shape.

All retrievers use Neo4j reads. The vector retrievers call
`VectorCypherRetriever`; the agentic Cypher tool uses
`Text2CypherRetriever`. The Text2Cypher retriever runs `EXPLAIN` first and
rejects a generated query unless Neo4j reports it as read-only.

## Vector retrieval

`build_vector_retriever()` creates a `VectorCypherRetriever` with the
`chunk_embeddings` index and `VECTOR_RETRIEVAL_QUERY` from
[`graphrag/retrieval/retrievers.py`](../../graphrag/retrieval/retrievers.py#L7-L53).
The retriever embeds the question, finds similar `Chunk` nodes, and runs the
retrieval query for each match.

The query returns four values:

| Value | Query expression | Meaning |
| --- | --- | --- |
| `text` | `node.text` | The matched chunk text. |
| `entities` | `collect(DISTINCT entity.name)` | Entity names connected to the chunk through `FROM_CHUNK`. |
| `graph_facts` | A collection of path maps | Entity paths near the chunk, with node names and relationship direction. |
| `score` | The vector index score | The similarity score from Neo4j. |

The query builds `graph_facts` in two subqueries:

1. Match every `__Entity__` connected to the chunk through `FROM_CHUNK`.
2. Start from those entities and follow an undirected path of one or two relationships.
3. Drop paths that contain `FROM_CHUNK`, `NEXT_CHUNK`, or `FROM_DOCUMENT`.
4. Keep at most 25 paths for the chunk.
5. Convert each path to `{nodes, relationships}`. A relationship stores its source name, type, and target name.

The query does not ask Neo4j to order the graph paths. The vector score orders
the matched chunks, while the graph paths inside one chunk are the collected
context returned by the query.

```mermaid
flowchart LR
    Q[Question] --> E[Embed question]
    E --> I[chunk_embeddings]
    I --> C[Matching Chunk]
    C --> N[Linked entity names]
    C --> P[Entity paths, up to 2 hops]
    N --> R[Retriever item]
    P --> R
    C --> R
```

![Vector retrieval flow](retrieval-vector.svg)

## Entity-vector retrieval

`build_entity_vector_retriever()` uses the `entity_embeddings` index and
`ENTITY_VECTOR_RETRIEVAL_QUERY` from
[`graphrag/retrieval/retrievers.py`](../../graphrag/retrieval/retrievers.py#L38-L65).
The query is deliberately smaller than the chunk-vector query:

```cypher
MATCH (entity:__Entity__)-[:HAS_EMBEDDING]->(node)
RETURN node.text AS text, [entity.name] AS entities, [] AS graph_facts, score
```

This method searches the copied chunk embedding through an entity-linked node.
It returns the copied chunk text and the one linked entity name. It does not
traverse the graph. Its `graph_facts` value is always an empty list.

That difference matters when comparing methods. `entity_vector` can find an
entity-associated chunk, but it does not add the two-hop graph context that
the `vector` method adds.

## Agentic retrieval

`build_agentic_retriever()` builds two tools:

| Tool | Retriever | Input and read path |
| --- | --- | --- |
| `vector_search` | `VectorCypherRetriever` | Embeds the tool query and searches `chunk_embeddings` with the same graph-context query as `vector`. |
| `cypher_search` | `Text2CypherRetriever` | Sends the tool query and the current Neo4j schema to `gpt-5.6-luna`, then checks and runs the generated read-only Cypher. |

The agent uses the configured answer `ChatOpenAI` model. Its system instruction
sets this sequence:

```text
Before any evidence has been returned, call exactly one retrieval tool.
After reviewing evidence, either call the next best tool with a refined query
or make no tool call if the evidence is sufficient.
Do not repeat a search that returned no new evidence.
```

`ModelCallLimitMiddleware` sets a limit of five model calls for one agent run.
The agent can call the second tool after it sees the first tool's result.

The wrapper in
[`graphrag/retrieval/agentic.py`](../../graphrag/retrieval/agentic.py#L23-L69)
converts each tool result into a `neo4j.Record` with these fields:

| Field | Meaning |
| --- | --- |
| `content` | The underlying retriever item's content. |
| `tool_name` | The tool that returned the item. |
| `metadata` | The underlying metadata plus the tool name. |

`AgenticToolsRetriever` collects the artifacts from every `ToolMessage`. It
returns those records as the retrieval result. It does not return the agent's
final prose as an evidence item.

![Agentic retrieval flow](retrieval-agentic.svg)

## Failure and boundary cases

- If a selected method is not one of `agentic`, `vector`, or `entity_vector`, `answer_question()` raises `ValueError`.
- If `agentic` is selected and schema loading fails, no agentic retriever is built.
- If a vector index is missing, `VectorCypherRetriever` fails while it is built. Construction must create and await the indexes first.
- If a retriever returns no items, the answer is the fallback text and `retriever_result.items` is empty.
- If the agent makes no tool call or a tool returns no records, the agentic result has no evidence items.
- Retrieval reads the database. It does not create nodes, relationships, embeddings, or indexes.

The [evaluation reference](evaluation.md) explains how the returned context and
answer are scored. The [construction reference](construction.md) explains how
the database and indexes are built.
