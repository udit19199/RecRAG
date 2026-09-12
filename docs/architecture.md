# System overview

The main user flow lives in `streamlit_app.py`. The `GraphRAG` class connects
the app to Neo4j, the language model, and the embedding model. The code keeps
construction, retrieval, and evaluation in separate modules so the experiment
can change one part at a time.

## What happens when you click Run

The app lets you choose the number of records, construction methods, retrieval
methods, and whether to run DeepEval scoring. The record slider allows 1 to 20
records and starts at 2. Both construction methods and all four retrieval
methods are selected by default.

For each selected record, the app builds one graph for each selected construction
method. It then runs each selected retrieval method against each graph and shows
the answer and returned context. With the default choices, two records produce
four graph builds and sixteen answer attempts.

## The four stages

### 1. Load a record

The app calls `load_records(record_count)` from
`graphrag/dataset_records/two_wiki_multihopqa.py`.

The loader reads `datasets/2wikimultihopqa/dev.json`. For each record, it keeps:

- the question;
- the source pages and their passages;
- the expected answer and answer ID;
- the supporting page and passage indexes;
- the dataset's evidence triples and evidence IDs.

When evaluation checks an answer, the loader also reads aliases and demonyms
from `id_aliases.json`. This lets the evaluator accept names that mean the same
thing as the expected answer.

### 2. Build a graph

The app calls `GraphRAG.construct(...)` once for each selected construction
method. That method calls `rebuild_graph(...)` in
`graphrag/construction/construction.py`.

The builder joins each page title and its passages into text. It sends that text
to `SimpleKGPipeline` from `neo4j-graphrag`. The pipeline splits the text into
1,000-character chunks with 100 characters of overlap. It asks the LLM to find
entities and relationships, then writes the graph to Neo4j.

The app uses a separate Neo4j database for each record and construction method.
For example, a record ID becomes a database name such as
`recrag-standard-<record-id>` or `recrag-ontology-guided-<record-id>`. This keeps
the two graph versions separate while the app compares them.

The two construction choices are:

| Method | Code behavior | Reason for the comparison |
| --- | --- | --- |
| `standard` | Uses the default extraction schema and prompt. | Measures open-ended extraction. |
| `ontology_guided` | Supplies fixed top-level node and relationship types, while allowing additional types. | Measures whether a small set of named kinds improves the graph. |

The ontology-guided prompt also tells the extractor to give every node a name,
fill fields only when the text states them, use `Thing` as a fallback, use short
upper-snake-case relationship names, and avoid facts outside the text.

### 3. Create search indexes

After graph construction, the builder creates two indexes on the `Chunk` nodes:

- `chunk_embeddings` supports meaning-based search.
- `chunk_fulltext` supports exact-word search on chunk text.

The code waits for both indexes before retrieval starts. The graph, chunks, and
indexes all stay in Neo4j.

### 4. Search and answer

The app calls `GraphRAG.answer(...)`. For each selected retrieval method, the
code creates a retriever and passes it to Neo4j's `GraphRAG.search(...)`.

The search returns both an answer and the context used to produce that answer.
If no context is found, the answer falls back to:

```text
I could not find supporting context for this question.
```

The four retrieval methods work as follows:

- `text2cypher` reads the Neo4j schema and lets an LLM generate a Cypher query.
  Neo4j runs the query and returns matching graph records.
- `vector` embeds the question and searches `chunk_embeddings`. Its retrieval
  query adds entity names from each chunk and graph paths of up to two edges.
- `hybrid` searches both `chunk_embeddings` and `chunk_fulltext`. It adds the
  same entity names and graph paths as `vector`.
- `agentic` gives an agent both a vector-search tool and a Cypher-search tool.
  The agent must use one tool before it can answer. After it sees evidence, it
  can refine the search or stop. The code limits the agent to five model calls.

The answer model receives the question and the retrieved context. It writes the
final answer. The app shows the answer and lets the reader open every returned
context item.

## How evaluation works

Evaluation is optional in the app. The `Score with DeepEval` checkbox controls
whether the app runs the evaluation functions.

### Graph evaluation

`evaluate_construction(...)` compares the graph's entity relationships with two
contexts:

- all source passages for the record;
- only the dataset's supporting passages.

It asks an LLM judge to score groundedness, completeness, and supporting
evidence coverage. The code also reads graph counts such as entities,
relationships, relation types, duplicate names, isolated entities, and
self-loops.

### Retrieval evaluation

`evaluate_retrieval(...)` keeps the first five returned items by default. It
compares them with the record's supporting sentences and scores contextual
precision, contextual recall, and contextual relevancy.

### Answer evaluation

`evaluate_answer(...)` checks the generated answer against the expected answer
and its aliases. It also scores answer relevancy and correctness. The code gives
the metric the dataset's supporting sentences as `context` and the returned
items as `retrieval_context`. Faithfulness therefore checks the answer against
the gold supporting sentences. When no returned context exists, it reports that
faithfulness was not scored.

The exact alias check is stricter than the answer judge. It requires the full
normalized answer to equal the expected answer or an alias. A correct answer
with extra explanation can fail the exact check and still receive a good judge
score.

## Configuration used by the code

`GraphRAG.from_config()` reads non-sensitive settings from `config.toml` and
loads credentials from `.env`.

- The chat model is `gpt-5.6-luna` with medium reasoning effort.
- The embedding model is `text-embedding-3-small`.
- Neo4j connection details come from `NEO4J_URI` and `NEO4J_AUTH`.
- Model and embedding timeouts come from `config.toml`.

## Code map

- [`streamlit_app.py`](../streamlit_app.py) provides the controls and displays the run.
- [`graphrag/graph_rag.py`](../graphrag/graph_rag.py) creates the clients and exposes `construct` and `answer`.
- [`graphrag/construction/construction.py`](../graphrag/construction/construction.py) builds graphs and search indexes.
- [`graphrag/construction/default_extraction.py`](../graphrag/construction/default_extraction.py) selects open-ended extraction.
- [`graphrag/construction/ontology_guided.py`](../graphrag/construction/ontology_guided.py) defines the guided schema and rules.
- [`graphrag/retrieval/answering.py`](../graphrag/retrieval/answering.py) selects retrievers and generates answers.
- [`graphrag/retrieval/retrievers.py`](../graphrag/retrieval/retrievers.py) defines vector, hybrid, and Cypher retrieval.
- [`graphrag/retrieval/agentic.py`](../graphrag/retrieval/agentic.py) defines the agent that chooses search tools.
- [`graphrag/evals/construction.py`](../graphrag/evals/construction.py) evaluates graphs and answers.
- [`graphrag/evals/retrieval.py`](../graphrag/evals/retrieval.py) evaluates retrieved context.
