# Frozen: Agentic GraphRAG final version (13 Nov 2026)

Frozen before any question of the final 500-question study ran. No change after that.

- Code commit (Agentic GraphRAG): `16cb0ff7ab22e6972320e02698d0db9cd9fe8fba`
- Study plumbing commit (flex builds, question files, NQ shared corpus, judge code): `b058b3d9464e1772f4d83c44524d22327896295c`
- Working tree code files at freeze (sha256), all equal to commit `b058b3d`:
  - `graphrag_retrieval.py` `171561462c58b1b628cf5e5cd927a97b5886d390381eb4e69824d8d121280b7c`
  - `graphrag_construction.py` `17a6e225ce208872f8b964ddd443f4a256930d7d52a44b513805fd67d3f44fad`
  - `benchmark.py` `994258948c5e8aef3d2ca1249bd8bc90f7d1e712a75624d11db8423e64e8fe8c`
  - `cost.py` `a996985599154277a4dd63664db1a9b0b582b5438b196cd7040f319e92e4c45a`
  - `dataset.py` `3aefdb69ebedb494bda2d9ffbf7b8f3660132bdc325ce72b37da4d79e620d426`
  - `rag_retrieval.py` `6cd5feaf1baf7bba50cd0996019aec679884bd9859573c0f33d28d8de75515ab`
  - `raptor_retrieval.py` `7bb5655d311f4ee4d75b6fcad592041e7e318913c3cedfe3d203981a5b9bea7d`
  - `hipporag_retrieval.py` `5b5013c4f74528fe2558f631c2b500b3d3e6ef3391b730e30d40102b39c9ece2`
  - `evals/qa_metrics.py` `10e0a1035e56ac9bee82334a7b0a2025e0ea7a3029d0dbd6436ad5afe2f94cbe`
  - `config.toml` `42cff53feade089ac387aeb3af5a65ab8dd6d3205f9dc746aa932eb12a8a17d2`
- Diffs against the previous commit: `frozen_agentic.diff`, `frozen_plumbing.diff`.

## What changed in this version (and why)

1. The agent picks the evidence. Every search result line carries a short ID (`E` + 6 hex characters of the sha256 of the result text). When it stops searching, the agent replies with the IDs the answer needs, at most 10, from either tool. Those items go to the answer model first, and leftover slots are filled by the old rule: vector passages by score, then graph rows. Before this, code kept the 10 highest-scoring vector passages, and graph rows reached the answer model in 0 of 300 comparison runs.
2. Graph rows carry graph facts only. `Chunk.text` is removed from the schema the Cypher writer sees. A generated query that returns `.text` is refused, and the agent sees the refusal. `text` and `embedding` keys are dropped from rows, and lists in a row keep at most 20 items, plus a "(+N more)" marker.

Evidence: 38 comparison questions that swung most in the 10 Nov repeats, 3 repeats each, Haiku judges (`runs/agentic_tune/graph_facts/compare.json`).

## Settings

- Answer and agent model: gpt-6-luna, reasoning effort medium (`config.toml` `[llm]`)
- Embeddings: text-embedding-3-small, 1536 dimensions
- Items to the answer model: 10; vector_search returns 10 passages; Cypher rows at most 10; entities per passage 15; list items per graph row 20
- Agent model calls per question: at most 5 (`ModelCallLimitMiddleware(run_limit=5)`)
- Chunks: 256 tokens, overlap 20
- Ontology: news (MultiHop-RAG), biography (HotpotQA), general (Natural Questions)

## config.toml
````toml
# Non-sensitive settings for local research runs.

[embedding]
model = "text-embedding-3-small"
dimensions = 1536
timeout = 120

[llm]
model = "gpt-6-luna"
reasoning_effort = "medium"
timeout = 120

[graphrag]
namespace = "graphrag"
timeout = 30
index_timeout = 600
extraction_concurrency = 20
ontology = "dataset"

[graphrag.graph]
uri = "neo4j://localhost:7687"

[graphrag.retrieval]
timeout = 10

[hipporag]
# Includes reasoning tokens, unlike upstream's small text-only budgets.
max_output_tokens = 16384

[pricing]
# Standard direct API USD per million tokens. Question cost excludes indexing.
source = "https://developers.openai.com/api/docs/pricing"
verified_date = "2026-10-04"
llm_model = "gpt-6-luna"
input_per_million = 0.10
cached_input_per_million = 0.01
cache_write_per_million = 0.125
output_per_million = 0.50
embedding_model = "text-embedding-3-small"
embedding_per_million = 0.02
````

## Agent instruction (verbatim)
````text
Use vector_search to find passages about a topic. Use cypher_search to look up specific named entities and how they are connected. First call vector_search with the question. Then call cypher_search once about the named entities in that evidence. After that, either call the next best tool with a refined query or make no tool call if the evidence is sufficient. Do not repeat a search that returned no new evidence. Every result line starts with an ID such as [E1a2b3c]. When you stop searching, do not answer the question. Reply only with the IDs of the results the answer needs, at most 10, most important first, one per line. Results from either tool count. Pick evidence for every article, person or part the question asks about.
````

## Tools given to the agent (verbatim source)
````python
tools=[
            vector.convert_to_tool(
                name="vector_search",
                description=(
                    "Find source text passages similar in meaning to the query. "
                    "Each result has the passage, its page title, the entities it "
                    "mentions, and a few graph facts about them."
                ),
                parameter_descriptions={
                    "query_text": "What to search for, in plain words.",
                },
            ),
            cypher.convert_to_tool(
                name="cypher_search",
                description=(
                    "Query the knowledge graph extracted from the source text: "
                    f"{graph_contents(ontology)} and the relationships between them. "
                    "Generates and runs a read-only Cypher query. Returns graph "
                    "facts (names, titles, dates, relationships), never passage text."
                ),
                parameter_descriptions={
                    "query_text": (
                        "A short lookup about specific named entities, such as "
                        "'what is connected to <name>' or 'which articles "
                        "mention <name>'."
                    ),
                },
            ),
        ],
        ````

## Cypher writer prompt (verbatim template)
````text

Task: Generate one read-only Cypher statement for a Neo4j 5 graph database from a user input.

Schema:
{schema}

Examples:
{examples}

Rules:
- Use only node labels, relationship types, and properties listed in the schema.
- Every extracted entity also has the label __Entity__ with a name and optional aliases.
- Match names case-insensitively: compare toLower(n.name), and each alias in coalesce(n.aliases, []), with a lowercase string.
- Return graph facts only: names, titles, dates, short properties and relationship types. Never return passage text; vector_search finds passages.
- Write alternative relationship types as [:A|B].
- Return names and short properties, never embedding, and end with LIMIT 10 or less.

Input:
{query_text}

Return only the Cypher statement, with no triple backticks or other text.

Cypher query:
````

## Cypher examples (verbatim)
````text
USER INPUT: 'What is connected to Ada Lovelace?' QUERY: MATCH (n:__Entity__)-[r]-(m:__Entity__) WHERE (toLower(n.name) CONTAINS 'ada lovelace' OR any(alias IN coalesce(n.aliases, []) WHERE toLower(alias) CONTAINS 'ada lovelace')) RETURN n.name AS entity, type(r) AS relationship, m.name AS related LIMIT 10
USER INPUT: 'Which pages mention Acme Corp?' QUERY: MATCH (n:__Entity__)-[:FROM_CHUNK]->(:Chunk)-[:FROM_DOCUMENT]->(d:Document) WHERE (toLower(n.name) CONTAINS 'acme corp' OR any(alias IN coalesce(n.aliases, []) WHERE toLower(alias) CONTAINS 'acme corp')) RETURN DISTINCT n.name AS entity, d.title AS page LIMIT 10
USER INPUT: 'Which organizations is Jane Doe part of?' QUERY: MATCH (p:Person)-[:MEMBER_OF|WORKS_FOR]->(o:Organization) WHERE (toLower(p.name) CONTAINS 'jane doe' OR any(alias IN coalesce(p.aliases, []) WHERE toLower(alias) CONTAINS 'jane doe')) RETURN p.name AS person, o.name AS organization LIMIT 10
````

## Answer prompts (verbatim)
System, MultiHop-RAG:
````text
You are an AI assistant that answers questions using the provided context. Always base your answers strictly on the given context and provide clear, concise, and accurate responses. Follow these rules when answering: 1. If the question requires inference (e.g., identifying an entity, extracting information, or summarizing context), provide the answer as the relevant entity or a short phrase. 2. If the question cannot be answered with the provided context, respond ONLY with 'Insufficient Information'. 3. For binary questions (e.g., those starting with 'Does', 'Was', 'Is', etc.), respond ONLY with 'Yes' or 'No'.
````
System, HotpotQA and Natural Questions:
````text
You are an AI assistant that answers questions using the provided context. Always base your answers only on the given context and provide clear, concise, and accurate responses. You must give ONLY the answer as a single word or phrase. Do not give additional information. If the provided information is insufficient to answer the question, respond 'Insufficient Information'.
````
User message (`evals.qa_metrics.benchmark_prompt`):
````python
def benchmark_prompt(context: str, question: str) -> str:
    return (
        f"\n\nContext:\n{context}\n\nQuestion:\n{question}? "
        "Please just give a short answer without explanation.\n\nAnswer:\n"
    )
````

## Graph extraction prompt (construction, verbatim)
````text

You are a top-tier algorithm designed for extracting
information in structured formats to build a knowledge graph.

Extract the entities (nodes) and specify their type from the following text.
Also extract the relationships between these nodes.

Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "Person", "properties": {{"name": "John"}} }}],
"relationships": [{{"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {{"since": "2024-08-01"}} }}] }}

Use only the following node and relationship types (if provided):
{schema}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and
the relationship direction.

Make sure you adhere to the following rules to produce valid JSON objects:
- Do not return any additional information other than the JSON in it.
- Omit any backticks around the JSON - simply output the JSON on its own.
- The JSON object must not wrapped into a list - it is its own JSON object.
- Property names must be enclosed in double quotes

Examples:
{examples}

Input text:

{text}


Rules:
- Every node needs a name.
- Fill a field only when text states it. Omit it when not stated.
- Use Thing only when no narrower kind fits.
- Link name is UPPER_SNAKE, short, from text.
- One fact per link. No facts outside text.

````
