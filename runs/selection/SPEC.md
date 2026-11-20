# No-agent selection baselines (19 Nov 2026)

Udit's words (19 Nov 2026):
> Well can we do it for less
> No need for new questions

Question: does picking evidence help without an agent? If an LLM picking from
Normal RAG's top 50 gets the same gain, the gain comes from selection, not from
the agent or the graph.

## Variants (Normal RAG index, never rebuilt; 250 MultiHop-RAG comparison + temporal questions of runs/nograph/questions.json; 3 repeats)

| variant | candidates | who keeps 10 |
|---|---|---|
| B0 Normal RAG (reused: runs/final500) | top 10 | none |
| B1 | top 50 by vector score | cross-encoder BAAI/bge-reranker-large (benchmark.py --rerank, candidates 50) |
| B2 | top 50 by vector score | GPT-6 Luna picks up to 10 by ID |
| B3 | GPT-6 Luna writes up to 3 queries; question top 20 + each query top 10, pooled, at most 50 | GPT-6 Luna picks up to 10 by ID |

Picks fewer than 10 are filled in retrieval order; no valid ID means the first 10.

## Prompts (also in each run folder's variant.json)

Selection (system message), the agent's own picking sentences from graphrag_retrieval.AGENT_INSTRUCTION:
> Every result line starts with an ID such as [E1a2b3c]. Do not answer the question. Reply only with the IDs of the results the answer needs, at most 10, most important first, one per line. Pick evidence for every article, person or part the question asks about.

User message: `Question: <question>\n\nResults:\n[<id>] <passage, whitespace collapsed>` one per line. IDs are graphrag_retrieval.evidence_id of the passage.

B3 query writing (system message; user message is the question):
> Write search queries that find source passages for the question: one query per article, person or part the question asks about, at most 3. Do not answer the question. Reply only with the queries, one per line.

## Models and settings

gpt-6-luna, reasoning effort medium, for query writing, selection and the answer; text-embedding-3-small; the answer step is benchmark.answer_saved with the Normal RAG answer prompt. Every call is metered into the question's cost.

## Scoring (same as runs/final500)

answered_right: exact match, then the Claude Haiku answer judge, blind to method.
gold_facts: Claude Haiku fact judge on the frozen runs/final500 claims.
Per question type, each repeat.

## Status

19 Nov: script written (scripts/run.py), not yet run. Blocked: the OpenAI key in .env has expired (401).
Estimated cost: B1 about $0.25; B2 about $1.5; B3 about $2; total about $4 for 3 repeats.
