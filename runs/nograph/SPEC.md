# Graph ablation: Agentic Hybrid RAG with no graph (18 Nov 2026)

Udit's words (18 Nov 2026):
> The results of the latest experiments do not have a clear finding that we can claim in a research paper, is there a way to salvage this without doing such a big rerun?
> But this is not robust enough, we would need something to verify it
> This would be really expensive, we need a cheaper way. Why new questions?
> No assumptions here.
> Run it on the mac.

Question: does the knowledge graph cause Agentic Hybrid RAG's gain on MultiHop-RAG
comparison and temporal questions? The latest run (runs/final500) has the full
agent and Normal RAG on these questions; this run adds only the same agent with
the graph removed, so the three can be compared question by question.

## Run settings

dataset: multihop_rag
questions: 250 = runs/final500 multihop_rag all500, comparison_query (125) and temporal_query (125) only, same order (runs/nograph/questions.json)
method: graphrag with RECRAG_AGENT_NO_GRAPH=1
repeats: 3
shards_per_repeat: 2
machine: Mac (Neo4j Desktop DBMS started with a 2 GB heap and 1 GB page cache from a temporary config copy; the saved config is not changed)
index: existing runs/graphrag/indexes MultiHop-RAG graph and chunk index, no rebuild
answer_model: gpt-6-luna
reasoning_effort: medium
embedding_model: text-embedding-3-small
items_to_answer_model: 10
expected_rows: 750 (250 per repeat)

## The one difference from the frozen final version (runs/final500/FROZEN.md)

- The cypher_search tool is removed.
- vector_search results carry no entity names and no graph facts (empty lists); passage text and page title are the same.
- The agent instruction drops only the sentences about cypher_search (`NO_GRAPH_AGENT_INSTRUCTION` in graphrag_retrieval.py); the rest is word for word the same.
- Same model-call limit, same evidence picking, same answer prompt, same 10 items.
- Code otherwise equal to the frozen commit; changes since the freeze are comments and a build-only flex timeout.

## Compared with (reused, not rerun)

- Full agent: runs/final500 multihop_rag graphrag, repeats 1-3, same 250 questions.
- Normal RAG: runs/final500 multihop_rag rag, repeats 1-3, same 250 questions.

## Scoring (same as runs/final500)

answered_right: exact match, then the Claude Haiku answer judge (evals/subagent_judge.py), blind to method
gold_facts: Claude Haiku fact judge on the frozen runs/final500 claims, quote checked word for word
reporting: per question type, each repeat; the difference no-graph minus full agent and no-graph minus Normal RAG, with a 95% paired bootstrap range over questions. No pass/fail margin is set in advance (Udit: "No assumptions here"). Also reported on the 50 new100 questions and without the 38 tuning questions.

## Code at launch

- Base commit: 6745602e4ad9364279643d231881d90fd8f4670e plus the uncommitted no-graph switch in graphrag_retrieval.py (sha256 ee37fc89c80f13f4432ee26a822bc5e227147b8bda0cd5ba0ea5f0973b855bec) and runs/nograph/scripts/.
- Smoke test (2 questions, parked in /tmp/recrag_smoke, not used): 0 cypher calls, vector results with empty entities and facts.
