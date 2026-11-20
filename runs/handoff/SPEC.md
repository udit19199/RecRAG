# Hand-off replays: same saved agent searches, different hand-off rule (19 Nov 2026)

Udit's words (19 Nov 2026):
> Well can we do it for less
> No need for new questions

Question: with the agent's searches held fixed, how much of Agentic Hybrid RAG's
gain comes from the agent picking the evidence, and how much from graph rows
reaching the answer model? Only the answer step is rerun; no search is repeated.

## Variants (all on the 250 MultiHop-RAG comparison + temporal questions of runs/nograph/questions.json, repeats 1-3)

| variant | searches from | evidence sent to the answer model |
|---|---|---|
| A3 (reused, not rerun) | runs/final500 Agentic Hybrid RAG | agent's picks, rest filled by vector score (frozen final version) |
| A1 (reused, not rerun) | runs/nograph | agent's picks, vector search only |
| A2_graph_top10 | A3's saved searches | old rule: the 10 pooled vector passages with the highest score, graph rows only in empty slots |
| A0_vector_top10 | A1's saved searches | old rule, as above |
| A3_picks_no_graph_rows | A3's saved searches | agent's picks with graph rows removed, rest filled by vector score |

Repeat r of a replay uses repeat r of its source run.

## How the evidence is rebuilt

- Every tool result is read back from the saved trace (`[E......] content` lines of each successful vector_search / cypher_search call); each ID is checked against `evidence_id(content)`.
- Vector scores are not saved for passages the agent did not keep, so each passage is scored again: cosine of the saved query's embedding (text-embedding-3-small, 1536 dims, re-embedded) and the chunk's stored embedding in Neo4j, reported as (1 + cos) / 2 like the Neo4j cosine index. A passage found by several searches keeps its best score, as in AgenticToolsRetriever.
- Check before writing anything: the agent's own rule, rebuilt this way, must give exactly the saved evidence for all 1,500 rows (A3 and A1, 3 repeats each), and recomputed scores are compared with the saved scores of kept passages (runs/handoff/build_check.json).
- Offline check, 19 Nov (no model calls): for all 1,500 rows, every saved evidence item is found in the trace and the agent's picks rebuilt from its final reply equal the start of the saved evidence. The agent picked on average 3.0 items (A3) and 2.7 (A1); 43 A3 rows and 52 A1 rows had no valid pick.

## Answer step

`benchmark.answer_saved` on each rebuilt retrieval.jsonl: gpt-6-luna, reasoning effort medium, the same system instruction and prompt as the source runs (settings.json copied from the source run).

## Scoring (same as runs/final500)

answered_right: exact match, then the Claude Haiku answer judge, blind to method.
gold_facts: Claude Haiku fact judge on the frozen runs/final500 claims, quote checked word for word.
Per question type, each repeat.

## Status

19 Nov: scripts written (scripts/replay.py). Blocked: the OpenAI key in .env has expired (401); the build step needs it for query embeddings.
Estimated cost: answers about 2,250 calls x $0.0003 = about $0.70; embeddings under $0.01.
