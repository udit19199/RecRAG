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
- Scores: a passage the agent run kept carries its saved score. A passage it did not keep has no saved score, so it is scored again with the same Neo4j vector index query the agent's vector_search ran (`db.index.vector.queryNodes('chunk_embeddings', 10, <query embedding>)`, the saved query text re-embedded with text-embedding-3-small, 1536 dims). 11 passages the index query no longer returned were scored by exact cosine, (1 + cos) / 2. A passage found by several searches keeps its best score, as in AgenticToolsRetriever.
- Why not exact cosine for all: the Neo4j index reports approximate scores (up to 0.0016 off exact cosine), so exact cosine reordered near-ties. Re-embedding the same query is also not always bit-identical, so a near-tie can still order differently.
- Check: the agent's own rule, rebuilt this way, must give exactly the saved evidence. Result (21 Nov, runs/handoff/build_check.json): 1,496 of 1,500 rows identical; 4 differ in one fill slot from a near-tie (A3 r1 multihop-rag-952, A3 r2 multihop-rag-161, A1 r1 multihop-rag-2366, A1 r2 multihop-rag-1651). The build stops if more than 15 rows (1%) differ.
- Evidence changed vs the source run: A2_graph_top10 228-234 of 250 rows per repeat; A0_vector_top10 220-229; A3_picks_no_graph_rows 38-43 (so graph rows reached the answer model in about 1 in 6 questions of the full agent).
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
