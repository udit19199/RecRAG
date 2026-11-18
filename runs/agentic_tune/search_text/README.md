# Search text check: question as written vs the agent's own first search (10 Nov 2026)

Question: does the agent's rewritten vector_search text find more gold facts than
searching with the question as written? Comparison questions, 100.

## Sets (each 10 plain passages from the graph's chunk index, `chunk_embeddings`)
- `question/`: the question as written, searched with the same retriever code as the
  agent's vector_search (`_ChunkVectorRetriever`, top 10), database
  `recrag-benchmark-e5315c2b5c9f57e2595cc51c`. 100 embedding calls, no LLM.
- `agent_first_r1..r3/`: the 10 passages the agent's first vector_search returned in the
  10 Nov runs `runs/agentic_tune/graphrag/comparison100{,_rep2,_rep3}` (saved traces).
- Built by `runs/heldout400/scripts/search_text_sets.py`.

## Judge
- Same as `runs/agentic_tune/fact_judge/README.md`: Claude Haiku 5.5 subagents, prompt
  version `2026-11-10.haiku.2`, the same frozen facts (`judge/claims/` copied from
  `fact_judge/claims/`), quote verified in code. Judge code sha256
  `ee3eb6163dc2680c95bb41cfa95b2618d8167dca2d4374975e9ada0758180281`.
- Subagent task prompt: as in `fact_judge/AGENT_PROMPTS.md`, with paths under
  `runs/agentic_tune/search_text/judge/judge/`.
- 400 (set, question) pairs, 394 distinct, 71 files. Token use per file: `judge/usage.jsonl`
  (4,494,439 subagent tokens in total).
- Statuses: 1,463 stated, 1,433 missing, 38 contradicted; 0 stated with an unverified quote.
- Two subagents noted borderline calls (verdict-057: one fact marked missing; verdict-068:
  one fact marked stated from "set to launch sometime next year"). Left as judged.

## Result (`comparison.json`, script `runs/heldout400/scripts/search_text_compare.py`)
Gold facts found, agent first search / question; cells [both, only agent, only question, neither]; exact McNemar.

| | r1 | r2 | r3 |
|---|---|---|---|
| >=50% | 56 / 50 [47, 9, 3, 41] p=.146 | 54 / 50 [45, 9, 5, 41] p=.424 | 54 / 50 [44, 10, 6, 40] p=.454 |
| >=75% | 31 / 26 [21, 10, 5, 64] p=.302 | 27 / 26 [20, 7, 6, 67] p=1.0 | 26 / 26 [19, 7, 7, 67] p=1.0 |
| 100% | 21 / 19 [13, 8, 6, 73] p=.791 | 18 / 19 [13, 5, 6, 76] p=1.0 | 17 / 19 [11, 6, 8, 75] p=.791 |
