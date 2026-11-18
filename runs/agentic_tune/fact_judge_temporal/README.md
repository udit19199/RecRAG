# Gold facts found: Claude Haiku judge (temporal questions, 11 Nov 2026)

Same procedure, judge instructions and prompt version (`2026-11-10.haiku.2`) as `../fact_judge/README.md` (comparison).

- Questions: the 100 held-out MultiHop-RAG temporal questions.
- Runs: Normal RAG r1-r3 and Agentic GraphRAG 2 Nov r1-r3 (from `runs/heldout400/manifest.jsonl`, exit 0; r2 is split over 5 resume folders), Agentic GraphRAG 10 Nov `runs/agentic_tune/graphrag/temporal100`, `_rep2`, `_rep3`.
- 900 (run, question) pairs, 190 judge files. Model: `haiku` alias (Claude Haiku 5.5) as Claude Code Agent subagents, one per file. Task prompts: `AGENT_PROMPTS.md`. Token use per file: `usage.jsonl` (judge stage: 10.7M tokens logged for 188 of 190 files; the first session did not log 2 files).
- Timeline: claims 11 Nov 12:00; judge files 000-131 from 12:07 to 12:33 (one session; it stopped there); files 132-189 finished in a second session, 11 Nov afternoon, same prompt.
- Checks: every verdict file has every item id and every fact label exactly once (score stage). Statuses: stated 2,568, missing 2,518, contradicted 34; 19 stated facts had no verified quote and count as not found.
- Tables: `runs/heldout400/scripts/haiku_gold_facts_temporal.py` -> `temporal_final.json` (per repeat, average and lowest-highest, 2-of-3 matrices vs Normal RAG). Shown on slide 05 of the Experiments scene.
- HippoRAG and RAPTOR were not included.

Judge code sha256: `ee3eb6163dc2680c95bb41cfa95b2618d8167dca2d4374975e9ada0758180281` (evals/haiku_fact_judge.py, uncommitted).
