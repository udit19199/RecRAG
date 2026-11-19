# Re-judge the previous run's Agentic Hybrid RAG gold facts with Claude Haiku

> Re-judge with Haiku (Recommended)
(Udit's choice, 18 Nov 2026, for: the previous run's gold-fact bars were judged by GPT-6 Luna, the latest by Claude Haiku, so they could not be compared.)

method: graphrag (Agentic Hybrid RAG) only; the baselines' previous passages are byte-identical to final500's and already Haiku-judged there
dataset: multihop_rag, runs/heldout400 (previous run), 100 questions per type
types: comparison_query, temporal_query, inference_query (null has no gold facts)
repeats: r1, r2, r3 as in runs/final500/two_run_analysis/compute.py (r2 = r2resume folders + 20261101T231404 folders); latency pass excluded
facts: frozen Haiku facts copied from runs/final500/judge/claims (prompt 2026-11-10.haiku.2)
judge: Claude Haiku subagents (Agent tool, model haiku), one per file, task prompt as in runs/final500/judge/AGENT_PROMPTS.md "judge" with this folder's paths
batch: 60000 characters of retrieved text per file (as final500)
expected (run, question) pairs: 900 (300 per repeat, 3 repeats)
output: runs/heldout400/haiku_rejudge_agentic/judge/verdict-*.json, then `score` -> per_question.jsonl
