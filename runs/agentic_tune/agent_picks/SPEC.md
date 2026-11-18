# Agent picks the evidence: rerun on the swinging questions

> Without running anything find the root cause and once you do find the root cause run it on the questions that had the variable performance

method: Agentic GraphRAG, change of 13 Nov 2026 (agent lists IDs of the results the answer needs; those go to the answer model first, rest of the 10 slots filled by the old rule)
dataset: MultiHop-RAG held-out 400, comparison_query
questions: 38 (record_ids in questions.json; swing >= 0.25 in gold facts found across the 3 repeats of the 10 Nov run)
repeats: 3
llm: gpt-6-luna, reasoning effort medium
top_k: 10
graph: existing Neo4j database recrag-benchmark-e5315c2b5c9f57e2595cc51c (no rebuild)
retrieval judge: Claude Haiku subagents, evals/haiku_fact_judge.py, prompt 2026-11-10.haiku.2, frozen facts from runs/agentic_tune/fact_judge/claims
answer judge: exact match, then Claude Haiku subagent for non-exact answers
outputs: runs/agentic_tune/agent_picks/rep1, rep2, rep3 (each 38 results rows)
