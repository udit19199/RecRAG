# Graph rows carry graph facts only: rerun on the swinging questions

> Go ahead with your last proposal: hide chunk text from the graph tool so rows carry only graph facts, cap long lists at 20 entities, keep the agent's picks (up to 10 items from either tool). Rerun the same 38 comparison questions, 3 repeats, same Haiku judges, and compare against both earlier runs. Spend limit for this check: $2 OpenAI.

(Coordinating thread, on Udit's behalf, 13 Nov 2026.)

method: Agentic GraphRAG, 13 Nov graph-facts change on top of agent picks
dataset: MultiHop-RAG held-out 400, comparison_query
questions: 38 (same record_ids as runs/agentic_tune/agent_picks/questions.json)
repeats: 3
llm: gpt-6-luna, reasoning effort medium
top_k: 10
cypher_list_limit: 20
graph: existing Neo4j database recrag-benchmark-e5315c2b5c9f57e2595cc51c (no rebuild)
retrieval judge: Claude Haiku subagents, evals/haiku_fact_judge.py, prompt 2026-11-10.haiku.2, frozen facts
answer judge: exact match, then Claude Haiku subagent for non-exact answers
spend limit: 2 USD OpenAI
outputs: runs/agentic_tune/graph_facts/rep1, rep2, rep3 (each 38 results rows)
