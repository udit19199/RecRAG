# Graph rows carry graph facts only: 38-question check (13 Nov 2026)

Asked for (coordinating thread for Udit): hide chunk text from the graph tool,
cut lists at 20, keep the agent's picks; rerun the 38 swinging comparison
questions, 3 repeats, same Haiku judges; compare with both earlier runs.
Spend limit $2; spent $0.336 (114 rows, mean $0.00295 per question).

- Code: commit 16cb0ff (Agentic GraphRAG final version); prompts and settings
  verbatim in runs/final500/FROZEN.md. Change on top of agent picks: see
  "What changed" there.
- Runs: rep1, rep2, rep3 (38 rows each, row count checked); 1-question pilot in pilot/.
- Retrieval judge: Claude Haiku subagents, evals/haiku_fact_judge.py, prompt
  2026-11-10.haiku.2, frozen facts; files items-208..227 in ../fact_judge/judge
  (112 new tasks; usage rows appended to ../fact_judge/usage.jsonl, 1,260,922
  subagent tokens). verdict-216: the judge typed one item id with one wrong
  character (6d3c4e... for 6d3c6e...); the id was corrected, nothing else.
- Answer judge: exact match, then Claude Haiku; every non-exact answer already
  had a verdict (keyed by question + answer), so no new answer judging.
- Results: compare.json (compare.py). Context tokens to the answer model:
  median 2,270, largest 3,439 (agent picks: largest 29,041).
- Picks: 106 of 114 runs ended with valid IDs; the rest used the fallback rule.
