# Agent picks the evidence (13 Nov 2026)

Rerun of Agentic GraphRAG on the 38 MultiHop-RAG held-out comparison questions whose
gold-facts-found share moved by 0.25 or more across the 3 repeats of the 10 Nov run.
Spec: `SPEC.md`. Question list and old shares: `questions.json`.

## Root cause found (no runs needed)

The agent only chose which searches to run. After it stopped, code picked the answer
model's 10 items: the 10 vector_search chunks with the highest similarity score; graph
rows only filled empty slots. Vector search always filled all 10, so no graph row reached
the answer model (0 of 300 comparison runs), and the agent's final message was ignored.
In 59 of the 111 old runs missing a named article from the final 10, the graph lookup
had returned that article.

## The change

Every tool result line starts with an ID: "E" + first 6 hex of sha256 of the normalized
text. The agent ends by listing the IDs the answer needs (at most 10, either tool). Those
go to the answer model first, in its order; remaining slots are filled by the old rule. If
no valid ID is named, the old rule picks all 10. The reply and pick count are saved as a
final `select_evidence` step in each trace. Nothing else changed: gpt-6-luna (medium),
top_k 10, forced first vector_search + one cypher_search, same graph
(recrag-benchmark-e5315c2b5c9f57e2595cc51c, no rebuild).

Agent system prompt, verbatim:

> Use vector_search to find passages about a topic. Use cypher_search to look up specific named entities and how they are connected. First call vector_search with the question. Then call cypher_search once about the named entities in that evidence. After that, either call the next best tool with a refined query or make no tool call if the evidence is sufficient. Do not repeat a search that returned no new evidence. Every result line starts with an ID such as [E1a2b3c]. When you stop searching, do not answer the question. Reply only with the IDs of the results the answer needs, at most 10, most important first, one per line. Results from either tool count. Pick evidence for every article, person or part the question asks about.

Selection code as run: `selection_code.py`. graphrag_retrieval.py sha256 at run time:
`a2aa5ae84f31b7ac3eb86b4888e4b6db2a6c6bb1b4e3f92589ddb9b36b7fd769` (uncommitted).

## Runs

- rep1, rep2, rep3 link to folders under `runs/agentic_tune/graphrag/`. 38 rows each.
- rep1 first crashed at start (same folder timestamp as rep2 when started together;
  `pilot/rep1-collision.log`) and was rerun alone. One-question trial: `pilot/reptrial-1q`.
- Agent picks were valid in 107 of 114 runs. 6 runs hit the 5-call limit before picking and
  1 wrote 5-character IDs; those used the old rule.

## Judging

- Gold facts found: Claude Haiku subagents (`haiku` = claude-haiku-5-5),
  `evals/haiku_fact_judge.py` (sha256 `ee3eb6163dc2680c95bb41cfa95b2618d8167dca2d4374975e9ada0758180281`), prompt version
  2026-11-10.haiku.2, frozen facts from `../fact_judge/claims`. Added with
  `judge --add --batch 60000`: files items-182 to items-207 (114 tasks), task prompt as in
  `../fact_judge/AGENT_PROMPTS.md`. Token use: `../fact_judge/usage.jsonl`. A first export
  without --batch made one file per task; it was deleted before any judging.
- Answered right: exact match, then the Haiku answer judge. All 25 non-exact answers already
  had verdicts in earlier judge files (same question and answer text).

## Results

`compare.py` -> `compare.json`: per repeat, in all 3 / none / mixed, and 2-of-3 matrices
(new vs old) for answered right and gold facts found at >=50%, >=75%, 100%.

## Limits

The 38 were picked because the old code was inconsistent on them, so the old "in all 3"
counts are low by selection. The old code was not rerun on them here, and the new code has
not been run on the other 62 comparison questions or on temporal.
