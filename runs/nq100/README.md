# Natural Questions, first 100 validation questions (planned 11 Nov 2026)

Normal RAG vs Agentic GraphRAG (10 Nov version, with the 10 Nov Cypher prompt fix
and the 11 Nov hand-off change below). Rules below were fixed before any question was run.

## Agentic GraphRAG hand-off change (11 Nov 2026, user decision)
The answer model now gets the 10 best vector passages AND the cypher_search rows:
at most 10 rows, each cut at 2,000 characters, a row dropped when a passage it returns
is already passed (`graphrag_retrieval.py`, `_repeats_passage`). Before, rows only
filled empty slots, which never happened (0 of 300 MultiHop-RAG comparison runs), so
graph results never reached the answer. The answer model can now read more text than
Normal RAG's 10 passages; report the text size per method. Results from this version
must not be pooled with the 10 Nov MultiHop-RAG runs.

## Data
- `google-research-datasets/natural_questions`, `default` config, validation split,
  parquet files only (`dataset.py` downloads only `default/validation-*.parquet`).
- First 100 rows in order (indices 0-99), no filtering. Frozen in `questions.json`
  (index, NQ id, page title, chunks, gold answer, has evidence).
- Each question has its own Wikipedia page (100 different pages, 4,877 chunks;
  median 35, largest 324). Each method builds one index per page; Agentic GraphRAG
  builds one Neo4j database per page with the general ontology.

## Settings that are not method changes
- Warm-up off (`RECRAG_WARMUP=0`): with one index per question the warm-up would run
  100 times, unmetered, with a news question. Latency is not reported for this run.
- Neo4j `dbms.max_databases` raised from 100 to 200 (Neo4j Desktop instance
  `dbms-a040ad66…`, conf backup in `~/Documents/HPE/RecRAG/backups/`).

## Scoring rules, fixed before running
- Answered right: exact match, else the Haiku answer judge (`evals/subagent_judge.py`,
  unchanged).
- Questions whose gold is "No answer" (32 of 100): the model's "Insufficient
  Information" counts as right; any other answer counts as wrong. (The shared judge
  never judges abstentions, so this rule is applied in the NQ report script.)
- Gold answers that are a long paragraph (19 of 100): decided by the Haiku answer judge.
- Gold facts found (>=50% / >=75% / 100%): only the 68 questions with an evidence
  paragraph (`has_evidence`). Some evidence is a flattened table with many facts, so
  100% is very strict; all three levels are reported, none dropped.
- Report by method, per repeat, as both / only-A / only-B / neither, with exact McNemar.

## Known limits (state when reporting)
- Each question searches only its own page, unlike MultiHop-RAG's shared corpus;
  results are not comparable with the MultiHop-RAG numbers.
- The Cypher prompt fix of 10 Nov runs here for the first time; NQ uses the general
  Cypher examples.

## Running plan
1. Normal RAG, all 100.
2. Agentic GraphRAG pilot: first 5 questions (`NQ_LIMIT=5`), then compare the metered
   build cost in `build_cost.json` with the estimate (about $0.001 per chunk).
3. Agentic GraphRAG, the remaining 95 (resume with the pilot's `retrieval.jsonl`).
   Watch Neo4j memory (4 GB heap, about 107 databases) and stop if errors appear.
4. Merge resumed folders; Haiku answer judge; Haiku fact judge (claims from evidence).

Estimated OpenAI cost: about $5.00 graph builds (one time) + about $0.30 questions.

## Run 1 results (11 Nov 2026)
Runs: Normal RAG `rag/20261110T200339107433Z`; Agentic GraphRAG `graphrag/run1`
(merged from three folders, see `graphrag/run1/PARTS.md`; answers written once for all 100).

Answered right (`answered_right.json`, `scripts/answered_right.py`), Agentic / Normal RAG,
cells [both, only Agentic, only Normal RAG, neither], exact McNemar:
- all 100: 70 / 68 [66, 4, 2, 28] p=.69
- gold not "No answer" (68): 55 /53 [51, 4, 2, 11] p=.69
- gold "No answer" (32): 15 / 15 [15, 0, 0, 17] p=1.0

Gold facts found, 68 questions (`gold_facts.json`, `scripts/gold_facts.py`):
- >=50%: 54 / 54 [52, 2, 2, 12] p=1.0
- >=75%: 46 / 45 [42, 4, 3, 19] p=1.0
- 100%: 38 / 36 [35, 3, 1, 29] p=.63
- Facts Agentic GraphRAG found only in Cypher rows: 0 of 1,039 found facts.

Judges: Haiku answer judge (`judge/`, 101 items, 2 subagents); Haiku fact judge
(`fact_judge/`, prompt `2026-11-10.haiku.2`, 7 claim files, 24 judge files, 5,512 fact
verdicts, 0 stated with an unverified quote; tokens in `fact_judge/usage.jsonl`, 2.48M).
Note: one question (Titanic passenger table, 863 facts) got 5 facts found; a rough check
suggests up to about 40 more may be in the text. Either way it stays far below 50%, so no
reported level changes.

Text passed to the answer model, median characters: Normal RAG 9,239; Agentic 10,283.

Cost (`build_cost_summary.json`): Agentic GraphRAG graph builds $2.34 metered for 1,737
chunks + $4.22 ESTIMATED for 3,139 chunks (part 2 was stopped before writing its cost
file) = about $6.56 list; questions $0.19. Normal RAG: build $0.02, questions $0.03.

## Version note (11 Nov 2026)
Natural Questions Agentic GraphRAG results are from the 11 Nov version (new Cypher prompt,
rows cut at 2,000 characters, Cypher rows passed to the answer model); user decision to keep
them. The code was then put back to the 10 Nov version (used for all MultiHop-RAG runs);
the 11 Nov changes are in `runs/agentic_tune/prompts/graphrag_retrieval-11nov-changes-REVERTED.diff`.
A partial rerun on the 10 Nov version (`graphrag/20261111T052737673792Z`) was stopped and is not used.
