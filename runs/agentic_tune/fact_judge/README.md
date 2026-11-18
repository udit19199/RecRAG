# Gold facts found: Claude Haiku judge (comparison questions, 10 Nov 2026)

The only retrieval measure. For each question: the share of its gold facts that the
retrieved text states. Reported as questions with >=50%, >=75% and 100% of gold
facts found, per method, per repeat, as both / only-A / only-B / neither cells.
No gpt-6-luna is used anywhere in this judge.

## What was judged

- 100 held-out MultiHop-RAG comparison questions (`runs/heldout400/questions.json`, `by_type.comparison_query`).
- Normal RAG r1, r2, r3 and Agentic GraphRAG r1, r2, r3 (run folders from `runs/heldout400/manifest.jsonl`, exit 0 only; r2 Agentic GraphRAG includes its resume folders).
- New Agentic GraphRAG run with the 10 Nov fixes: `runs/agentic_tune/graphrag/comparison100`.
- 700 (run, question) pairs, 500 distinct (identical retrieved text is judged once; Normal RAG retrieval is the same in every repeat).

## Model

Claude Haiku, run as Claude Code subagents (`Agent` tool, `model: haiku`, `subagent_type: general-purpose`).
In this session the `haiku` alias is Claude Haiku 5.5 (`claude-haiku-5-5`). One subagent per input file, no shared context between files.

## Procedure (code: `evals/haiku_fact_judge.py`)

1. **Fact split** (`claims` stage), once per question, frozen for every method.
   Instructions: `CLAIMS_GUIDE`, stored verbatim in `claims/items-*.json` (`guide` field).
   Replies: `claims/verdict-*.json`, as written by the subagent.
   Result: 743 facts for 100 questions.
2. **Fact check** (`judge` stage), per (method, question).
   Instructions: `JUDGE_GUIDE`, prompt version `2026-11-10.haiku.2`, stored verbatim at the top of every `judge/items-*.md` (what the subagent reads) and in `judge/items-*.json` (what the scorer reads).
   Retrieved items are shown exactly as given to the answer model, de-duplicated, and shuffled with a fixed seed per (dataset, question, method) so rank order is hidden.
   Long lines are wrapped at spaces for reading; quote checks collapse whitespace, so wrapping does not change any result.
   Files hold about 60,000 characters of retrieved text each (147 files); a larger single question gets its own file.
   Replies: `judge/verdict-*.json`, as written. Map from (run, question) to task: `judge/links.jsonl`.
3. **Scoring** (`score` stage, in code, no model): a fact counts as found only if the judge says `stated` and at least one quote, after normalization (NFKC, lower case, collapsed whitespace), is a substring of the item it cites. Contradicted, missing, or stated with no verified quote all count as not found.

Subagent task prompts (verbatim): `AGENT_PROMPTS.md`. Per-subagent token use and wall time: `usage.jsonl`.

## Pilot and the one prompt change

- Pilot v1 (`2026-11-10.haiku.1`) on `judge/items-000`: kept in `pilot-v1/`. Quotes were sometimes cut short before the specific (e.g. a quote ending at "$14 billion" for a fact about repaying lenders), and one loose match.
- v2 added one rule: the quotes together must contain the words that state every specific of the fact; a quote that only sets the topic, or stops before the specific, is not evidence. All reported results use v2.

## Checks run during judging

- Every verdict file: ids match, every fact label answered exactly once.
- Quote verification rate on `stated` facts (logged as judging ran).
- False-missing audit: gold sentences that appear word for word in the retrieved text but have a fact marked missing. 7 of 356 at 101 files; the cases inspected were correct calls, because the fact adds words the sentence lacks (e.g. a name for "He", "Argentina's rugby team").
- Files whose token use was below their own size were re-checked; the facts near the end of the file were found, so the file was read in full.

## Known limits

- Fact split by Haiku sometimes resolves a pronoun or adds a gloss (see audit above). It is the same frozen split for every method, so it does not favour one method.
- Old Agentic GraphRAG runs (2 Nov) gave the answer model passages with graph facts and entity lists; the judge sees that same rendering (`evals.retrieval_judge.render_graphrag`). The 10 Nov run gives plain passage text; the judge sees that.

Judge code sha256 at judging time: `35f48f3ac8d6e4661db9fbc391f8b1f5e978912f88fbbfa6056fb5bc4ddd6f34` (`evals/haiku_fact_judge.py`, uncommitted).

## Addition, 10 Nov 2026 (later): new Agentic GraphRAG repeats 2 and 3

- Runs: `runs/agentic_tune/graphrag/comparison100_rep2`, `comparison100_rep3` (each merged from two folders after an interruption; settings identical; answers written once for all 100).
- Exported with `judge --add` (same prompt version `2026-11-10.haiku.2`, same frozen facts): 200 new (run, question) pairs, 197 new tasks, files `items-147` to `items-181`. Earlier files and verdicts unchanged.
- Judge code sha256 after adding `--add`: `ee3eb6163dc2680c95bb41cfa95b2618d8167dca2d4374975e9ada0758180281`. The judging instructions did not change.
- Totals over all 182 files: 5,183 fact verdicts; 2,795 stated, 7 of them with no verified quote (counted as not found); 77 contradicted. False-missing audit: 11 of 787 word-for-word gold sentences.
- Tables: `runs/heldout400/scripts/haiku_comparison_final.py` -> `comparison_final.json`.
