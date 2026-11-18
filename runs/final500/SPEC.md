# Final 500-question study: spec (13 Nov 2026)

Udit's words (13 Nov 2026):
> before the next meeting I need to finish a 500 question run so we have done the 400 version now we have to do the 500 one for all three data sets so that we can finish this project as well as the paper
> I want the experiments to be handled and in a way that they don't cost too much and we are still achieving like the best case performance without changing the method a lot
> in a particular type of question say temporal or comparison this is the rag architecture someone should use and this is the rag is the baseline for the comparisons and on top of that there is hippo rag raptor rag and our agentic graph rag
> make sure everything is research grade, nothing in our methodology should be questionable, we want the retrieval judge and everything else to be so good that no one questions or doubts that.
Answers to the plan questions (13 Nov): budget "Up to $55, add Batch API"; MultiHop-RAG "Extend 400 to 500"; Natural Questions "One shared corpus".

## Run settings

datasets: multihop-rag, hotpotqa, natural-questions
questions_per_dataset: 500
methods: rag, raptor, hipporag, agentic_graphrag
repeats: 3
answer_model: gpt-6-luna
reasoning_effort: medium
embedding_model: text-embedding-3-small
deadline: 2026-11-18 (results in the Experiments scene before the HPE meeting)
budget_openai_usd: 55 (list price, all builds and answers together; stop and report if the projection goes over)

### MultiHop-RAG
multihop_questions: the 400 in runs/heldout400/questions.json plus 100 new held-out questions, 25 per type (comparison, temporal, inference, null)
multihop_new_question_rule: never used in any earlier run, tuning run or agentic_tune run; fixed seed; selection script and seed saved
multihop_reuse: Normal RAG, RAPTOR and HippoRAG answers and retrieved text for the 400 are reused from runs/heldout400 (same code, model, indexes); they answer only the 100 new questions
multihop_agentic: Agentic GraphRAG final version answers all 500, 3 repeats
multihop_indexes: existing (runs/{rag,raptor,hipporag,graphrag}/indexes), no rebuild

### HotpotQA
hotpotqa_split: distractor, validation
hotpotqa_selection: 250 bridge + 250 comparison (dataset "type" field), fixed seed 42, selection saved
hotpotqa_corpus: one shared corpus = every paragraph of the 500 questions; one index per method

### Natural Questions
nq_split: default config, validation
nq_selection: rows 0-499 in order, no filtering (rows 0-99 are the NQ 100 pilot)
nq_corpus: one shared corpus = the Wikipedia pages of the 500 questions; one index per method (one Neo4j database)
nq_note: NQ 100 used one index per page; it is the pilot and is not pooled with this run

### Builds
build_batching: OpenAI Batch API for graph-construction and other offline build calls where the call is a plain request; change only the transport, never prompts, models or settings
build_reuse: reuse cached construction results for chunks already built, where the cache key is the exact chunk text

## Frozen before any question runs
- Agentic GraphRAG final version: commit hash, prompts and settings saved to runs/final500/FROZEN.md. No change after the first question of this study runs.
- Question sets, seeds and selection scripts saved.
- Judge prompts saved word for word with version labels.

## Scoring (same for every method, old or new)
answered_right: exact match first, then the Claude Haiku answer judge (evals/subagent_judge.py) on the rest
gold_facts: Claude Haiku fact judge only; no gpt-6-luna anywhere in scoring
gold_fact_claims: extracted once per question from the gold evidence, frozen, shared by every method and repeat
gold_fact_verdicts: a fact counts only when the judge marks it stated with a quote that code finds word for word in the retrieved text
judge_blinding: the judge never sees the method name; items are shuffled across methods
judge_reuse: unchanged methods retrieve identical text each repeat; a verdict is reused only when the retrieved text is byte-identical (checked by hash)
rejudge_400: every MultiHop-RAG retrieval reused from runs/heldout400 is judged again with this same fact judge; no old verdict is mixed in
judge_audit: a stratified random sample of 200 fact verdicts (all methods, all datasets) re-checked by Claude Opus; report agreement and every disagreement
closed_book_baseline: evals/closed_book.py on all 3 datasets, 3 repeats, same answer model (checks whether the model already knows the answers)
context_size: report the median and largest tokens passed to the answer model per method

## Reporting (repo AGENTS.md rules)
- Measures: Answered right; gold facts found at >=50%, >=75%, 100%. Exact labels.
- Split by question type, each repeat separately, questions that fail in all 3 repeats.
- Agentic GraphRAG vs Normal RAG confusion matrices per type and per repeat; the other methods against Normal RAG as well.
- Questions used to tune Agentic GraphRAG (the 38 comparison questions in runs/agentic_tune, and any other question inspected during tuning) are listed; Agentic GraphRAG results are reported both on all questions and on the questions never used in tuning.
- No pooled numbers, no cost or latency in results.
- Results go into the Experiments Excalidraw scene before the work counts as done.

## Outputs
runs/final500/<dataset>/<method>/<repeat>/ with results.jsonl, retrieval.jsonl, settings.json, summary.json
runs/final500/STATUS.md (stage, progress, last error, updated, spend so far)
runs/final500/judge/ (prompts, claims, verdicts, audit)
