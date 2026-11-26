# No-agent selection baselines (19 Nov 2026)

Udit's words (19 Nov 2026):
> Well can we do it for less
> No need for new questions

Question: does picking evidence help without an agent? If an LLM picking from
Normal RAG's top 50 gets the same gain, the gain comes from selection, not from
the agent or the graph.

## Variants (Normal RAG index, never rebuilt; 250 MultiHop-RAG comparison + temporal questions of runs/nograph/questions.json; 3 repeats)

| variant | candidates | who keeps 10 |
|---|---|---|
| B0 Normal RAG (reused: runs/final500) | top 10 | none |
| B1 | top 50 by vector score | cross-encoder BAAI/bge-reranker-large (benchmark.py --rerank, candidates 50) |
| B2 | top 50 by vector score | GPT-6 Luna picks up to 10 by ID |
| B3 | GPT-6 Luna writes up to 3 queries; question top 20 + each query top 10, pooled, at most 50 | GPT-6 Luna picks up to 10 by ID |

Picks fewer than 10 are filled in retrieval order; no valid ID means the first 10.

## Prompts (also in each run folder's variant.json)

Selection (system message), the agent's own picking sentences from graphrag_retrieval.AGENT_INSTRUCTION:
> Every result line starts with an ID such as [E1a2b3c]. Do not answer the question. Reply only with the IDs of the results the answer needs, at most 10, most important first, one per line. Pick evidence for every article, person or part the question asks about.

User message: `Question: <question>\n\nResults:\n[<id>] <passage, whitespace collapsed>` one per line. IDs are graphrag_retrieval.evidence_id of the passage.

B3 query writing (system message; user message is the question):
> Write search queries that find source passages for the question: one query per article, person or part the question asks about, at most 3. Do not answer the question. Reply only with the queries, one per line.

## Models and settings

gpt-6-luna, reasoning effort medium, for query writing, selection and the answer; text-embedding-3-small; the answer step is benchmark.answer_saved with the Normal RAG answer prompt. Every call is metered into the question's cost.

## Scoring (same as runs/final500)

answered_right: exact match, then the Claude Haiku answer judge, blind to method.
gold_facts: Claude Haiku fact judge on the frozen runs/final500 claims.
Per question type, each repeat.

## Status

19 Nov: script written (scripts/run.py), not yet run. Blocked: the OpenAI key in .env has expired (401).
Estimated cost: B1 about $0.25; B2 about $1.5; B3 about $2; total about $4 for 3 repeats.

## B4: top k straight to the answer, no picking (24 Nov 2026)

Bhanu Swami's words (WhatsApp, 24 Nov 2026, passed on by Udit):
> Normal RAG's top 50 by vector score, the same 50 candidates as the "LLM picks" setup, all sent to the answer model in retrieval order. No picking call.
> While keeping identical: the same 250 comparison and temporal questions, 3 repeats, GPT-6 Luna, the same answer prompt, and Haiku scoring.
> A size curve: also run top 20 and top 30 direct.
> A ceiling check, with no model calls: using the same text-match check as the evidence funnel, count how often all the gold evidence is in the top 10 versus the top 50.
> Optional order check: top 50 in reversed or shuffled order.

variants: B4 (top 50), B4_20, B4_30, B4_rev (top 50 reversed), B4_shuf (top 50 shuffled, seed sha256(question), same order in every repeat)
questions: 250 (runs/nograph/questions.json, 125 comparison + 125 temporal)
repeats: 3
index: runs/rag/indexes/07062033… (Normal RAG, never rebuilt), llama-index store, same retrieval call as B2
answer model: gpt-6-luna, reasoning effort medium, Normal RAG answer prompt (benchmark.py)
picking call: none
scoring: answered right (exact match, then Claude Haiku answer judge); gold facts found (Claude Haiku fact judge, frozen runs/final500 claims)
ceiling: runs/selection/scripts/ceiling.py, text match of evals/retrieval_quality.py (locate_gold, ranked_items, any_overlap), top 10/20/30/50, no model calls
where: Instabox VM, runs/selection/scripts/vm_b4.sh, 3 lanes in parallel
estimated cost: about $4 at list price

## B4_100 and WRONG10 (25 Nov 2026)

Udit's words (25 Nov 2026):
> Run all of them.
(after the list of checks: a top-100 size step, and a floor where the answer model gets no relevant evidence; his follow-up: "we need to get these done asap" and "launch this stuff in the background")

B4_100: Normal RAG's top 100 by vector score, all 100 sent to the answer model in retrieval order; as B4 in everything else (same 250 questions, 3 repeats, GPT-6 Luna, same answer prompt, Haiku scoring: answered right and gold facts found on the frozen final500 claims).
WRONG10: the floor. Each question is answered from the Normal RAG top 10 of another question of the same type; a donor is accepted only if none of its 10 chunks lies on an article that holds a gold sentence of the question and none overlaps a gold sentence. Donor map: seed 20261125, fixed across repeats (runs/selection/paired/wrong10_donors.json). No retrieval call; answer step = benchmark.answer_saved, same prompt. Scoring: answered right only (Haiku answer judge). No gold-facts judging: by construction no gold sentence is in the context.

variants: B4_100, WRONG10
questions: 250 (runs/nograph/questions.json, 125 comparison + 125 temporal)
repeats: 3
index: runs/rag/indexes/07062033… (Normal RAG, never rebuilt)
answer model: gpt-6-luna, reasoning effort medium, Normal RAG answer prompt (benchmark.py)
scoring: answered right (exact match, then Claude Haiku answer judge); gold facts found for B4_100 only (Claude Haiku fact judge, frozen runs/final500 claims)
where: Instabox VM, runs/selection/scripts/vm_new.sh, 2 lanes in parallel
estimated cost: about $5 at list price (B4_100 about 33,000 input tokens per question; WRONG10 about 3,000)

CLOSED (25 Nov 2026, Udit: "you said we need a closed box run as well"): closed book. The answer model gets the question and no chunks at all (empty context), same answer prompt and settings; 250 questions, 3 repeats; answered right only (Haiku answer judge). Run: runs/selection/scripts/vm_closed.sh on the Instabox VM (wrong10.py with CLOSED=1). With the Normal RAG prompt the model is told to say 'Insufficient Information' when the context cannot answer, so this run shows how much it answers from the question alone; WRONG10 shows the same with unrelated chunks in the context.

## B0/B2/B4 on inference and null questions (25 Nov 2026)

The finding "the gain is mostly fewer 'Insufficient Information' answers" raises the question whether more chunks make the model answer unanswerable (null) questions wrongly. Same three setups on the inference and null questions of the latest run.

variants: B0 (Normal RAG top 10 straight, non-Milvus), B2 (LLM picks 10 of top 50), B4 (top 50 straight)
questions: 250 (runs/selection/questions_infnull.json: the 125 inference_query + 125 null_query record indices of the latest run, from runs/final500/questions.json multihop_rag; RECRAG_QUESTIONS env var in run.py)
repeats: 3
index: runs/rag/indexes/07062033… (Normal RAG, never rebuilt)
answer model: gpt-6-luna, reasoning effort medium, Normal RAG answer prompt (benchmark.py)
scoring: answered right only (exact match, then Claude Haiku answer judge; on null questions right = said 'Insufficient Information'). No gold-fact judging.
where: Instabox VM, runs/selection/scripts/vm_infnull.sh, 3 lanes in parallel

## B4_200 and B2_100, plus the B2 inference/null retry (25 Nov 2026)

Top 100 straight beats every setup including Agentic Hybrid RAG, so two follow-ups: does the curve keep rising past 100 (B4_200: Normal RAG top 200 straight, as B4 in everything else), and does picking from a bigger pool close the gap (B2_100: the B2 picking call on the top 100 candidates, up to 10 picked). Also a retry of the B2 inference/null run, whose first attempt died on a transient API error mid r1 (jobs/INFNULL_lanes.log); B0 and B4 finished there.

variants: B4_200 (top 200 straight), B2_100 (LLM picks up to 10 of top 100, same picking prompt and code as B2), B2 retry on questions_infnull.json
questions: 250 each (runs/nograph/questions.json for B4_200/B2_100; runs/selection/questions_infnull.json for the B2 retry)
repeats: 3
index: runs/rag/indexes/07062033… (Normal RAG, never rebuilt)
answer model: gpt-6-luna, reasoning effort medium, Normal RAG answer prompt (benchmark.py); B2_100 reuses the B2 picking prompt (SELECT_INSTRUCTION) word for word
scoring: answered right (exact match, then Claude Haiku answer judge); gold facts found for B2_100 (Claude Haiku fact judge, frozen runs/final500 claims); B4_200 answered right only ("not scored" for facts)
where: Instabox VM, runs/selection/scripts/vm_next.sh, 3 lanes in parallel

## The selection setups on HotpotQA and Natural Questions, at the flex price (25 Nov 2026)

Udit's words (25 Nov 2026):
> We need to complete the results for other datasets to man we have so many approaches but they have not been tested on other datasets
> Bro we cannot spend that much are you dumb. We have 10 dollars at max to verify everything. Why would we run everything from scratch we need to run the extra stuff
> We need to use batch pricing
> No retries this time.

Question: do the MultiHop-RAG findings (more chunks straight beat picking past about 100; the LLM picker beats sending 50 straight; the reranker trails) hold on HotpotQA and Natural Questions? Only the missing setups run; Normal RAG top 10, Agentic Hybrid RAG, RAPTOR, HippoRAG and closed book on these datasets are reused from runs/final500.

datasets: hotpotqa, natural_questions
questions: 500 per dataset = the runs/final500 questions (runs/final500/questions.json; HotpotQA 250 bridge + 250 comparison from runs/final500/hotpotqa/questions_rows.json, record indices 0-499; Natural Questions validation rows 0-499), loaded exactly as runs/final500 loaded them
variants: B4_20, B4_30, B4_50 (= B4), B4_100, B4_200 (top k straight); B2 (LLM picks up to 10 of top 50); B2_100 (LLM picks up to 10 of top 100); B1 (BAAI/bge-reranker-large keeps 10 of top 50)
repeats: 1
index: the runs/final500 Normal RAG index of each dataset, never rebuilt: runs/final500/hotpotqa/rag/indexes/5dff86f578fac0ee98ceddd583c63294ee8a51349b436bda8d4f81883734fa73 and runs/final500/natural_questions/rag/indexes/f1fbc2708604669b3504eb39ea7c8172c11217d3f05513532b546e9be650e045
index check: the first 10 passages of every new top-k retrieval equal runs/final500 Normal RAG repeat 1's 10 passages for the same question
answer model: gpt-6-luna, reasoning effort medium, Normal RAG answer prompt (benchmark.py); picking prompt SELECT_INSTRUCTION word for word
service tier: flex (Batch API price) for every picking and answer call, RECRAG_QUESTION_SERVICE_TIER=flex; a flex request the API keeps refusing (429) is not resent at the default tier: the question fails
retries: none. A failed run or question is reported, not rerun.
budget: $10 list price for everything in this section, OpenAI only; stop before any launch that would pass it
also, MultiHop-RAG: B4_200_rev = top 200 in reversed order (best-scoring last), the default 250 comparison and temporal questions (runs/nograph/questions.json), 1 repeat, flex
scoring: answered right (exact match, then the Claude Haiku answer judge, evals/subagent_judge.py). HotpotQA is reported pooled (bridge + comparison), Natural Questions as one set.
not run: Agentic Hybrid RAG with the graph off on these datasets (needs the Neo4j Desktop graphs on the Mac; waits for Udit); bge-reranker-v2-m3 (chunks average 122-227 tokens, under bge-reranker-large's 512-token limit, so the old reranker is not cut short)
where: Instabox VM, one dataset at a time (VM /data has about 1.4 GB free): copy that dataset's index in, run, copy the run folders back to the Mac, remove them and the index copy from the VM
lost: the HotpotQA B4_200 run completed (500 questions) but its run folder was lost during copy-back to the Mac (a directory transfer failed past the 32 MiB limit and the VM copy was removed); not a model or API failure, and under the no-retries rule it is not rerun

not run: B1 was not run to completion on HotpotQA or Natural Questions — bge-reranker-large on the VM CPU was too slow (~4 min/question; host CPU steal ~60-70% for most of the run window made even thread-pinned scoring ~20+ h per dataset). HotpotQA B1 reached 76/500 scored-and-cached pairs before the stop (cache retained in runs/selection/cache, not committed); the NQ B1 lane was stopped at 1/500. Recorded per the no-retries rule; not a model or API failure.

Results (1 repeat, flex on the answer calls; answered right = exact match, then the Claude Haiku judge):

hotpotqa (500 questions, bridge + comparison pooled; B4_200 lost during copy-back, B1 not run):
  B4_20  87.8% ($0.0702)   B4_30 87.6% ($0.1014)   B4      87.4% ($0.1637)
  B4_100 88.8% ($0.3180)   B2    87.0% ($0.2109)   B2_100  88.2% ($0.3702)
  0 empty answers in every variant; 0 questions failed (transient flex 429s retried in-run, not counted as failures).

natural_questions (500 questions, one set; B1 not run):
  B4_20  57.6% ($0.1224)   B4_30 60.0% ($0.1780)   B4      59.0% ($0.2883)
  B4_100 59.0% ($0.5643)   B4_200 59.2% ($1.1172)  B2      59.4% ($0.3688)   B2_100  59.4% ($0.6525)
  0 empty answers, 0 failed questions in every variant.

multihop_rag B4_200_rev (250 comparison + temporal questions, top 200 reversed, best last):
  comparison 87.2%, temporal 85.6% ($0.5667) — vs B4_200 repeat 1 on the same questions:
  comparison 86.4%, temporal 83.2%. Reversing the order did not lower the score.

List-price total for the full dataset runs above: about $5.1 (well under the $10 cap; every answer
call carried service_tier "flex", embeddings calls have no flex tier and are charged "default").

## MultiHop-RAG inference + null chunk counts, and the HotpotQA top-200 rerun (26 Nov 2026)

Udit's words (26 Nov 2026):
> 1 repeat, and no need for checking order like reversed and stuff
> (LLM picks 10 of 50/100) are not to be run again nor should they be mentioned again — "Remove it everywhere"
> (HotpotQA top 200 rerun) "Yes, rerun"

Two runs, both at the flex price, 1 repeat, no retries, no default-tier fallback:

- MultiHop-RAG inference + null (runs/selection/questions_infnull.json, 250 questions):
  B4_20, B4_30, B4_100, B4_200 straight, and WRONG10 (the wrong-chunks floor; donor
  map built for inference/null the same way as wrong10_donors.json — same question
  type, no gold-article overlap — saved as runs/selection/paired/wrong10_infnull_donors.json,
  the comparison/temporal map is not overwritten). No order checks, no B2/B2_100, no reranker.
- HotpotQA B4_200 straight, the 500 questions, 1 repeat: the rerun Udit approved of the
  run lost during copy-back.

Same index, answer model, prompt, service tier and index check (first 10 passages ==
Normal RAG repeat 1) as the sections above. Max 3 lanes on the VM. Budget for this
batch: about $2.50 list price.
