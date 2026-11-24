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
