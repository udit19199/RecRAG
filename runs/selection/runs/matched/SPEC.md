# Matched-budget agent and forced-answer controls (27 Nov 2026)

> No need for repeats. Run option 1 and 2
# option 1 (from the agreed list): "Give the agent the same room as top 100 (~$1.70). Keep the items the agent picked
# and fill the rest up to 100 chunks." option 2: "Force an answer (~$0.40). Rerun with
# 'Insufficient Information' removed as an option, for top 10 and for no chunks."
# "No need for repeats" = no extra repeats of the older single-repeat runs; these controls get
# 3 repeats like the runs they are compared with (agent repeat r feeds AGENT_FILL100 repeat r).

questions: 250
question set: runs/nograph/questions.json comparison_query + temporal_query
AGENT_FILL100 passages: 100
FORCED_TOP10 passages: 10
FORCED_CLOSED passages: 0
repeats: 3
answer model: gpt-6-luna
reasoning effort: medium
forced prompt env: RECRAG_FORCED_ANSWER=1
service tier: default
where: Instabox VM, runs/selection/scripts/vm_matched.sh, 3 lanes in parallel
scoring: answered right (exact match, then Claude Haiku answer judge); no gold-facts judging
compare with: B4_100 (top 100 straight), rag (Normal RAG top 10), CLOSED

## Result (27 Nov 2026; scores.json, score.py)
Answered right, comparison / temporal, mean of 3 repeats:
- Agent's 10 + top-100 fill: 85.6 / 85.6 (top 100 straight 82.7 / 84.8; paired +2.9 [-0.3, 6.7],
  +0.8 [-2.9, 4.5]). Same budget, agent first: no clear gain over top 100.
- Forced answer, top 10: 67.7 / 83.5 (Normal RAG 51.7 / 47.7).
- Forced answer, no chunks: 76.8 / 78.7 (closed book with refusal allowed: 0.5 / 4.5). The model
  answers most of these yes/no questions right from the question alone when it may not refuse
  (always 'Yes': 62.4 / 48.0). It says 'Yes' to 76.5% / 58.4% of them.
Judge: 74 + 7 new Haiku answer items (items/verdicts-answers-021, 022). Cost fields were not recorded
by answer_saved for these prepared runs (question_cost None), as for WRONG10 and CLOSED.
AGENT_FILL100 results/retrieval are committed as .jsonl.gz (51 MB / 26 MB raw each).
