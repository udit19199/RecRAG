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
