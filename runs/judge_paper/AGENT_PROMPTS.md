# Prompts given to the Claude Haiku judge subagents (21 Nov runs: replays, selection baselines, Milvus)

Model: Claude Haiku via the Claude Code Agent tool (`model: haiku`, `subagent_type: general-purpose`), one subagent per file, no context shared between files. Instructions: the `guide` in each items file, the same as runs/final500/judge (gold facts: prompt version 2026-11-10.haiku.2, evals/haiku_fact_judge.py; answers: evals/subagent_judge.JUDGE_GUIDE). Facts: the frozen runs/final500/judge/claims. Tasks are blind to method and run.

Export: scripts/export.py (gold facts; runs listed in runs.json; byte-identical retrieved text reuses the earlier Haiku verdict from runs/final500 or runs/nograph, in items-reuse.json / verdict-reuse.json). Answers: exact match first; only answers not already judged in runs/final500 or runs/nograph go to the judge (answers/items-answers-000.json).

Launched by coordinator subagents (Claude Sonnet) that only launch Haiku subagents with the exact prompts below, at most 3 at a time each, run `scripts/check_verdicts.py judge FIRST LAST`, and relaunch a file once with the same prompt if its verdict is missing or fails the check. Coordinators never write or edit verdicts. A file that fails twice is reported, not fixed.

The first coordinators (Claude Sonnet) stopped on an API auth error after 308 files. The remaining files (81–103, 179–207, 289–311, 387–413) and four files that failed the check (1, 123, 250, 386; first verdicts kept in failed_once/) were relaunched under coordinators on the session's default model (Claude Opus), with the same Haiku judge prompt below. Only the coordinator model changed; every verdict is still written by Haiku.

## judge
Read the file /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/judge_paper/judge/items-N.md completely (it can be long; read it in parts until the line "=== END ITEM" of its last item). It starts with your instructions and then has the items. Follow the instructions exactly for every item. Write your answer to /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/judge_paper/judge/verdict-N.json as a JSON array, as the instructions describe. Quotes must be copied character for character from the item text; a line break in the file counts as a space. Do not modify any other file. Reply only with "done" and the number of items you wrote.

## answers
Read the file /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/judge_paper/answers/items-answers-N.json. Follow the instructions in its "guide" field exactly for every item in its "answers" list. Write your answer to /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/judge_paper/answers/verdicts-answers-N.json as a JSON array, as the guide describes. Do not modify any other file. Reply only with "done" and the number of items you wrote.
