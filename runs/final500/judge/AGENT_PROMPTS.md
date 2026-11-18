# Prompts given to the Claude Haiku judge subagents (final 500-question study)

Model: Claude Haiku via the Claude Code Agent tool (`model: haiku`, `subagent_type: general-purpose`), one subagent per file, with no context shared between files. The judging instructions are the `guide` field in each items file (prompt version `2026-11-10.haiku.2`, `evals/haiku_fact_judge.py`), stored word for word in every items file. The task prompt is below, with N replaced by the file number (three digits, e.g. 007).

The subagents were launched by batch-coordinator subagents. A coordinator only launches Haiku subagents with these exact prompts, runs `scripts/check_verdicts.py` on their output, and relaunches a file once with the same prompt if its verdict is missing or fails the check. It never writes or edits a verdict. A file that fails twice is reported, not fixed.

## claims
Read the file /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/final500/judge/claims/items-N.json. Follow the instructions in its "guide" field exactly for every item in its "items" list. Write your answer to /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/final500/judge/claims/verdict-N.json as the guide describes. Do not modify any other file. Reply only with "done" and the number of items you wrote.

## judge
Read the file /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/final500/judge/judge/items-N.md completely (it can be long; read it in parts until the line "=== END ITEM" of its last item). It starts with your instructions and then has the items. Follow the instructions exactly for every item. Write your answer to /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/final500/judge/judge/verdict-N.json as a JSON array, as the instructions describe. Quotes must be copied character for character from the item text; a line break in the file counts as a space. Do not modify any other file. Reply only with "done" and the number of items you wrote.

## answers (Answered right, after exact match)
Instructions: the `guide` field of each items-answers file (`evals/subagent_judge.JUDGE_GUIDE`, stored word for word in every file). Task prompt, with N replaced by the three-digit file number:

Read the file /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/final500/judge/answers/items-answers-N.json. Follow the instructions in its "guide" field exactly for every item in its "answers" list. Write your answer to /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/final500/judge/answers/verdicts-answers-N.json as a JSON array, as the guide describes. Do not modify any other file. Reply only with "done" and the number of items you wrote.

## audit (Claude Opus check of Haiku gold-fact verdicts)
Model: Claude Opus via the Claude Code Agent tool (`model: opus`, `subagent_type: general-purpose`), one subagent per file, launched directly (no coordinator), no context shared between files. Items come from `scripts/audit_export.py` (same guide, same gold sentence with one fact label, same retrieved items; no method name, no Haiku verdict). Task prompt, with N replaced by the three-digit file number:

Read the file /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/final500/judge/audit/items-audit-N.md completely (it can be long; read it in parts until the line "=== END ITEM" of its last item). It starts with your instructions and then has the items. Follow the instructions exactly for every item. Write your answer to /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/final500/judge/audit/verdict-audit-N.json as a JSON array, as the instructions describe. Quotes must be copied character for character from the item text; a line break in the file counts as a space. Do not modify any other file. Reply only with "done" and the number of items you wrote.
