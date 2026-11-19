# Prompts given to the Claude Haiku judge subagents (graph ablation gold facts)

Model: Claude Haiku via the Claude Code Agent tool (`model: haiku`, `subagent_type: general-purpose`), one subagent per file, no context shared between files. Instructions: the guide at the top of each items-N.md (prompt version 2026-11-10.haiku.2, evals/haiku_fact_judge.py), the same as runs/final500/judge. Facts: the frozen runs/final500/judge/claims, copied unchanged into this folder.

Export (19 Nov 2026): `evals.haiku_fact_judge.export_judge(runs/nograph/judge, <the 6 run folders in runs/nograph/run_registry.json>, kind=None, size=60000)`: 750 (run, question) pairs, 743 distinct tasks, files 000-133. 8 tasks have byte-identical retrieved text in runs/final500; they are judged again here, not reused.

Launched by 4 coordinator subagents (Claude Sonnet) that only launch Haiku subagents with the exact prompt below, run `runs/nograph/scripts/check_verdicts.py judge FIRST LAST`, and relaunch a file once with the same prompt if its verdict is missing or fails the check. Coordinators never write or edit verdicts. A file that fails twice is reported, not fixed.

## judge
Read the file /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/nograph/judge/judge/items-N.md completely (it can be long; read it in parts until the line "=== END ITEM" of its last item). It starts with your instructions and then has the items. Follow the instructions exactly for every item. Write your answer to /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/nograph/judge/judge/verdict-N.json as a JSON array, as the instructions describe. Quotes must be copied character for character from the item text; a line break in the file counts as a space. Do not modify any other file. Reply only with "done" and the number of items you wrote.
