# Prompts given to the Claude Haiku subagents (model: haiku, via Claude Code Agent tool)

The judging instructions themselves are the `guide` field inside every items-*.json file.
The subagent task prompt was, with N replaced by the file number:

## claims
Read the file /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/agentic_tune/fact_judge/claims/items-N.json. Follow the instructions in its "guide" field exactly for every item in its "items" list. Write your answer to /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/agentic_tune/fact_judge/claims/verdict-N.json as the guide describes. Do not modify any other file. Reply only with "done" and the number of items you wrote.

## judge
Read the file /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/agentic_tune/fact_judge/judge/items-N.md completely (it can be long; read it in parts until the line "=== END ITEM" of its last item). It starts with your instructions and then has the items. Follow the instructions exactly for every item. Write your answer to /Users/uditagarwal19199/Workspace/HPE/RecRAG/runs/agentic_tune/fact_judge/judge/verdict-N.json as a JSON array, as the instructions describe. Quotes must be copied character for character from the item text; a line break in the file counts as a space. Do not modify any other file. Reply only with "done" and the number of items you wrote.
