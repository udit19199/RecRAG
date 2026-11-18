# Re-judge: held-out 400 evidence items

> FInd out all of these and put them on excalidraw
> (item from the list: "Rescore with batch 2 re-judged and the 38 gaps filled")

items: 232 evidence items (194 decided by the older batch-2 wording, 38 never judged)
inputs: runs/heldout400/rejudge/items-000.json .. items-003.json
guide: current strict JUDGE_GUIDE in evals/subagent_judge.py (embedded in each items file)
judge model: Claude Haiku subagent (same as the original held-out judging)
outputs: runs/heldout400/rejudge/verdicts-000.json .. verdicts-003.json, one verdict per item id, no extra ids
original judge/ folder: unchanged
