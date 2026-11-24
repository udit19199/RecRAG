"""Export answers that still need the Haiku answer judge, for runs added to runs.json.

Same rule and item format as runs/nograph/scripts/answers_export.py: exact
match first; items are (question, normalized answer) pairs needing a judge,
minus those already judged in runs/final500, runs/nograph or here.
Writes the next runs/judge_paper/answers/items-answers-NNN.json.

Usage (repo root, light): PYTHONPATH=. .venv/bin/python runs/judge_paper/scripts/answers_export.py
"""
import glob, json, random
from pathlib import Path
from evals.subagent_judge import JudgeItems, answer_items

runs = json.loads(Path("runs/judge_paper/runs.json").read_text())
dirs = [Path(d) for reps in runs.values() for ds in reps.values() for d in ds]
done = {v["id"] for p in glob.glob("runs/final500/judge/answers/verdicts-answers-*.json")
        + glob.glob("runs/nograph/judge/answers/verdicts-answers-*.json")
        + glob.glob("runs/judge_paper/answers/*-answers-*.json") for v in
        (json.load(open(p)) if "verdicts" in p else json.load(open(p))["answers"])}
items = [i for i in answer_items(dirs) if i.id not in done]
if items:
    n = len(glob.glob("runs/judge_paper/answers/items-answers-*.json"))
    random.Random(20261122).shuffle(items)
    # At most 100 items per file (24 Nov: the B4 runs add several hundred).
    for k in range(0, len(items), 100):
        out = Path(f"runs/judge_paper/answers/items-answers-{n:03d}.json")
        out.write_text(JudgeItems(answers=items[k:k + 100]).model_dump_json(indent=1))
        print(len(items[k:k + 100]), "new answer items in", out)
        n += 1
else:
    print("no new answer items")
