"""Export the no-graph run's answers that still need the Haiku answer judge.

Same rule and item format as runs/final500/scripts/answers_export.py: exact
match first; items are (question, normalized answer) pairs needing a judge,
minus those already judged in runs/final500 (identical answer, same verdict).
Writes runs/nograph/judge/answers/items-answers-000.json.
"""
import glob, json, random
from pathlib import Path
from evals.subagent_judge import JudgeItems, answer_items

reg = json.loads(Path("runs/nograph/run_registry.json").read_text())
dirs = [Path(d) for ds in reg.values() for d in ds]
done = {v["id"] for p in glob.glob("runs/final500/judge/answers/verdicts-answers-*.json") for v in json.load(open(p))}
items = [i for i in answer_items(dirs) if i.id not in done]
random.Random(20261113).shuffle(items)
Path("runs/nograph/judge/answers/items-answers-000.json").write_text(JudgeItems(answers=items).model_dump_json(indent=1))
print(len(items), "new answer items")
