"""Export answers for the Claude Haiku answer judge (evals/subagent_judge.py).

Usage (repo root): PYTHONPATH=. .venv/bin/python runs/final500/scripts/answers_export.py DATASET...

Answered right = exact match, then the Haiku answer judge on the rest. Items
are the answers with exact match 0 that are not an abstention and have a
gold answer (subagent_judge.needs_answer_judge), from every run of the given
datasets in run_registry.json and from the closed-book baseline. An item is
one (question, normalized answer) pair, so the same answer is judged once
for every method and repeat; items never show the method. No earlier verdict
is used. New items are shuffled (seed 20261113) and written 60 per file to
runs/final500/judge/answers/items-answers-NNN.json, numbered after the last
existing file.
"""

import json
import random
import sys
from pathlib import Path

from evals.subagent_judge import JudgeItems, answer_items

OUT = Path("runs/final500/judge/answers")
OUT.mkdir(parents=True, exist_ok=True)
registry = json.loads(Path("runs/final500/run_registry.json").read_text())
dirs: list[Path] = []
for dataset in sys.argv[1:]:
    for reps in registry[dataset].values():
        dirs += [Path(d) for ds in reps.values() for d in ds]
    dirs += sorted(Path("runs/final500/closed_book", dataset).glob("r*"))
done = {
    item["id"]
    for path in OUT.glob("items-answers-*.json")
    for item in json.loads(path.read_text())["answers"]
}
items = [item for item in answer_items(dirs) if item.id not in done]
random.Random(20261113).shuffle(items)
first = 1 + max((int(p.stem.rsplit("-", 1)[1]) for p in OUT.glob("items-answers-*.json")), default=-1)
for number, start in enumerate(range(0, len(items), 60), start=first):
    (OUT / f"items-answers-{number:03d}.json").write_text(
        JudgeItems(answers=items[start : start + 60]).model_dump_json(indent=1)
    )
print(len(dirs), "run folders,", len(items), "new answer items, files from", f"{first:03d}")
