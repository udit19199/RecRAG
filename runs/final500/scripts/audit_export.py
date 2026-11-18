"""Export the Claude Opus audit of Haiku gold-fact verdicts (final500 judge).

Usage (repo root): PYTHONPATH=. .venv/bin/python runs/final500/scripts/audit_export.py DATASET...

A stratified random sample: for each dataset, method and Haiku outcome
(found / not found), an equal share of distinct (task, fact) pairs, seed
20261113, about 200 in total over the three datasets (67, 67, 66). Each
sampled fact becomes one audit item: the same guide (JUDGE_GUIDE), the same
gold sentence, only that fact label, and the same retrieved items. Opus never
sees Haiku's verdict or the method. Items are shuffled and written 20 per file
to runs/final500/judge/audit/items-audit-NNN.{json,md}; sample.json keeps the
key (task, fact, dataset, method, Haiku status and found) for scoring.
"""

import json
import random
import sys
from pathlib import Path

from evals.haiku_fact_judge import JUDGE_GUIDE, PROMPT_VERSION, JudgeTask, render_task

OUT = Path("runs/final500/judge")
AUDIT = OUT / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)
SHARE = {"multihop_rag": 67, "hotpotqa": 67, "natural_questions": 66}
tasks = {
    item["id"]: JudgeTask.model_validate(item)
    for path in sorted((OUT / "judge").glob("items-*.json"))
    for item in json.loads(path.read_text())["items"]
}
pairs: dict[tuple, dict] = {}
for line in open(OUT / "per_question.jsonl"):
    row = json.loads(line)
    if row["dataset"] not in sys.argv[1:]:
        continue
    for fact in row["facts"]:
        key = (row["task"], fact["fact"])
        pairs.setdefault(key, {"dataset": row["dataset"], "method": row["method"], "task": row["task"],
                               "fact": fact["fact"], "judged": fact["judged"], "found": fact["found"]})
rng = random.Random(20261113)
sample_path = AUDIT / "sample.json"
sample = json.loads(sample_path.read_text()) if sample_path.exists() else []
done = {s["dataset"] for s in sample}
for dataset in sys.argv[1:]:
    if dataset in done:
        continue
    strata = [(m, f) for m in ("rag", "raptor", "hipporag", "graphrag") for f in (True, False)]
    per = [SHARE[dataset] // len(strata)] * len(strata)
    for i in range(SHARE[dataset] - sum(per)):
        per[i] += 1
    for (method, found), n in zip(strata, per):
        pool = sorted(
            (p for p in pairs.values() if p["dataset"] == dataset and p["method"] == method and p["found"] == found),
            key=lambda p: (p["task"], p["fact"]),
        )
        sample += rng.sample(pool, n)
sample_path.write_text(json.dumps(sample, indent=1))
# Files already written are kept as they are; only samples not yet in a file
# are shuffled and written to new files, so a dataset added later (Natural
# Questions finished last) does not reshuffle items that may already be judged.
written = {
    item["id"]
    for path in AUDIT.glob("items-audit-*.json")
    for item in json.loads(path.read_text())["items"]
}
first = len(list(AUDIT.glob("items-audit-*.json")))
items = []
for n, s in enumerate(sample):
    if f"a{n:03d}-{s['task']}" in written:
        continue
    task = tasks[s["task"]]
    sentence = int(s["fact"].split(".")[0])
    gold = [{"sentence": task.gold[sentence]["sentence"], "facts": {s["fact"]: task.gold[sentence]["facts"][s["fact"]]}}]
    items.append(JudgeTask(id=f"a{n:03d}-{s['task']}", gold=gold, items=task.items))
order = list(range(len(items)))
random.Random(7).shuffle(order)
items = [items[i] for i in order]
for number, start in enumerate(range(0, len(items), 20), start=first):
    batch = items[start : start + 20]
    name = f"items-audit-{number:03d}"
    (AUDIT / f"{name}.json").write_text(json.dumps({"prompt_version": PROMPT_VERSION, "guide": JUDGE_GUIDE, "items": [t.model_dump() for t in batch]}, ensure_ascii=False, indent=1))
    (AUDIT / f"{name}.md").write_text(f"Prompt version {PROMPT_VERSION}\n\n{JUDGE_GUIDE}\n\nThis file has {len(batch)} items.\n\n" + "\n".join(render_task(t) for t in batch))
print(len(sample), "sampled facts,", len(items), "new items in", (len(items) + 19) // 20, "new files")
