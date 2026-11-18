"""Export gold-fact judge tasks for the study's runs (evals/haiku_fact_judge.py).

Usage (repo root): PYTHONPATH=. .venv/bin/python runs/final500/scripts/judge_export.py DATASET...

For every run folder in run_registry.json of the given datasets: each
(question, retrieved text) pair becomes a task, judged on the question's
frozen facts. The retrieved text is what the answer model was given
(haiku_fact_judge.answer_items: de-duplicated, shuffled with a fixed seed so
rank order is hidden). The task id hashes the question, its facts and the
exact retrieved text, so byte-identical text is judged once and its verdict
reused; nothing from any earlier judge folder is used. Tasks never show the
method or run. New tasks are shuffled across methods and datasets (seed
20261113) before they are cut into files of about 60,000 characters of
retrieved text, numbered after the last existing file. links.jsonl maps every
(dataset, method, repeat, question) to its task.
"""

import json
import random
import sys
from pathlib import Path

from evals.haiku_fact_judge import (
    JudgeTask,
    _hash,
    _write_judge_batches,
    answer_items,
    load_claims,
)
from evals.retrieval_quality import load_run

OUT = Path("runs/final500/judge")
datasets = sys.argv[1:]
registry = json.loads(Path("runs/final500/run_registry.json").read_text())
questions = json.loads(Path("runs/final500/questions.json").read_text())
claims = load_claims(OUT)
folder = OUT / "judge"
folder.mkdir(parents=True, exist_ok=True)
links_path = folder / "links.jsonl"
old_links = links_path.read_text().splitlines() if links_path.exists() else []
linked = {
    (row["dataset"], row["method"], row["rep"], row["record_id"])
    for row in map(json.loads, old_links)
}
done = {
    item["id"]
    for path in folder.glob("items-*.json")
    for item in json.loads(path.read_text())["items"]
}
tasks: dict[str, JudgeTask] = {}
links = []
for dataset in datasets:
    for method, reps in registry[dataset].items():
        for rep, dirs in reps.items():
            seen = set()
            for run_dir in dirs:
                for saved in load_run(Path(run_dir)):
                    assert saved.dataset == dataset and saved.method == method
                    seen.add(saved.record_id)
                    key = (dataset, method, rep, saved.record_id)
                    if not saved.supporting_sentences or key in linked:
                        continue
                    facts = claims[saved.record_id]
                    gold = [
                        {
                            "sentence": sentence,
                            "facts": {f"{s}.{f}": fact for f, fact in enumerate(facts[s])},
                        }
                        for s, sentence in enumerate(saved.supporting_sentences)
                    ]
                    items = answer_items(saved)
                    task_id = _hash(saved.record_id, json.dumps(gold), *items)
                    if task_id not in done:
                        tasks.setdefault(task_id, JudgeTask(id=task_id, gold=gold, items=items))
                    links.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "rep": rep,
                            "run": run_dir,
                            "record_id": saved.record_id,
                            "question_type": saved.question_type,
                            "task": task_id,
                        }
                    )
            assert len(seen) == 500, (dataset, method, rep, len(seen))
new = list(tasks.values())
random.Random(20261113).shuffle(new)
first = 1 + max((int(p.stem.split("-")[1]) for p in folder.glob("items-*.json")), default=-1)
_write_judge_batches(folder, new, 60000, first)
links_path.write_text(
    "".join(line + "\n" for line in old_links)
    + "".join(json.dumps(link) + "\n" for link in links)
)
last = max((int(p.stem.split("-")[1]) for p in folder.glob("items-*.json")), default=-1)
print(f"{len(links)} (run, question) pairs, {len(new)} new tasks, files {first:03d}-{last:03d}")
