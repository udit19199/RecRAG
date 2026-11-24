"""Gold-fact judge tasks for the 21 Nov runs (replays, selection baselines, Milvus).

Usage (repo root, Mac, light): PYTHONPATH=. .venv/bin/python runs/judge_paper/scripts/export.py

Reads runs/judge_paper/runs.json: {label: {rep: [run folder, ...]}}. Same task
rule as runs/final500 (evals/haiku_fact_judge.export_judge): one task per
(question, retrieved text), judged on the question's frozen final500 facts; the
task id hashes the question, its facts and the exact retrieved text. A task
already judged in runs/final500/judge or runs/nograph/judge (byte-identical
retrieved text) keeps that Haiku verdict: copied into items-reuse.json and
verdict-reuse.json. Every other new task goes into numbered files of about
60,000 characters, after the last existing file. Run again after adding runs:
tasks and links already exported are kept. links.jsonl maps (label, rep,
question) to its task.
"""

import json
import random
from pathlib import Path

from evals.haiku_fact_judge import JudgeTask, _hash, _write_judge_batches, answer_items, load_claims
from evals.retrieval_quality import load_run

OUT = Path("runs/judge_paper")
folder = OUT / "judge"
folder.mkdir(parents=True, exist_ok=True)
claims = load_claims(Path("runs/final500/judge"))
runs = json.loads((OUT / "runs.json").read_text())

prior_items, prior_verdicts = {}, {}
for src in (Path("runs/final500/judge/judge"), Path("runs/nograph/judge/judge")):
    for p in src.glob("items-*.json"):
        for item in json.loads(p.read_text())["items"]:
            prior_items[item["id"]] = item
    for p in src.glob("verdict-*.json"):
        for v in json.loads(p.read_text()):
            prior_verdicts[v["id"]] = v

links_path = folder / "links.jsonl"
old_links = links_path.read_text().splitlines() if links_path.exists() else []
linked = {(r["label"], r["rep"], r["record_id"]) for r in map(json.loads, old_links)}
done = {i["id"] for p in folder.glob("items-*.json") for i in json.loads(p.read_text())["items"]}
reuse_items = json.loads((folder / "items-reuse.json").read_text())["items"] if (folder / "items-reuse.json").exists() else []
reuse_verdicts = json.loads((folder / "verdict-reuse.json").read_text()) if (folder / "verdict-reuse.json").exists() else []

new: dict[str, JudgeTask] = {}
links = []
# B4_rev and B4_shuf give the answer model B4's exact 50 passages in another
# order; gold facts found depend only on which text was given, so they take B4's
# verdicts (runs/judge_paper/slide_data.py) and get no tasks of their own.
SAME_TEXT_AS = {"B4_rev": "B4", "B4_shuf": "B4"}
# B4 items hold 20-50 passages; files of about 120,000 characters (24 Nov) keep
# the file count down. Same guide and task per item; only how many items share
# a file changes.
SIZE = 120000 if any(l.startswith("B4") for l in runs) else 60000
for label, reps in runs.items():
    if label in SAME_TEXT_AS or label.endswith("_infnull"):
        # infnull runs (inference + null questions) get answered-right judging
        # only; SPEC: no gold-fact judging for them.
        continue
    for rep, dirs in reps.items():
        for run_dir in dirs:
            for saved in load_run(Path(run_dir)):
                if not saved.supporting_sentences or (label, rep, saved.record_id) in linked:
                    continue
                facts = claims[saved.record_id]
                gold = [{"sentence": s, "facts": {f"{i}.{f}": fact for f, fact in enumerate(facts[i])}}
                        for i, s in enumerate(saved.supporting_sentences)]
                items = answer_items(saved)
                task_id = _hash(saved.record_id, json.dumps(gold), *items)
                if task_id not in done and task_id not in new:
                    if task_id in prior_verdicts:
                        reuse_items.append(prior_items[task_id])
                        reuse_verdicts.append(prior_verdicts[task_id])
                        done.add(task_id)
                    else:
                        new[task_id] = JudgeTask(id=task_id, gold=gold, items=items)
                links.append({"label": label, "rep": rep, "run": run_dir, "record_id": saved.record_id,
                              "question_type": saved.question_type, "task": task_id})

guide = json.loads(next(Path("runs/final500/judge/judge").glob("items-*.json")).read_text()).get("guide")
(folder / "items-reuse.json").write_text(json.dumps({"guide": guide, "items": reuse_items}, ensure_ascii=False))
(folder / "verdict-reuse.json").write_text(json.dumps(reuse_verdicts, ensure_ascii=False))
tasks = list(new.values())
random.Random(20261121).shuffle(tasks)
numbered = [int(p.stem.split("-")[1]) for p in folder.glob("items-*.json") if p.stem.split("-")[1].isdigit()]
first = 1 + max(numbered, default=-1)
if tasks:
    _write_judge_batches(folder, tasks, SIZE, first)
links_path.write_text("".join(l + "\n" for l in old_links) + "".join(json.dumps(l) + "\n" for l in links))
last = max([int(p.stem.split("-")[1]) for p in folder.glob("items-*.json") if p.stem.split("-")[1].isdigit()], default=-1)
print(f"{len(links)} new (run, question) pairs; {len(reuse_items)} tasks reuse an earlier verdict; "
      f"{len(tasks)} new tasks in files {first:03d}-{last:03d}")
