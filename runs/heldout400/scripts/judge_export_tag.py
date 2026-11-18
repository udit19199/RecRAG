"""Export unjudged items for one pass (by manifest tag) into 80-item batch files."""
import json, glob, os, subprocess, sys
from pathlib import Path
from evals.subagent_judge import JUDGE_GUIDE, answer_items, evidence_items
tag, out = sys.argv[1], Path(sys.argv[2])
dirs = [Path(json.loads(l)["run_dir"]) for l in open("runs/heldout400/manifest.jsonl")
        if json.loads(l)["tag"] == tag and json.loads(l)["run_dir"]]
done = set()
for f in glob.glob("runs/heldout400/judge/verdicts-*.json"):
    done |= {v["id"] for v in json.load(open(f))}
items = [i for i in [*answer_items(dirs), *evidence_items(dirs, "multihop_rag")] if i.id not in done]
out.mkdir(parents=True, exist_ok=True)
for n, s in enumerate(range(0, len(items), 80)):
    ch = items[s:s + 80]
    json.dump({"guide": JUDGE_GUIDE, "answers": [i.model_dump() for i in ch if i.kind == "answer"],
               "evidence": [i.model_dump() for i in ch if i.kind == "evidence"]}, open(out / f"items-{n:03d}.json", "w"), indent=1)
print(len(dirs), "run dirs;", sum(i.kind == "answer" for i in items), "new answers,", sum(i.kind == "evidence" for i in items), "new evidence ->", (len(items) + 79) // 80, "files")
