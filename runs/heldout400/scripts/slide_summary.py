"""Numbers for the result slides: average (lowest-highest) over 3 repeats, and one
2x2 matrix per measure vs Normal RAG, counting a question as right/found when the
method got it in at least 2 of 3 repeats. Haiku judges only.
Run from the repo root: PYTHONPATH=. uv run python runs/heldout400/scripts/slide_summary.py"""

import glob
import json
from pathlib import Path

from evals.subagent_judge import answer_key

H = Path("runs/heldout400")
TAGS = ("r1", "r2", "r3")
verd = {}
for f in sorted(glob.glob(str(H / "judge/verdicts-*.json"))) + sorted(glob.glob("runs/agentic_tune/judge/verdicts-*.json")):
    for v in json.load(open(f)):
        verd[v["id"]] = v


def right(r):
    return bool(r["exact_match"]) or bool(verd.get(answer_key(r["question"], r["answer"]), {}).get("correct"))


NEW = {"comparison_query": ("comparison100", "comparison100_rep2", "comparison100_rep3"),
       "temporal_query": ("temporal100", "temporal100_rep2", "temporal100_rep3")}
FACTS = {"comparison_query": "runs/agentic_tune/fact_judge", "temporal_query": "runs/agentic_tune/fact_judge_temporal"}


def collect(kind):
    """{method: {tag: {qid: {"right": bool, "share": float|None}}}}"""
    dirs = {}
    for line in open(H / "manifest.jsonl"):
        m = json.loads(line)
        if m["tag"] in TAGS and m["exit"] == 0 and m["run_dir"]:
            d = m["run_dir"]; d = d if d.startswith("/") else str(H / d)
            dirs.setdefault((m["method"], m["tag"]), []).append(d)
    for tag, n in zip(TAGS, NEW[kind]):
        dirs[("agentic_10nov", tag)] = [f"runs/agentic_tune/graphrag/{n}"]
    share = {}
    pq = Path(FACTS[kind]) / "per_question.jsonl"
    if pq.exists():
        for line in open(pq):
            r = json.loads(line)
            share[(str(Path(r["run"]).resolve()), r["record_id"])] = r["share"]
    out = {}
    for (meth, tag), ds in dirs.items():
        for d in ds:
            for x in open(Path(d) / "results.jsonl"):
                r = json.loads(x)
                if r["question_type"] != kind:
                    continue
                out.setdefault(meth, {}).setdefault(tag, {})[r["record_id"]] = {
                    "right": right(r), "share": share.get((str(Path(d).resolve()), r["record_id"]))}
    return out


MEASURES = [("Answered right", lambda v: v["right"]),
            ("Gold facts ≥50%", lambda v: v["share"] is not None and v["share"] >= 0.5 - 1e-9),
            ("Gold facts ≥75%", lambda v: v["share"] is not None and v["share"] >= 0.75 - 1e-9),
            ("Gold facts 100%", lambda v: v["share"] is not None and v["share"] >= 1 - 1e-9)]
summary = {}
for kind in ("comparison_query", "temporal_query"):
    data = collect(kind)
    qs = sorted(data["rag"]["r1"])
    for meth, reps in data.items():
        for name, f in MEASURES:
            if name != "Answered right" and any(v["share"] is None for t in TAGS for v in reps[t].values()):
                continue
            counts = [sum(f(reps[t][q]) for q in qs) for t in TAGS]
            row = {"avg": round(sum(counts) / 3), "low": min(counts), "high": max(counts), "per_repeat": counts}
            if meth != "rag":
                maj = lambda m, q: sum(f(data[m][t][q]) for t in TAGS) >= 2
                a = [maj(meth, q) for q in qs]; b = [maj("rag", q) for q in qs]
                row["matrix_2of3"] = [sum(x and y for x, y in zip(a, b)), sum(x and not y for x, y in zip(a, b)),
                                      sum(y and not x for x, y in zip(a, b)), sum(not x and not y for x, y in zip(a, b))]
            summary[f"{kind}|{meth}|{name}"] = row
            print(kind, meth, name, row)
Path("runs/agentic_tune/slide_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
