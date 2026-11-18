"""Compare a new Agentic GraphRAG run with the held-out r1-r3 runs on the same questions.

Usage: PYTHONPATH=. uv run python runs/heldout400/scripts/agentic_tune_compare.py NEW_RUN_DIR
Prints cost (billed estimate and uncached list price), how much input came from
OpenAI's prompt cache at each agent step, time, and what the agent's tool calls did."""

import collections
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
H = ROOT / "runs/heldout400"
new_dir = Path(sys.argv[1])
new = [json.loads(line) for line in (new_dir / "results.jsonl").open()]
ids = {r["record_id"] for r in new}

old = collections.defaultdict(list)
for line in (H / "manifest.jsonl").open():
    row = json.loads(line)
    if row["method"] != "graphrag" or row["tag"] not in {"r1", "r2", "r3"} or not row["run_dir"]:
        continue
    run = Path(row["run_dir"])
    run = run if run.is_absolute() else H / run
    for result in (run / "results.jsonl").open():
        r = json.loads(result)
        if r["record_id"] in ids:
            old[row["tag"]].append(r)
# r2 was finished in resume folders that are not in the manifest as r2.
for run in (H / "runs/graphrag").glob("*r2resume"):
    for result in (run / "results.jsonl").open():
        r = json.loads(result)
        if r["record_id"] in ids and all(r["record_id"] != o["record_id"] for o in old["r2"]):
            old["r2"].append(r)


def summary(name, rows):
    if not rows:
        print(f"{name}: no rows")
        return
    n = len(rows)
    billed = sum(r["estimated_question_cost_usd"] for r in rows) / n * 1000
    listed = sum(r["list_question_cost_usd"] for r in rows) / n * 1000
    calls = [c for r in rows for c in r["question_cost"]["calls"] if c["kind"] == "responses"]
    inp = sum(c["input_tokens"] for c in calls)
    cached = sum(c["cached_input_tokens"] for c in calls)
    writes = sum(c["cache_write_tokens"] for c in calls)
    print(
        f"{name:5s} n={n:3d}  $/1000q billed {billed:5.2f} list {listed:5.2f}  "
        f"model calls/q {len(calls) / n:4.1f}  input/q {inp / n:7.0f}  "
        f"cached {cached / inp * 100:3.0f}%  cache-written {writes / inp * 100:3.0f}%  "
        f"median search s {st.median(r['retrieval_seconds'] for r in rows):5.1f}  "
        f"EM {sum(r['exact_match'] for r in rows) / n:.2f}"
    )
    steps = collections.defaultdict(lambda: [0, 0, []])
    for r in rows:
        resp = [c for c in r["question_cost"]["calls"] if c["kind"] == "responses"]
        for k, c in enumerate(resp):
            role = "answer" if k == len(resp) - 1 else f"call{min(k, 5)}"
            steps[role][0] += c["input_tokens"]
            steps[role][1] += c["cached_input_tokens"]
            steps[role][2].append(c["seconds"])
    print(
        "      cached by call: "
        + "  ".join(
            f"{k} {v[1] / max(v[0], 1) * 100:3.0f}% ({st.median(v[2]):.1f}s)"
            for k, v in sorted(steps.items())
        )
    )


for tag in ("r1", "r2", "r3"):
    summary(tag, old[tag])
summary("new", new)

retrieval = [json.loads(line) for line in (new_dir / "retrieval.jsonl").open()]
tools = collections.Counter()
for saved in retrieval:
    trace = saved["retrieval"].get("trace", [])
    for step in trace:
        tools[step["tool"], step["status"], "empty" if step["results"] == 0 else "rows"] += 1
print("tool calls (tool, status, results):", dict(tools))
