"""How often Agentic Hybrid RAG's graph queries (cypher_search) failed or came back empty in the latest run.

Reads every Agentic Hybrid RAG retrieval.jsonl listed in runs/final500/run_registry.json
(3 datasets x 3 repeats x 500 questions) and every cypher_search step in each row's
retrieval.trace. A step whose status is not "success" failed: GPT-6 Luna wrote a query
Neo4j rejected, it hit the 10 s limit, or it ran out of database memory. A successful step
with 0 results ran but found nothing.

Run from the repo root: python3 runs/final500/cypher_failures/compute.py
Writes cypher_failures.json next to this file and prints the table in SUMMARY.md.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("cypher_failures.json")
DATASETS = ["multihop_rag", "hotpotqa", "natural_questions"]


def cause(output: str) -> str:
    first = output.split("\n")[0]
    if "Refused a query that returns passage" in output:
        return "refused: asked for article text"
    if "would write" in output:
        return "refused: write query"
    if "terminated" in output or "timed out" in output.lower():
        return "timeout (10 s)"
    if "allocation" in output:
        return "out of database memory"
    if "Invalid input" in first:
        return "not valid Cypher"
    return "valid Cypher, impossible request"


def main() -> None:
    registry = json.loads((ROOT / "runs/final500/run_registry.json").read_text())
    per_dataset: dict[str, Counter] = {d: Counter() for d in DATASETS}
    causes: Counter = Counter()
    messages: Counter = Counter()
    hit: dict[str, set] = defaultdict(set)
    for dataset in DATASETS:
        for repeat, folders in registry[dataset]["graphrag"].items():
            for folder in folders:
                # the registry stores absolute paths from the machine that ran it; keep the part from runs/ on
                parts = Path(folder).parts
                path = ROOT.joinpath(*parts[parts.index("runs"):], "retrieval.jsonl")
                for line in path.open():
                    row = json.loads(line)
                    per_dataset[dataset]["answers"] += 1
                    for step in (row.get("retrieval") or {}).get("trace") or []:
                        if step.get("tool") != "cypher_search":
                            continue
                        c = per_dataset[dataset]
                        c["calls"] += 1
                        if step.get("status") != "success":
                            c["failed"] += 1
                            out = step.get("output") or ""
                            causes[cause(out)] += 1
                            messages[out.split("\n")[0][:80]] += 1
                            hit[dataset].add((repeat, row.get("record_id")))
                        elif step.get("results") == 0:
                            c["empty"] += 1
    for dataset in DATASETS:
        per_dataset[dataset]["answers_with_failure"] = len(hit[dataset])
    result = {
        "per_dataset": {d: dict(c) for d, c in per_dataset.items()},
        "causes": dict(causes.most_common()),
        "top_messages": dict(messages.most_common(15)),
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")

    print("| Dataset | Graph queries | Failed | Ran, found nothing | Answers with a failed query |")
    print("|---|---|---|---|---|")
    total = Counter()
    for d in DATASETS:
        c = per_dataset[d]
        total.update(c)
        print(f"| {d} | {c['calls']:,} | {c['failed']} ({c['failed'] / c['calls']:.1%}) | "
              f"{c['empty']} ({c['empty'] / c['calls']:.1%}) | {c['answers_with_failure']} of {c['answers']:,} |")
    print(f"| total | {total['calls']:,} | {total['failed']} ({total['failed'] / total['calls']:.1%}) | "
          f"{total['empty']:,} ({total['empty'] / total['calls']:.1%}) | "
          f"{total['answers_with_failure']} of {total['answers']:,} |")
    print()
    for k, v in causes.most_common():
        print(f"- {v}: {k}")


if __name__ == "__main__":
    main()
