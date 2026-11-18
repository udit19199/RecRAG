"""Where the 10 items Agentic Hybrid RAG passed to the answer model came from (MultiHop-RAG, final500).

For each item: graph row or text passage; if a passage, whether the first vector_search
already returned it, or only a later search. Also: picked by the agent or filled by the
old rule. Read-only; prints a table.
"""
import json
import re
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

F = Path(__file__).resolve().parents[1]
registry = json.load(open(F / "run_registry.json"))["multihop_rag"]["graphrag"]
ID = re.compile(r"^\[(E[0-9a-f]{6})\] ", re.M)


def key(content: str) -> str:
    try:
        text = json.loads(content)["text"]
    except (ValueError, KeyError, TypeError):
        text = content
    return " ".join(text.lower().split())


def eid(content: str) -> str:
    return "E" + sha256(key(content).encode()).hexdigest()[:6]


def is_passage(content: str) -> bool:
    try:
        return "text" in json.loads(content) and "entities" in json.loads(content)
    except (ValueError, TypeError):
        return False


stats: dict[str, Counter] = defaultdict(Counter)
for rep, dirs in registry.items():
    for d in dirs:
        for line in open(Path(d) / "retrieval.jsonl"):
            r = json.loads(line)
            t = r["question_type"].replace("_query", "")
            ret = r["retrieval"]
            trace = ret.get("trace") or []
            if not trace or trace[-1]["tool"] != "select_evidence":
                continue  # previous-run rows reused without the final code
            s = stats[t]
            s["runs"] += 1
            first = next((st for st in trace if st["tool"] == "vector_search"), None)
            first_ids = set(ID.findall(first["output"])) if first else set()
            picked_ids = re.findall(r"E[0-9a-f]{6}", trace[-1]["output"])
            graph_in = later_in = 0
            for i, item in enumerate(ret["evidence"]):
                s["items"] += 1
                if i < trace[-1]["results"]:
                    s["picked"] += 1
                if not is_passage(item):
                    s["graph"] += 1
                    graph_in += 1
                elif eid(item) not in first_ids:
                    s["later_passage"] += 1
                    later_in += 1
            s["runs_with_graph"] += graph_in > 0
            s["runs_with_later"] += later_in > 0
            s["picks_nonempty"] += bool(picked_ids)

print(f"{'type':11} {'runs':>5} {'agent-picked items':>19} {'graph rows':>12} {'later-search passages':>22} {'runs w/ graph':>14} {'runs w/ later':>14}")
for t, s in sorted(stats.items()):
    n = s["items"]
    print(
        f"{t:11} {s['runs']:5} {100*s['picked']/n:18.0f}% {100*s['graph']/n:11.1f}% {100*s['later_passage']/n:21.1f}%"
        f" {100*s['runs_with_graph']/s['runs']:13.0f}% {100*s['runs_with_later']/s['runs']:13.0f}%"
    )
