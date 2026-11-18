"""What fills Agentic GraphRAG's 10 slots. Within one vector search, scores go down; a score that goes
up again marks a later search. Graph query rows have score -1.
Run from the repo root: python3 runs/heldout400/scripts/agentic_slot_use.py
"""
import json
from collections import Counter
from pathlib import Path

H = Path("runs/heldout400")
Q = json.load(open(H / "questions.json"))
qtype = {f"multihop-rag-{i}": t for t, ids in Q["by_type"].items() for i in ids}
for t in ("r1", "r2", "r3"):
    c = Counter(); later_q = Counter()
    for m in map(json.loads, open(H / "manifest.jsonl")):
        if m["tag"] != t or m["method"] != "graphrag" or not m["run_dir"]:
            continue
        for r in map(json.loads, open(Path(m["run_dir"]) / "retrieval.jsonl")):
            qt = qtype[r["record_id"]]
            s = r["retrieval"]["scores"]
            search, last = 1, None
            later = 0
            for x in s:
                if x == -1:
                    c[(qt, "graph row")] += 1; continue
                if last is not None and x > last + 1e-9:
                    search += 1
                if search == 1:
                    c[(qt, "first search")] += 1
                else:
                    c[(qt, "later search")] += 1; later += 1
                last = x
            later_q[(qt, later > 0)] += 1
    for qt in ("comparison_query", "temporal_query", "inference_query", "null_query"):
        print(f"{t} {qt[:5]}: slots from first search {c[(qt,'first search')]}, later searches {c[(qt,'later search')]}, "
              f"graph rows {c[(qt,'graph row')]} | questions with any later-search chunk {later_q[(qt, True)]}/100")
