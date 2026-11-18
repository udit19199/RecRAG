"""Does the Cypher call steer the agent's next search? Comparison, 10 Nov runs, logs only.
Run from the repo root: python3 runs/heldout400/scripts/cypher_steering.py"""
import json, re, collections
from pathlib import Path
NEW = {"r1": "runs/agentic_tune/graphrag/comparison100", "r2": "runs/agentic_tune/graphrag/comparison100_rep2", "r3": "runs/agentic_tune/graphrag/comparison100_rep3"}
share = {}
for l in open("runs/agentic_tune/fact_judge/per_question.jsonl"):
    r = json.loads(l); share[(Path(r["run"]).name, r["record_id"])] = r["share"]
first = {}
for l in open("runs/agentic_tune/search_text/judge/per_question.jsonl"):
    r = json.loads(l); first[(Path(r["run"]).name, r["record_id"])] = r["share"]
words = lambda s: {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'’-]{3,}|\d{2,}", s)}
stat = collections.defaultdict(lambda: collections.Counter())
terms = collections.Counter()
for t, d in NEW.items():
    for l in open(Path(d) / "retrieval.jsonl"):
        s = json.loads(l); q = s["record_id"]; tr = s["retrieval"]["trace"]
        vs = [x for x in tr if x["tool"] == "vector_search"]; cy = [x for x in tr if x["tool"] == "cypher_search"]
        c = cy[0] if cy else None
        status = "no cypher" if not c else "failed" if c["status"] == "error" else "no rows" if c["results"] == 0 else "rows"
        g = stat[status]; g["questions"] += 1
        g["2nd search"] += len(vs) >= 2
        p10 = share[(Path(d).name, q)]; f1 = first[(f"agent_first_{t}", q)]
        g["gained >=50% over first search"] += (p10 >= .5) and (f1 < .5)
        g["first search >=50%"] += f1 >= .5
        if len(vs) >= 2 and c and status == "rows":
            q2 = words(vs[1]["args"].get("query_text", ""))
            known = words(s["question"]) | words(vs[0]["output"]) | words(vs[0]["args"].get("query_text", ""))
            new_terms = q2 - known
            from_cy = new_terms & words(c["output"])
            terms["2nd searches after rows"] += 1
            terms["with a new word that only Cypher had shown"] += bool(from_cy)
            if from_cy:
                gained = (p10 >= .5) and (f1 < .5)
                terms["  of these, gained >=50%"] += gained
            else:
                terms["  others, gained >=50%"] += (p10 >= .5) and (f1 < .5)
for k, v in stat.items():
    print(k, dict(v))
print(dict(terms))
