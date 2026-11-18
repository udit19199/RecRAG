"""Where do Agentic GraphRAG's found gold facts come from: article text, or only the graph lines
(facts, entities, graph query rows)? A claim counts as article-text if any of its quotes appears
in the normalized article text of the retrieved items.
Run from the repo root: PYTHONPATH=. uv run python runs/heldout400/scripts/agentic_fact_source.py
"""
import json
from collections import defaultdict
from pathlib import Path
from evals.retrieval_quality import normalize

H = Path("runs/heldout400")
def article_text(p):
    try:
        d = json.loads(p)
    except ValueError:
        return p
    if not isinstance(d, dict):
        return ""
    out = [d.get("text", "")]
    for k, v in d.items():
        if "passage" in k.lower():
            vs = v if isinstance(v, list) else [v]
            for x in vs:
                out.append(x.get("passage", "") if isinstance(x, dict) else str(x))
    return "\n".join(out)

cache = {}
def texts(run):
    if run not in cache:
        cache[run] = {}
        d = Path(run) if (Path(run) / "retrieval.jsonl").exists() else H / run
        for r in map(json.loads, open(d / "retrieval.jsonl")):
            cache[run][r["record_id"]] = normalize("\n".join(article_text(p) for p in r["retrieval"]["passages"]))
    return cache[run]

tot = defaultdict(lambda: [0, 0, 0])   # (rep, type) -> [stated claims, via article text, only graph lines]
q_change = defaultdict(lambda: [0, 0, 0, 0])  # questions at 100% with all text vs article text only, and >=50%
for t in ["r1", "r2", "r3"]:
    for s in range(4):
        for r in map(json.loads, open(H / "judge_support" / f"{t}-s{s}" / "per_question.jsonl")):
            if r["method"] != "graphrag":
                continue
            art = texts(r["run"])[r["record_id"]]
            c = [x for g in r["gold"] for x in g["claims"]]
            st = [x for x in c if x["full_status"] == "stated"]
            via = [x for x in st if any(normalize(q) and normalize(q) in art for q in x["quotes"])]
            k = (t, r["question_type"])
            tot[k][0] += len(st); tot[k][1] += len(via); tot[k][2] += len(st) - len(via)
            if c:
                q_change[k][0] += len(st) == len(c); q_change[k][1] += len(via) == len(c)
                q_change[k][2] += len(st) / len(c) >= .5; q_change[k][3] += len(via) / len(c) >= .5
for k in sorted(tot):
    a, b, c = tot[k]
    print(f"{k[0]} {k[1]:17} stated claims {a:4}, found in article text {b:4}, only in graph lines {c:3} | "
          f"questions at 100%: {q_change[k][0]} -> {q_change[k][1]} article text only; >=50%: {q_change[k][2]} -> {q_change[k][3]}")
