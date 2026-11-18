"""Score the Claude Opus audit against the Haiku gold-fact verdicts.

Usage (repo root): PYTHONPATH=. .venv/bin/python runs/final500/scripts/audit_score.py

Opus verdicts get the same in-code quote check as Haiku's (a fact counts as
found only when judged "stated" with at least one quote that is really in the
item). Writes runs/final500/judge/audit/report.json: agreement overall, per
dataset and per method, the 2x2 table (Haiku found x Opus found), Cohen's
kappa, and every disagreement with both verdicts' quotes.
"""

import json
from collections import Counter
from pathlib import Path

from evals.haiku_fact_judge import JudgeTask, JudgeVerdict, _verified

AUDIT = Path("runs/final500/judge/audit")
sample = json.loads((AUDIT / "sample.json").read_text())
tasks = {
    item["id"]: JudgeTask.model_validate(item)
    for path in sorted(AUDIT.glob("items-audit-*.json"))
    for item in json.loads(path.read_text())["items"]
}
verdicts = {
    row["id"]: JudgeVerdict.model_validate(row)
    for path in sorted(AUDIT.glob("verdict-audit-*.json"))
    for row in json.loads(path.read_text())
}
haiku = {}
for path in sorted(Path("runs/final500/judge/judge").glob("verdict-*.json")):
    for row in json.loads(path.read_text()):
        haiku[row["id"]] = row

rows = []
for n, s in enumerate(sample):
    item_id = f"a{n:03d}-{s['task']}"
    if item_id not in verdicts:
        continue
    fact = verdicts[item_id].facts[0]
    items = tasks[item_id].items
    opus_found = fact.status == "stated" and any(_verified(q, items) for q in fact.evidence)
    haiku_fact = next(f for f in haiku[s["task"]]["facts"] if f["fact"] == s["fact"])
    rows.append(
        {
            "id": item_id,
            "dataset": s["dataset"],
            "method": s["method"],
            "fact": s["fact"],
            "gold": next(iter(tasks[item_id].gold[0]["facts"].values())),
            "haiku_found": s["found"],
            "haiku_status": s["judged"],
            "opus_found": opus_found,
            "opus_status": fact.status,
            "haiku_quotes": [q["quote"][:300] for q in haiku_fact.get("evidence", [])],
            "opus_quotes": [q.quote[:300] for q in fact.evidence],
        }
    )


def summary(part: list[dict]) -> dict:
    table = Counter((r["haiku_found"], r["opus_found"]) for r in part)
    n = len(part)
    agree = table[(True, True)] + table[(False, False)]
    po = agree / n
    h = (table[(True, True)] + table[(True, False)]) / n
    o = (table[(True, True)] + table[(False, True)]) / n
    pe = h * o + (1 - h) * (1 - o)
    return {
        "n": n,
        "agree": agree,
        "agreement": round(po, 3),
        "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None,
        "haiku_found_opus_found": table[(True, True)],
        "haiku_found_opus_not": table[(True, False)],
        "haiku_not_opus_found": table[(False, True)],
        "haiku_not_opus_not": table[(False, False)],
    }


report = {
    "overall": summary(rows),
    "by_dataset": {d: summary([r for r in rows if r["dataset"] == d]) for d in sorted({r["dataset"] for r in rows})},
    "by_method": {m: summary([r for r in rows if r["method"] == m]) for m in sorted({r["method"] for r in rows})},
    "disagreements": [r for r in rows if r["haiku_found"] != r["opus_found"]],
}
(AUDIT / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
print(json.dumps({k: report[k] for k in ("overall", "by_dataset", "by_method")}, indent=1))
print(len(report["disagreements"]), "disagreements")
