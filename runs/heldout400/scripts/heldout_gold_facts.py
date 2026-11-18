"""Held-out 400: Answered right and gold facts found, per question type and per repeat.

Answered right: exact match, then the Claude Haiku judge (scoring from heldout_analysis.py).
Gold facts found: the share of a question's gold facts (claims the gpt-6-luna judge extracted
from the gold sentences) whose full_status is "stated", i.e. stated with a quote that appears
in the retrieved text. Thresholds >=50%, >=75%, 100%. Null questions have no gold facts.
No pooling across repeats or types, no cost or latency.

Writes runs/heldout400/gold_facts_by_run.json:
  per_method[type][method][repeat]      = {"answered_right": n, "0.5": n, "0.75": n, "1.0": n}
  pair[type][baseline][repeat][measure] = [both, Agentic GraphRAG only, baseline only, neither]
  fails_all3[type][baseline][measure]   = [both, Agentic GraphRAG only, baseline only]
  repeats_found[type][method][measure]  = questions right/found in [3, 2, 1, 0] repeats
  no_gold_facts[type][method][repeat]   = questions whose judge record has no gold facts

Run from the repo root: PYTHONPATH=. uv run python runs/heldout400/scripts/heldout_gold_facts.py
"""
import json
from collections import Counter
from pathlib import Path

src = Path("runs/heldout400/scripts/heldout_analysis.py").read_text()
ns = {"__name__": "heldout_gold_facts"}
exec(src[: src.index("def per_question")], ns)
runs, reps, NAMES = ns["runs"], ns["reps"], ns["NAMES"]

TYPES = ["comparison_query", "temporal_query", "inference_query", "null_query"]
AG, BASE = "graphrag", ["rag", "raptor", "hipporag"]
THS = ("0.5", "0.75", "1.0")
SUP = Path("runs/heldout400/judge_support")

frac = {}  # (repeat, method) -> record_id -> share of gold facts found (None if no facts)
for t in reps:
    for s in range(4):
        d = SUP / f"{t}-s{s}"
        assert (d / "DONE").exists(), f"{d} not finished"
        for line in open(d / "per_question.jsonl"):
            r = json.loads(line)
            claims = [c for g in r["gold"] for c in g["claims"]]
            f = sum(c["full_status"] == "stated" for c in claims) / len(claims) if claims else None
            frac.setdefault((t, r["method"]), {})[r["record_id"]] = f


def ok(m, t, q, measure):
    if measure == "answered_right":
        return bool(runs[(t, m)][q]["correct"])
    f = frac[(t, m)].get(q)
    return f is not None and f >= float(measure) - 1e-9


out = {"repeats": reps, "questions": {}, "per_method": {}, "pair": {}, "fails_all3": {},
       "repeats_found": {}, "no_gold_facts": {}}
for qt in TYPES:
    qs = sorted(q for q, r in runs[(reps[0], AG)].items() if r["type"] == qt)
    out["questions"][qt] = len(qs)
    measures = ["answered_right"] + ([] if qt == "null_query" else list(THS))
    for m in NAMES:
        for t in reps:
            assert all(q in runs[(t, m)] for q in qs), f"{m} {t} {qt} missing questions"
            if qt != "null_query":
                missing = [q for q in qs if q not in frac[(t, m)]]
                assert not missing, f"{m} {t} {qt}: {len(missing)} questions not judged"
                out["no_gold_facts"].setdefault(qt, {}).setdefault(m, {})[t] = sum(
                    frac[(t, m)][q] is None for q in qs)
            out["per_method"].setdefault(qt, {}).setdefault(m, {})[t] = {
                ms: sum(ok(m, t, q, ms) for q in qs) for ms in measures}
        k = {ms: Counter(sum(ok(m, t, q, ms) for t in reps) for q in qs) for ms in measures}
        out["repeats_found"].setdefault(qt, {})[m] = {ms: [k[ms][i] for i in (3, 2, 1, 0)] for ms in measures}
    for b in BASE:
        for t in reps:
            out["pair"].setdefault(qt, {}).setdefault(b, {})[t] = {}
            for ms in measures:
                c = Counter((ok(AG, t, q, ms), ok(b, t, q, ms)) for q in qs)
                out["pair"][qt][b][t][ms] = [c[(True, True)], c[(True, False)], c[(False, True)], c[(False, False)]]
        out["fails_all3"].setdefault(qt, {})[b] = {ms: [
            sum(all(not ok(AG, t, q, ms) and not ok(b, t, q, ms) for t in reps) for q in qs),
            sum(all(not ok(AG, t, q, ms) for t in reps) and all(ok(b, t, q, ms) for t in reps) for q in qs),
            sum(all(not ok(b, t, q, ms) for t in reps) and all(ok(AG, t, q, ms) for t in reps) for q in qs),
        ] for ms in measures}

dest = Path("runs/heldout400/gold_facts_by_run.json")
dest.write_text(json.dumps(out, indent=1))
print(f"wrote {dest}")
