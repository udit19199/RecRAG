"""Per-run, per-type confusion matrices for the held-out 400 study.

Writes runs/heldout400/confusion_by_run.json. No pooling across runs, no cost or latency.

  pair_answer[type][baseline][run]           = [both correct, Agentic only, baseline only, both wrong]
  pair_retrieval[type][baseline][run][th]    = [both found, Agentic only, baseline only, both missed]
      found = gold-sentence recall >= th (0.5, 0.75, 1.0); null questions have no gold facts
  consistency_answer[type][method]           = questions right in [3, 2, 1, 0] of the 3 runs
  consistency_retrieval[type][method][th]    = questions found in [3, 2, 1, 0] of the 3 runs
  wrong_all_runs[type][baseline]             = [Agentic wrong x3 & baseline right x3,
                                                baseline wrong x3 & Agentic right x3, both wrong x3]

Run from the repo root: PYTHONPATH=. uv run python runs/heldout400/scripts/heldout_confusion_by_run.py
"""
import json
from collections import Counter
from pathlib import Path

# heldout_analysis.py rewrites analysis.json when run in full; only load its scoring part.
src = Path("runs/heldout400/scripts/heldout_analysis.py").read_text()
ns = {"__name__": "heldout_confusion_by_run"}
exec(src[: src.index("def per_question")], ns)
runs, reps, NAMES = ns["runs"], ns["reps"], ns["NAMES"]

TYPES = ["comparison_query", "temporal_query", "inference_query", "null_query"]
BASE = ["rag", "hipporag", "raptor"]
THS = (0.5, 0.75, 1.0)
AG = "graphrag"


def cells(c):
    return [c[(True, True)], c[(True, False)], c[(False, True)], c[(False, False)]]


def by_count(k):
    return [k[3], k[2], k[1], k[0]]


out = {"runs": reps, "questions_per_run": {}, "pair_answer": {}, "pair_retrieval": {},
       "consistency_answer": {}, "consistency_retrieval": {}, "wrong_all_runs": {}}

for qt in TYPES:
    qs = sorted({q for t in reps for q, r in runs[(t, AG)].items() if r["type"] == qt})
    for t in reps:
        for m in NAMES:
            missing = [q for q in qs if q not in runs[(t, m)]]
            assert not missing, f"{m} {t} {qt} missing {len(missing)} questions"
    out["questions_per_run"][qt] = len(qs)
    has_gold = qt != "null_query"
    found = lambda r, th: r["recall"] is not None and r["recall"] >= th - 1e-9
    right = lambda m, t, q: bool(runs[(t, m)][q]["correct"])

    pa, pr = out["pair_answer"].setdefault(qt, {}), out["pair_retrieval"].setdefault(qt, {})
    for b in BASE:
        for t in reps:
            A, B = runs[(t, AG)], runs[(t, b)]
            pa.setdefault(b, {})[t] = cells(Counter((right(AG, t, q), right(b, t, q)) for q in qs))
            if has_gold:
                pr.setdefault(b, {})[t] = {
                    str(th): cells(Counter((found(A[q], th), found(B[q], th)) for q in qs)) for th in THS}
        out["wrong_all_runs"].setdefault(qt, {})[b] = [
            sum(all(not right(AG, t, q) and right(b, t, q) for t in reps) for q in qs),
            sum(all(right(AG, t, q) and not right(b, t, q) for t in reps) for q in qs),
            sum(all(not right(AG, t, q) and not right(b, t, q) for t in reps) for q in qs)]

    for m in NAMES:
        out["consistency_answer"].setdefault(qt, {})[m] = by_count(
            Counter(sum(right(m, t, q) for t in reps) for q in qs))
        if has_gold:
            out["consistency_retrieval"].setdefault(qt, {})[m] = {
                str(th): by_count(Counter(sum(found(runs[(t, m)][q], th) for t in reps) for q in qs))
                for th in THS}

# Answer support (gpt-6-luna judge, 0-3): "stated" = score >= 2, "complete" = score 3.
# Questions the judge failed on for either method are left out and counted in support_missing.
SUP = Path("runs/heldout400/judge_support")
LEVELS = {"stated": 2, "complete": 3}
support = {}  # (run, method) -> record_id -> score
for t in reps:
    for s in range(4):
        f = SUP / f"{t}-s{s}" / "per_question.jsonl"
        if (SUP / f"{t}-s{s}" / "DONE").exists() and f.exists():
            for line in open(f):
                r = json.loads(line)
                support.setdefault((t, r["method"]), {})[r["record_id"]] = (r["question_type"], r["support_score"])
out["support_shards_done"] = sum((SUP / f"{t}-s{s}" / "DONE").exists() for t in reps for s in range(4))
out["pair_support"], out["consistency_support"], out["support_missing"] = {}, {}, {}
for qt in TYPES:
    qs = sorted({q for t in reps for q, (ty, _) in support.get((t, AG), {}).items() if ty == qt})
    if not qs:
        continue
    sc = lambda m, t, q: support.get((t, m), {}).get(q, (None, None))[1]
    for b in BASE:
        for t in reps:
            both = [q for q in qs if sc(AG, t, q) is not None and sc(b, t, q) is not None]
            out["support_missing"].setdefault(qt, {}).setdefault(b, {})[t] = out["questions_per_run"][qt] - len(both)
            out["pair_support"].setdefault(qt, {}).setdefault(b, {})[t] = {
                lv: cells(Counter((sc(AG, t, q) >= k, sc(b, t, q) >= k) for q in both)) for lv, k in LEVELS.items()}
    for m in NAMES:
        full = [q for q in qs if all(sc(m, t, q) is not None for t in reps)]
        out["consistency_support"].setdefault(qt, {})[m] = {
            lv: by_count(Counter(sum(sc(m, t, q) >= k for t in reps) for q in full)) for lv, k in LEVELS.items()}
        out["consistency_support"][qt][m]["questions"] = len(full)

dest = Path("runs/heldout400/confusion_by_run.json")
dest.write_text(json.dumps(out, indent=1))
print(f"wrote {dest}")
