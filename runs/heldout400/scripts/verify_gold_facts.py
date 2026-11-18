"""Independent check of gold_facts_by_run.json (does not reuse heldout_analysis.py's loader).

1. Coverage: every repeat x method has exactly the 100 questions per type from questions.json,
   judged once in judge_support, and the judged run folder belongs to that repeat and method.
2. Duplicates: questions answered in more than one run folder of the same repeat and method.
3. Recompute Answered right (exact match, else Haiku verdict; later verdict files override) and
   gold facts found (share of claims with full_status == "stated") and compare every number.
4. Cross-check Answered right against the older confusion_by_run.json (written 8 Nov).
Run from the repo root: PYTHONPATH=. uv run python runs/heldout400/scripts/verify_gold_facts.py
"""
import glob, json
from collections import Counter, defaultdict
from pathlib import Path

from evals.subagent_judge import answer_key

H = Path("runs/heldout400")
Q = json.load(open(H / "questions.json"))
qtype = {f"multihop-rag-{i}": t for t, ids in Q["by_type"].items() for i in ids}
print("questions.json:", Counter(qtype.values()))
REPS, METHODS = ["r1", "r2", "r3"], ["rag", "raptor", "hipporag", "graphrag"]
problems = []

# --- answers, straight from results.jsonl ---
verdicts = {}
for f in sorted(glob.glob(str(H / "judge" / "verdicts-*.json"))):
    for v in json.load(open(f)):
        verdicts[v["id"]] = v
answers = defaultdict(dict)       # (rep, method) -> rid -> (run_dir, correct)
dupes = defaultdict(list)
run_of = {}
for m in map(json.loads, open(H / "manifest.jsonl")):
    if m["tag"] not in REPS or not m["run_dir"]:
        continue
    key = (m["tag"], m["method"])
    rel = "/".join(Path(m["run_dir"]).parts[-3:])
    run_of[rel] = key
    for r in map(json.loads, open(Path(m["run_dir"]) / "results.jsonl")):
        v = verdicts.get(answer_key(r["question"], r["answer"]))
        ok = bool(r["exact_match"]) or bool(v and v.get("correct"))
        if r["record_id"] in answers[key]:
            dupes[key].append((r["record_id"], answers[key][r["record_id"]], (rel, ok)))
        answers[key][r["record_id"]] = (rel, ok)
        if r["question_type"] != qtype.get(r["record_id"]):
            problems.append(f"type mismatch {r['record_id']}")
for key, d in dupes.items():
    flips = [x for x in d if x[1][1] != x[2][1]]
    print(f"duplicate answers {key}: {len(d)} questions answered twice; correctness differs on {len(flips)}")
for key in [(t, m) for t in REPS for m in METHODS]:
    if set(answers[key]) != set(qtype):
        problems.append(f"answers {key}: {len(answers[key])} questions, expected 400")

# --- gold facts, straight from judge_support ---
facts = defaultdict(dict)         # (rep, method) -> rid -> share found
judged_run = defaultdict(dict)
for t in REPS:
    for s in range(4):
        for r in map(json.loads, open(H / "judge_support" / f"{t}-s{s}" / "per_question.jsonl")):
            key = (t, r["method"])
            if r["record_id"] in facts[key]:
                problems.append(f"judged twice {key} {r['record_id']}")
            claims = [c for g in r["gold"] for c in g["claims"]]
            facts[key][r["record_id"]] = sum(c["full_status"] == "stated" for c in claims) / len(claims) if claims else None
            judged_run[key][r["record_id"]] = r["run"]
            if r["question_type"] != qtype.get(r["record_id"]):
                problems.append(f"judge type mismatch {r['record_id']}")
nonnull = {q for q, ty in qtype.items() if ty != "null_query"}
mismatch_run = Counter()
for key in [(t, m) for t in REPS for m in METHODS]:
    if set(facts[key]) != nonnull:
        problems.append(f"judge {key}: {len(facts[key])} questions, expected 300")
    for q, run in judged_run[key].items():
        rel = "/".join(Path(run).parts[-3:])
        if rel in run_of and run_of[rel] != key:
            problems.append(f"judged run {rel} belongs to {run_of[rel]}, not {key}")
        if rel in run_of and answers[key][q][0] != rel:
            mismatch_run[key] += 1
        if rel not in run_of:
            mismatch_run[(key, "run not in manifest: " + rel.split("/")[-1])] += 1
print("judged run differs from the run whose answer is counted:", dict(mismatch_run) or "none")

# --- recompute and compare ---
G = json.load(open(H / "gold_facts_by_run.json"))
def ok(m, t, q, ms):
    if ms == "answered_right":
        return answers[(t, m)][q][1]
    f = facts[(t, m)].get(q)
    return f is not None and f >= float(ms) - 1e-9
checked = 0
for qt in ["comparison_query", "temporal_query", "inference_query", "null_query"]:
    qs = sorted(q for q in qtype if qtype[q] == qt)
    meas = ["answered_right"] + ([] if qt == "null_query" else ["0.5", "0.75", "1.0"])
    for ms in meas:
        for m in METHODS:
            for t in REPS:
                checked += 1
                if sum(ok(m, t, q, ms) for q in qs) != G["per_method"][qt][m][t][ms]:
                    problems.append(f"per_method {qt} {m} {t} {ms}")
            k = Counter(sum(ok(m, t, q, ms) for t in REPS) for q in qs)
            checked += 1
            if [k[3], k[2], k[1], k[0]] != G["repeats_found"][qt][m][ms]:
                problems.append(f"repeats_found {qt} {m} {ms}")
        for t in REPS:
            c = Counter((ok("graphrag", t, q, ms), ok("rag", t, q, ms)) for q in qs)
            checked += 1
            if [c[(True, True)], c[(True, False)], c[(False, True)], c[(False, False)]] != G["pair"][qt]["rag"][t][ms]:
                problems.append(f"pair {qt} {t} {ms}")
        f3 = [sum(all(not ok("graphrag", t, q, ms) and not ok("rag", t, q, ms) for t in REPS) for q in qs),
              sum(all(not ok("graphrag", t, q, ms) and ok("rag", t, q, ms) for t in REPS) for q in qs),
              sum(all(ok("graphrag", t, q, ms) and not ok("rag", t, q, ms) for t in REPS) for q in qs)]
        checked += 1
        if f3 != G["fails_all3"][qt]["rag"][ms]:
            problems.append(f"fails_all3 {qt} {ms}: {f3} vs {G['fails_all3'][qt]['rag'][ms]}")
# older file, answers only
C = json.load(open(H / "confusion_by_run.json"))
for qt in C["pair_answer"]:
    for t in REPS:
        checked += 1
        if C["pair_answer"][qt]["rag"][t] != G["pair"][qt]["rag"][t]["answered_right"]:
            problems.append(f"old confusion_by_run differs {qt} {t}")
    for m in METHODS:
        checked += 1
        if C["consistency_answer"][qt][m] != G["repeats_found"][qt][m]["answered_right"]:
            problems.append(f"old consistency differs {qt} {m}")
print(f"{checked} checks;", "PROBLEMS:\n  " + "\n  ".join(problems) if problems else "no problems")
