"""Paired checks on the picking / top-k results. No model calls.

Same 250 MultiHop-RAG comparison + temporal questions, 3 repeats, per question type, never pooled.
Answered right and gold facts found are built exactly as runs/judge_paper/slide_data.py.

1. Is a gap real? Per question, mean answered-right over the 3 repeats for two setups; the paired
   difference with a 95% bootstrap range (questions resampled, 10,000 draws, seed 1), the 2x2
   "both / only A / only B / neither" by majority of the 3 repeats, and an exact sign test on the
   questions where exactly one setup wins. Yardstick for noise: B4 vs B4 reversed / shuffled
   (same 50 chunks, different order) and the same setup's repeat range.
2/4. Evidence in the context vs answered right: 2x2 of "all gold facts found" (and ">= half")
   against answered right, per setup; answers split into right / "Insufficient Information" / wrong;
   the same split by whether Normal RAG's top 50 holds every gold sentence (text match, as
   runs/selection/b4_analysis); yes/no gold answers vs other.
5. Does Normal RAG return the same top 50 in every repeat?

Usage (repo root, light): PYTHONPATH=. .venv/bin/python runs/selection/paired/analysis.py
  -> runs/selection/paired/analysis.json and a printed report (runs/selection/paired/report.txt)
"""
import glob
import json
import random
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

from benchmark import SavedRetrieval
from evals.qa_metrics import normalize_answer
from evals.retrieval_quality import any_overlap, load_corpus, locate_gold, ranked_items
from evals.subagent_judge import _Result, answer_key, needs_answer_judge

OUT = Path("runs/selection/paired")
QS = {f"multihop-rag-{i}" for i in json.load(open("runs/nograph/questions.json"))["indices"]}
TYPES = ["comparison_query", "temporal_query"]
ABSTAIN = normalize_answer("Insufficient Information")

verd = {}
for p in (glob.glob("runs/final500/judge/answers/verdicts-answers-*.json")
          + glob.glob("runs/nograph/judge/answers/verdicts-answers-*.json")
          + glob.glob("runs/judge_paper/answers/verdicts-answers-*.json")):
    for v in json.load(open(p)):
        verd[v["id"]] = v["correct"]


def right(row):
    if row.get("exact_match") == 1:
        return True
    r = _Result.model_validate({k: row[k] for k in ("record_id", "question", "answer", "expected_answers", "exact_match")})
    return needs_answer_judge(r) and verd[answer_key(row["question"], row["answer"])]


def loc(d):
    d = str(d)
    return Path(d[d.index("runs/"):])


final = json.load(open("runs/final500/run_registry.json"))["multihop_rag"]
REG = {"rag": final["rag"], "graphrag": final["graphrag"], "nograph": json.load(open("runs/nograph/run_registry.json"))}
REG.update(json.load(open("runs/judge_paper/runs.json")))
sel = json.load(open("runs/selection/run_registry.json"))
for name in sel:
    REG.setdefault(name, sel[name])
for name in ("B4_100", "WRONG10"):
    p = Path(f"runs/selection/paired/{name}_registry.json")
    if p.exists():
        REG[name] = json.load(open(p))[name]

run_rep = {str(loc(d)): (m, rep) for m, reps in REG.items() for rep, ds in reps.items() for d in ds}
shares = {}
for path, key in (("runs/final500/judge/per_question.jsonl", "final"), ("runs/nograph/judge/per_question.jsonl", "pilot"),
                  ("runs/judge_paper/per_question.jsonl", "paper")):
    for line in open(path):
        r = json.loads(line)
        if key == "final":
            if r["dataset"] != "multihop_rag" or r["method"] not in ("rag", "graphrag"):
                continue
            m, rep = r["method"], r["rep"]
        else:
            m, rep = (r["label"], r["rep"]) if key == "paper" else run_rep[str(loc(r["run"]))]
        if r["record_id"] in QS:
            shares[(m, rep, r["record_id"])] = r["share"]
SHARE_FROM = {"B4_rev": "B4", "B4_shuf": "B4"}

rows = {}
for m, reps in REG.items():
    for rep in "123":
        for d in reps.get(rep, []):
            for line in open(loc(d) / "results.jsonl"):
                r = json.loads(line)
                if r["record_id"] not in QS or r["question_type"] not in TYPES:
                    continue
                a = normalize_answer(r["answer"]) if r["answer"] else ""
                ok = bool(right(r))
                rows[(m, rep, r["record_id"])] = {
                    "qt": r["question_type"], "right": ok, "abstain": a == ABSTAIN and not ok,
                    "share": shares.get((SHARE_FROM.get(m, m), rep, r["record_id"])),
                    "yesno": [normalize_answer(e) for e in r["expected_answers"]] and
                             all(normalize_answer(e) in ("yes", "no") for e in r["expected_answers"]),
                    "expected": [normalize_answer(e) for e in r["expected_answers"]],
                }
qids = {qt: sorted({q for (_, _, q), v in rows.items() if v["qt"] == qt}) for qt in TYPES}
methods = sorted({m for m, _, _ in rows})
out = {}
lines = []


def say(s=""):
    lines.append(s)
    print(s)


def qmean(m, qt):
    if "+" in m:
        parts = [qmean(x, qt) for x in m.split("+")]
        return {q: sum(p[q] for p in parts) / len(parts) for q in set.intersection(*(set(p) for p in parts))}
    return {q: sum(rows[(m, rep, q)]["right"] for rep in "123" if (m, rep, q) in rows) /
            max(1, sum((m, rep, q) in rows for rep in "123")) for q in qids[qt] if (m, "1", q) in rows}


def paired(a, b, qt):
    ma, mb = qmean(a, qt), qmean(b, qt)
    qs = sorted(set(ma) & set(mb))
    d = [ma[q] - mb[q] for q in qs]
    rng = random.Random(1)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(10000))
    both = onlya = onlyb = neither = 0
    for q in qs:
        x, y = ma[q] >= 2 / 3, mb[q] >= 2 / 3
        both += x and y
        onlya += x and not y
        onlyb += y and not x
        neither += not x and not y
    n = onlya + onlyb
    k = max(onlya, onlyb)
    sign_p = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n) if n else 1.0
    return {"questions": len(qs), "mean_diff_points": 100 * sum(d) / len(d),
            "ci95": [100 * boots[250], 100 * boots[9750]],
            "both_right": both, "only_a": onlya, "only_b": onlyb, "neither": neither, "sign_test_p": sign_p}


PAIRS = [("B2", "B4"), ("B2", "rag"), ("B4", "rag"), ("B4", "B4_rev"), ("B4", "B4_shuf"), ("B4_rev", "B4_shuf"),
         ("B2", "graphrag"), ("B2", "nograph"), ("B2", "A3_picks_no_graph_rows"), ("graphrag", "nograph"),
         ("B2", "B3"), ("B3", "B4"), ("B2", "B1"), ("B4", "B4_30"), ("B4", "B4_20")]
ORDERS = "B4+B4_rev+B4_shuf"
PAIRS += [("B2", "B4_rev"), ("B2", "B4_shuf"), ("B2", ORDERS)]
if "B4_100" in methods:
    PAIRS += [("B4_100", "B4"), ("B4_100", "B2")]
if "WRONG10" in methods:
    PAIRS += [("rag", "WRONG10"), ("B4", "WRONG10")]
say("CHECK 1 and 6: paired differences (A minus B), answered right, points; 2x2 by majority of 3 repeats")
out["paired"] = {}
for a, b in PAIRS:
    if any(x not in methods for x in a.split("+") + b.split("+")):
        continue
    for qt in TYPES:
        r = paired(a, b, qt)
        out["paired"][f"{a} vs {b} {qt}"] = r
        say(f"  {a:>8} vs {b:<8} {qt[:4]}: diff {r['mean_diff_points']:+.1f} (95% {r['ci95'][0]:+.1f} to {r['ci95'][1]:+.1f})"
            f"  both {r['both_right']} only-{a} {r['only_a']} only-{b} {r['only_b']} neither {r['neither']}  sign-test p={r['sign_test_p']:.2f}")

say()
say("Repeat range of answered right (lowest-highest of 3 repeats), per setup")
for m in methods:
    for qt in TYPES:
        v = [100 * sum(rows[(m, rep, q)]["right"] for q in qids[qt] if (m, rep, q) in rows) /
             max(1, sum((m, rep, q) in rows for q in qids[qt])) for rep in "123"]
        out.setdefault("repeat_range", {}).setdefault(m, {})[qt] = v
        say(f"  {m:24} {qt[:4]} {min(v):.1f}-{max(v):.1f} (mean {sum(v) / 3:.1f})")

say()
say("CHECK 2: evidence found vs answered right (counts are per repeat, averaged over the 3 repeats)")
SETUPS = [m for m in ("rag", "B1", "B2", "B3", "B4_20", "B4_30", "B4", "B4_100", "graphrag", "nograph", "WRONG10") if m in methods]
BUCKETS = [("none (0 facts)", lambda s: s == 0), ("under half", lambda s: 0 < s < 0.5),
           ("half to all but some", lambda s: 0.5 <= s < 1 - 1e-9), ("all facts", lambda s: s >= 1 - 1e-9)]
out["evidence_vs_right"] = {}
for m in SETUPS:
    for qt in TYPES:
        say(f"  {m} {qt[:4]}")
        cell = {}
        for name, f in BUCKETS:
            n = ok = ab = 0
            for rep in "123":
                for q in qids[qt]:
                    v = rows.get((m, rep, q))
                    if v and v["share"] is not None and f(v["share"]):
                        n += 1
                        ok += v["right"]
                        ab += v["abstain"]
            cell[name] = {"questions_per_repeat": n / 3, "right_pct": 100 * ok / n if n else None,
                          "insufficient_pct": 100 * ab / n if n else None}
            say(f"    {name:22} {n / 3:6.1f} q/repeat  right {100 * ok / n if n else float('nan'):5.1f}%  said Insufficient Information {100 * ab / n if n else float('nan'):5.1f}%")
        out["evidence_vs_right"][f"{m} {qt}"] = cell

say()
say("Answer mix per setup (all questions of the type, mean of repeats): right / Insufficient Information / wrong")
out["answer_mix"] = {}
for m in SETUPS:
    for qt in TYPES:
        c = Counter()
        for rep in "123":
            for q in qids[qt]:
                v = rows.get((m, rep, q))
                if v:
                    c["right" if v["right"] else "abstain" if v["abstain"] else "wrong"] += 1
        t = sum(c.values())
        out["answer_mix"][f"{m} {qt}"] = {k: 100 * c[k] / t for k in ("right", "abstain", "wrong")}
        say(f"  {m:10} {qt[:4]} right {100 * c['right'] / t:5.1f}  insufficient {100 * c['abstain'] / t:5.1f}  wrong {100 * c['wrong'] / t:5.1f}")

say()
say("Evidence found (all gold facts) vs answered right, per repeat on average")
out["evidence_2x2"] = {}
for m in SETUPS:
    for qt in TYPES:
        c = Counter()
        for rep in "123":
            for q in qids[qt]:
                v = rows.get((m, rep, q))
                if v and v["share"] is not None:
                    c[("all" if v["share"] >= 1 - 1e-9 else "notall", "right" if v["right"] else "notright")] += 1
        cell = {f"{a}_{b}": c[(a, b)] / 3 for a in ("all", "notall") for b in ("right", "notright")}
        out["evidence_2x2"][f"{m} {qt}"] = cell
        say(f"  {m:10} {qt[:4]} all facts: right {cell['all_right']:.1f} not right {cell['all_notright']:.1f} | not all: right {cell['notall_right']:.1f} not right {cell['notall_notright']:.1f}")

say()
say("How answers change: Normal RAG answer -> other setup answer, same question and repeat number (all 3 repeats pooled, out of 375)")
out["transition"] = {}
for tgt in ("B2", "B4"):
    for qt in TYPES:
        c = Counter()
        for rep in "123":
            for q in qids[qt]:
                a, b = rows.get(("rag", rep, q)), rows.get((tgt, rep, q))
                if a and b:
                    oc = lambda v: "right" if v["right"] else "insufficient" if v["abstain"] else "wrong"
                    c[(oc(a), oc(b))] += 1
        cell = {f"{a}->{b}": c[(a, b)] for a in ("right", "insufficient", "wrong") for b in ("right", "insufficient", "wrong")}
        out["transition"][f"rag->{tgt} {qt}"] = cell
        say(f"  rag->{tgt} {qt[:4]} " + "  ".join(f"{k}:{v}" for k, v in cell.items()))

say()
say("CHECK 4: does a question whose Normal RAG top 50 lacks gold still get answered right? (text match, every setup)")
corpus = load_corpus("multihop_rag")
base = {}
for d in REG["B4"]["1"]:
    for line in open(loc(d) / "retrieval.jsonl"):
        s = SavedRetrieval.model_validate_json(line)
        if s.record_id in QS:
            base[s.record_id] = s
complete = {}
for q, s in base.items():
    items = ranked_items(s, corpus)
    golds = locate_gold(s.supporting_sentences, corpus)
    complete[q] = all(any_overlap(g, [sp for it in items for sp in it.spans]) for g in golds)
exp = {q: rows[("B4", "1", q)]["expected"] for q in complete}
out["incomplete"] = {}
for qt in TYPES:
    qs = qids[qt]
    n_c = sum(complete[q] for q in qs)
    yn_all = sum(rows[("B4", "1", q)]["yesno"] for q in qs)
    yn_inc = sum(rows[("B4", "1", q)]["yesno"] for q in qs if not complete[q])
    say(f"  {qt[:4]}: complete {n_c}, incomplete {len(qs) - n_c}; yes/no gold answers: {yn_all} of {len(qs)} overall, {yn_inc} of the {len(qs) - n_c} incomplete")
    ans = Counter(e for q in qs for e in exp[q] if rows[("B4", "1", q)]["yesno"])
    say(f"     gold yes/no split: {dict(ans)}")
    for m in [x for x in SETUPS if x not in ("WRONG10",)] + (["WRONG10"] if "WRONG10" in methods else []):
        for name, flag in (("complete", True), ("incomplete", False)):
            sub = [q for q in qs if complete[q] == flag]
            sub_yn = [q for q in sub if rows[("B4", "1", q)]["yesno"]]
            sub_other = [q for q in sub if not rows[("B4", "1", q)]["yesno"]]

            def rate(group):
                tot = [(rep, q) for rep in "123" for q in group if (m, rep, q) in rows]
                return 100 * sum(rows[(m, rep, q)]["right"] for rep, q in tot) / len(tot) if tot else float("nan")
            ab = [(rep, q) for rep in "123" for q in sub if (m, rep, q) in rows]
            ab_pct = 100 * sum(rows[(m, rep, q)]["abstain"] for rep, q in ab) / len(ab) if ab else float("nan")
            out["incomplete"][f"{m} {qt} {name}"] = {"n": len(sub), "right_pct": rate(sub), "yesno_n": len(sub_yn),
                                                    "yesno_right_pct": rate(sub_yn), "other_n": len(sub_other),
                                                    "other_right_pct": rate(sub_other), "insufficient_pct": ab_pct}
            say(f"     {m:10} {name:10} n={len(sub):3} right {rate(sub):5.1f}%  | yes/no n={len(sub_yn):3} {rate(sub_yn):5.1f}%  other n={len(sub_other):3} {rate(sub_other):5.1f}%  | insufficient {ab_pct:5.1f}%")

say()
say("CHECK 5: same top 50 in every repeat? (B4 repeats 2 and 3 against repeat 1)")
reps50 = {}
for rep in "123":
    reps50[rep] = {}
    for d in REG["B4"][rep]:
        for line in open(loc(d) / "retrieval.jsonl"):
            s = SavedRetrieval.model_validate_json(line)
            if s.record_id in QS:
                reps50[rep][s.record_id] = s
out["same_top50"] = {}
for rep in "23":
    same = reorder = differ = 0
    changed, first_pos, gold_changed = [], [], 0
    for q in sorted(QS):
        a, b = reps50["1"][q].retrieval.passages, reps50[rep][q].retrieval.passages
        if a == b:
            same += 1
        elif set(a) == set(b):
            reorder += 1
        else:
            differ += 1
            changed.append(len(set(a) - set(b)))
            first_pos.append(next(i for i, (x, y) in enumerate(zip(a, b)) if x != y))
            ga = locate_gold(reps50["1"][q].supporting_sentences, corpus)
            ca = all(any_overlap(g, [sp for it in ranked_items(reps50["1"][q], corpus) for sp in it.spans]) for g in ga)
            cb = all(any_overlap(g, [sp for it in ranked_items(reps50[rep][q], corpus) for sp in it.spans]) for g in ga)
            gold_changed += ca != cb
    r = {"identical": same, "same_set_other_order": reorder, "different_chunks": differ,
         "chunks_swapped_per_changed_question": sorted(Counter(changed).items()),
         "first_difference_at_rank": sorted(Counter(first_pos).items()), "gold_completeness_changed": gold_changed}
    out["same_top50"][f"rep{rep}"] = r
    say(f"  repeat {rep}: identical {same}, same chunks in another order {reorder}, different chunks {differ}; "
        f"chunks swapped {r['chunks_swapped_per_changed_question']}; first difference at rank {r['first_difference_at_rank']}; "
        f"questions whose 'all gold in top 50' changed {gold_changed}")
rag10 = {}
for rep in "123":
    rag10[rep] = {}
    for d in REG["rag"][rep]:
        for line in open(loc(d) / "retrieval.jsonl"):
            s = SavedRetrieval.model_validate_json(line)
            if s.record_id in QS:
                rag10[rep][s.record_id] = s.retrieval.passages
for rep in "23":
    same = sum(rag10["1"][q] == rag10[rep][q] for q in QS)
    sets = sum(set(rag10["1"][q]) == set(rag10[rep][q]) for q in QS)
    say(f"  Normal RAG top 10 (final500): repeat {rep} identical to repeat 1 for {same} of {len(QS)}, same set {sets}")
    out["same_top50"][f"rag10_rep{rep}"] = {"identical": same, "same_set": sets}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
(OUT / "report.txt").write_text("\n".join(lines) + "\n")
