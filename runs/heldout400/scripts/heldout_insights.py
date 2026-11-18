"""Every extra reading of the held-out run (400 questions x 3 repeats x 4 methods).

No model calls. Reuses heldout_analysis for scoring, so accuracy and recall
match analysis.json. Sections follow the analysis menu: significance,
stability, failure breakdown, hard questions, cost, Agentic GraphRAG
internals, routing, measurement checks. If runs/heldout400/rejudge/verdicts-*
exist, recall is also rescored with those strict re-judged verdicts.
Writes runs/heldout400/insights.json and a hand-label sample."""
import glob, io, contextlib, json, math, random, re, runpy, statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from evals.qa_metrics import normalize_answer
from evals.retrieval_quality import any_overlap, graphrag_items, load_run, locate_gold

with contextlib.redirect_stdout(io.StringIO()):
    A = runpy.run_path("runs/heldout400/scripts/heldout_analysis.py")
runs, reps, NAMES, OUT, corpus = A["runs"], A["reps"], A["NAMES"], A["OUT"], A["corpus"]
M = list(NAMES)
summary = {m["method"]: m for m in A["summary"]["methods"]}
manifest = [m for m in A["manifest"] if m["run_dir"] and m["tag"] != "latency"]

# Extra per-answer fields the analysis rows do not keep.
extra = {}  # (tag, method, qid) -> dict
for m in manifest:
    for r in map(json.loads, open(Path(m["run_dir"]) / "results.jsonl")):
        extra[(m["tag"], m["method"], r["record_id"])] = dict(
            question=r["question"], gold=r["expected_answers"], answer=r["answer"],
            norm=normalize_answer(r["answer"]), golds=len(r["supporting_sentences"]),
            overlap=r["overlap_accuracy"] or 0.0,
            calls=sum(c["kind"] == "responses" for c in r["question_cost"]["calls"]),
            graph_rows=sum(s == -1.0 for s in r["retrieval_scores"] or []))
for (t, m), rows in runs.items():
    if t in reps:
        for q, row in rows.items():
            row.update(extra[(t, m, q)])
            row["outcome"] = "right" if row["correct"] else "refused" if row["abstain"] else "false"

qids = sorted({q for (t, m) in runs if t in reps for q in runs[(t, m)]})
qtype = {q: runs[(reps[0], M[0])][q]["type"] for q in qids}
TYPES = sorted(set(qtype.values()))
def rows(m, qt=None):
    return [r for t in reps for r in runs[(t, m)].values() if qt is None or r["type"] == qt]
def mean_q(m, q, key="correct"):
    return st.mean(runs[(t, m)][q][key] for t in reps)
majority = {m: {q: mean_q(m, q) >= 0.5 for q in qids} for m in M}
pct = lambda x: round(100 * x, 1)
R = {}

# 1. Significance: exact McNemar on majority-of-3 correctness, Holm-corrected.
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
tests = []
for other in ("rag", "hipporag", "raptor"):
    for qt in [None, *TYPES]:
        qs = [q for q in qids if qt is None or qtype[q] == qt]
        b = sum(majority["graphrag"][q] and not majority[other][q] for q in qs)
        c = sum(majority[other][q] and not majority["graphrag"][q] for q in qs)
        d = [mean_q("graphrag", q) - mean_q(other, q) for q in qs]
        tests.append(dict(vs=NAMES[other], type=qt or "all", questions=len(qs), graphrag_only_right=b,
                          other_only_right=c, p=mcnemar(b, c), wins=sum(x > 0 for x in d),
                          ties=sum(x == 0 for x in d), losses=sum(x < 0 for x in d)))
order = sorted(range(len(tests)), key=lambda i: tests[i]["p"])
running = 0.0
for rank, i in enumerate(order):  # Holm step-down
    running = max(running, min(1.0, (len(tests) - rank) * tests[i]["p"]))
    tests[i]["p_holm"] = running
R["significance"] = tests

# 2. Stability across the 3 repeats.
R["stability"] = []
for m in M:
    k = Counter(sum(runs[(t, m)][q]["correct"] for t in reps) for q in qids)
    same = sum(len({runs[(t, m)][q]["norm"] for t in reps}) == 1 for q in qids)
    R["stability"].append(dict(method=NAMES[m], same_answer_all_repeats=pct(same / len(qids)),
        right_3of3=k[3], right_2of3=k[2], right_1of3=k[1], right_0of3=k[0],
        right_at_least_once=pct((len(qids) - k[0]) / len(qids)), right_every_time=pct(k[3] / len(qids)),
        accuracy_by_repeat=[pct(x) for x in summary[NAMES[m]]["accuracy_by_repeat"]]))

# 3. Failure breakdown: evidence level x answer outcome; null and yes/no behaviour.
def level(r):
    return "none" if r["recall"] == 0 else "all" if r["recall"] >= 1 - 1e-9 else "some"
R["evidence_outcome"] = []
for m in M:
    for qt in [None, *[t for t in TYPES if t != "null_query"]]:
        sub = [r for r in rows(m, qt) if r["recall"] is not None]
        c = Counter((level(r), r["outcome"]) for r in sub)
        R["evidence_outcome"].append(dict(method=NAMES[m], type=qt or "all (non-null)", answers=len(sub),
            **{f"{e}_{o}": c[(e, o)] for e in ("none", "some", "all") for o in ("right", "refused", "false")}))
R["null_questions"] = [dict(method=NAMES[m], answers=len(rows(m, "null_query")),
    refused_correctly=sum(r["outcome"] == "right" for r in rows(m, "null_query")),
    answered_anyway=sum(r["outcome"] != "right" for r in rows(m, "null_query")),
    examples=sorted({r["answer"] for r in rows(m, "null_query") if r["outcome"] != "right"})[:8]) for m in M]
yesno = [r for r in rows("rag") if normalize_answer(r["gold"][0]) in ("yes", "no")]
R["yes_no"] = {"gold_yes_share": pct(st.mean(normalize_answer(r["gold"][0]) == "yes" for r in yesno)), "methods": []}
for m in M:
    sub = [r for r in rows(m) if normalize_answer(r["gold"][0]) in ("yes", "no")]
    first = Counter(r["norm"].split()[0] if r["norm"] else "" for r in sub)
    R["yes_no"]["methods"].append(dict(method=NAMES[m], answers=len(sub), said_yes=pct(first["yes"] / len(sub)),
        said_no=pct(first["no"] / len(sub)), refused=pct(st.mean(r["abstain"] for r in sub)),
        accuracy_gold_yes=pct(st.mean(r["correct"] for r in sub if normalize_answer(r["gold"][0]) == "yes")),
        accuracy_gold_no=pct(st.mean(r["correct"] for r in sub if normalize_answer(r["gold"][0]) == "no"))))

# 4. Hard questions and method agreement.
all_fail = [q for q in qids if not any(majority[m][q] for m in M)]
R["hard_questions"] = dict(
    all_methods_fail=len(all_fail), all_methods_fail_by_type=Counter(qtype[q] for q in all_fail),
    all_methods_right=sum(all(majority[m][q] for m in M) for q in qids),
    only_this_method_right={NAMES[m]: sum(majority[m][q] and not any(majority[o][q] for o in M if o != m) for q in qids) for m in M},
    all_fail_examples=[dict(id=q, type=qtype[q], question=runs[(reps[0], "rag")][q]["question"],
                            gold=runs[(reps[0], "rag")][q]["gold"],
                            answers={NAMES[m]: runs[(reps[0], m)][q]["answer"][:120] for m in M}) for q in all_fail])
R["accuracy_by_gold_count"] = [dict(method=NAMES[m], **{f"{g}_gold": pct(st.mean(r["correct"] for r in rows(m) if r["golds"] == g and r["type"] != "null_query"))
                                                     for g in (2, 3, 4)}) for m in M]
R["gold_count_questions"] = dict(Counter(runs[(reps[0], "rag")][q]["golds"] for q in qids if qtype[q] != "null_query"))
def kappa(a, b):
    x, y = [majority[a][q] for q in qids], [majority[b][q] for q in qids]
    po = st.mean(i == j for i, j in zip(x, y)); px, py = st.mean(x), st.mean(y)
    pe = px * py + (1 - px) * (1 - py)
    return round((po - pe) / (1 - pe), 3)
R["agreement_kappa"] = [dict(a=NAMES[a], b=NAMES[b], kappa=kappa(a, b),
                             both_wrong=sum(not majority[a][q] and not majority[b][q] for q in qids))
                        for i, a in enumerate(M) for b in M[i + 1:]]

# 5. Cost and speed.
R["tradeoff"] = [dict(method=n, accuracy=pct(s["accuracy"]), query_usd=s["query_usd_list"], build_usd=s["build_usd"],
                      latency_median=s["latency_median"], latency_p90=s["latency_p90"], context_tokens=round(s["context_tokens"]),
                      usd_per_1000_correct=s["usd_per_1000_correct"]) for n, s in summary.items()]
R["total_cost_usd"] = {NAMES[m]: {n: round(summary[NAMES[m]]["build_usd"] + n * summary[NAMES[m]]["query_usd_list"], 2)
                                  for n in (400, 2556, 10000, 100000)} for m in M}
R["cost_by_outcome"] = [dict(method=NAMES[m], **{f"{o}_{k}": round(st.mean(r[f] for r in rows(m) if r["outcome"] == o), 5 if f == "usd" else 1)
                                                for o in ("right", "refused", "false") for k, f in (("usd", "usd"), ("sec", "sec"), ("tokens", "ctx"))
                                                if any(r["outcome"] == o for r in rows(m))}) for m in M]
R["accuracy_by_context_tercile"] = []
for m in M:
    sub = sorted(rows(m), key=lambda r: r["ctx"]); n = len(sub) // 3
    parts = [sub[:n], sub[n:2 * n], sub[2 * n:]]
    R["accuracy_by_context_tercile"].append(dict(method=NAMES[m],
        **{f"{name}_tokens": f"{p[0]['ctx']}-{p[-1]['ctx']}" for name, p in zip(("low", "mid", "high"), parts)},
        **{f"{name}_accuracy": pct(st.mean(r["correct"] for r in p)) for name, p in zip(("low", "mid", "high"), parts)}))
R["failures"] = [dict(method=n, failed_questions=s["failed_questions"], failed_calls=s["failed_calls"]) for n, s in summary.items()]

# 6. Agentic GraphRAG internals.
g = rows("graphrag")
by_calls = defaultdict(list)
for r in g:
    by_calls[min(r["calls"], 6)].append(r)
R["graphrag"] = dict(
    graph_rows_returned=pct(st.mean(r["graph_rows"] > 0 for r in g)),
    accuracy_with_graph_rows=pct(st.mean(r["correct"] for r in g if r["graph_rows"] > 0)),
    accuracy_without_graph_rows=pct(st.mean(r["correct"] for r in g if r["graph_rows"] == 0)),
    # Per type, because null questions rarely get graph rows and are mostly right.
    by_type=[dict(type=qt, graph_rows_share=pct(st.mean(r["graph_rows"] > 0 for r in sub)),
                  accuracy_with_rows=pct(st.mean([r["correct"] for r in sub if r["graph_rows"] > 0] or [0])),
                  accuracy_without_rows=pct(st.mean([r["correct"] for r in sub if r["graph_rows"] == 0] or [0])),
                  accuracy_5_calls=pct(st.mean([r["correct"] for r in sub if r["calls"] <= 5] or [0])),
                  accuracy_6plus_calls=pct(st.mean([r["correct"] for r in sub if r["calls"] > 5] or [0])),
                  share_6plus_calls=pct(st.mean(r["calls"] > 5 for r in sub)))
             for qt in TYPES for sub in [[r for r in g if r["type"] == qt]]],
    mean_graph_rows=round(st.mean(r["graph_rows"] for r in g), 2),
    by_model_calls=[dict(calls=f"{k}+" if k == 6 else k, answers=len(v), accuracy=pct(st.mean(r["correct"] for r in v)),
                         usd=round(st.mean(r["usd"] for r in v), 5)) for k, v in sorted(by_calls.items())])
# Gold sentences found by vector passages alone vs. with graph rows (offset match only, before the judge).
vec_only = with_graph = total = 0
from graphrag_retrieval import parse_vector_evidence
for m in manifest:
    if m["method"] != "graphrag":
        continue
    for saved in load_run(Path(m["run_dir"])):
        golds = locate_gold(saved.supporting_sentences, corpus)
        if not golds:
            continue
        vec = saved.model_copy(update={"retrieval": saved.retrieval.model_copy(update={
            "passages": [p for p in saved.retrieval.passages if parse_vector_evidence(p) is not None]})})
        s_all = [s for it in graphrag_items(saved, corpus) for s in it.spans]
        s_vec = [s for it in graphrag_items(vec, corpus) for s in it.spans]
        total += len(golds)
        with_graph += sum(any_overlap(x, s_all) for x in golds)
        vec_only += sum(any_overlap(x, s_vec) for x in golds)
R["graphrag"].update(gold_sentences=total, found_by_vector_only=vec_only, found_with_graph_rows=with_graph,
                     added_by_graph_text=with_graph - vec_only)

# 7. Routing and combining.
def evaluate(pick):  # pick(t, q) -> (correct, usd)
    out = [pick(t, q) for t in reps for q in qids]
    return dict(accuracy=pct(st.mean(c for c, _ in out)), usd=round(st.mean(u for _, u in out), 5))
best_type = {qt: max(M, key=lambda m: st.mean(r["correct"] for r in rows(m, qt))) for qt in TYPES}
plans = {f"always {NAMES[m]}": (lambda m: lambda t, q: (runs[(t, m)][q]["correct"], runs[(t, m)][q]["usd"]))(m) for m in M}
plans["best method per question type (" + ", ".join(f"{k.split('_')[0]}->{NAMES[v]}" for k, v in best_type.items()) + ")"] = \
    lambda t, q: (runs[(t, best_type[qtype[q]])][q]["correct"], runs[(t, best_type[qtype[q]])][q]["usd"])
plans["oracle (any method right; cost of all four)"] = lambda t, q: (max(runs[(t, m)][q]["correct"] for m in M), sum(runs[(t, m)][q]["usd"] for m in M))
for first in ("rag", "hipporag"):
    for second in [m for m in M if m != first]:
        def fallback(t, q, a=first, b=second):
            r1 = runs[(t, a)][q]
            if not r1["abstain"]:
                return r1["correct"], r1["usd"]
            return runs[(t, b)][q]["correct"], r1["usd"] + runs[(t, b)][q]["usd"]
        plans[f"{NAMES[first]}, then {NAMES[second]} if it refuses"] = fallback
def vote(t, q):
    rs = {m: runs[(t, m)][q] for m in M}
    pool = [m for m in M if not rs[m]["abstain"]] or M
    count = Counter(rs[m]["norm"] for m in pool)
    top = max(count.values())
    winner = next(m for m in ["graphrag", "rag", "hipporag", "raptor"] if m in pool and count[rs[m]["norm"]] == top)
    return rs[winner]["correct"], sum(r["usd"] for r in rs.values())
plans["majority vote of 4 (refusals ignored; ties -> Agentic GraphRAG)"] = vote
R["routing"] = [dict(plan=k, **evaluate(f)) for k, f in plans.items()]

# 8. Measurement: scoring-rule sensitivity, judge effect, re-judged recall, hand-label sample.
R["scoring_rules"] = [dict(method=NAMES[m], exact_match=pct(st.mean(r["em"] for r in rows(m))),
                           exact_or_judge=pct(st.mean(r["correct"] for r in rows(m))),
                           any_shared_word=pct(st.mean(r["overlap"] for r in rows(m))),
                           judge_upgrades=sum(r["correct"] and not r["em"] for r in rows(m)),
                           non_exact_non_refusal=sum(not r["em"] and not r["abstain"] for r in rows(m))) for m in M]
R["recall"] = [dict(method=NAMES[m], located_only=pct(summary[NAMES[m]]["retrieval_recall_located_only"]),
                    with_judge=pct(summary[NAMES[m]]["retrieval_recall"])) for m in M]
re_files = sorted(glob.glob(str(OUT / "rejudge" / "verdicts-*.json")))
if re_files:
    for f in re_files:
        for v in json.load(open(f)):
            A["verdicts"][v["id"]] = v  # score_run reads this dict
    rescored = defaultdict(list)
    for x in manifest:
        for r in A["score_run"](x["run_dir"]).values():
            if r["recall"] is not None:
                rescored[x["method"]].append(r["recall"])
    for row in R["recall"]:
        m = next(k for k, v in NAMES.items() if v == row["method"])
        row["with_strict_rejudge"] = pct(st.mean(rescored[m]))
rnd = random.Random(20261106)
items = {}
for f in glob.glob(str(OUT / "judge" / "*" / "items-*.json")) + glob.glob(str(OUT / "rejudge" / "items-*.json")):
    if "batch2_raw" in f:
        continue
    x = json.load(open(f))
    items.update({i["id"]: i for i in x["answers"] + x["evidence"]})
final = {}
for f in sorted(glob.glob(str(OUT / "judge" / "verdicts-*.json"))) + re_files:
    final.update({v["id"]: v for v in json.load(open(f))})
ans = [i for i in final if i in items and items[i]["kind"] == "answer"]
ev = [i for i in final if i in items and items[i]["kind"] == "evidence"]
sample = [dict(**items[i], judge=final[i].get("correct", final[i].get("supported")), human=None)
          for i in rnd.sample(sorted(ans), min(100, len(ans))) + rnd.sample(sorted(ev), min(100, len(ev)))]
(OUT / "hand_label_sample.json").write_text(json.dumps(sample, indent=1))
R["hand_label_sample"] = dict(answers=min(100, len(ans)), evidence=min(100, len(ev)), labeled=0)

json.dump(R, open(OUT / "insights.json", "w"), indent=1, default=str)
for k, v in R.items():
    if k != "hard_questions":
        print(f"\n## {k}\n{json.dumps(v, default=str)[:3000]}")
print("\n## hard_questions", {k: v for k, v in R["hard_questions"].items() if k != "all_fail_examples"})
