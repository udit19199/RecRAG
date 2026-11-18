"""Held-out study analysis: answer quality, retrieval quality, cost, latency.

Answer: exact match, upgraded to correct where the Haiku judge accepted a
non-exact answer. Retrieval: share of gold sentences located in the evidence
(graph-query rows with source text included), plus gold sentences the Haiku
judge found stated in unlocatable evidence. Cost: list price per question with
no cache discounts, plus the one-time index build spread over the 2,556
MultiHop-RAG questions. Latency: the one-worker latency pass, median and p90.
Uncertainty: paired bootstrap over questions; repeats averaged per question."""
import glob, json, random, statistics as st
from collections import defaultdict
from pathlib import Path

from evals.qa_metrics import normalize_answer
from evals.retrieval_quality import (any_overlap, graphrag_items, load_corpus, load_run,
                                     locate_gold, ranked_items)
from evals.subagent_judge import answer_key, unlocated_evidence

OUT = Path("runs/heldout400")
NAMES = {"rag": "RAG", "raptor": "RAPTOR", "hipporag": "HippoRAG", "graphrag": "Agentic GraphRAG"}
QUESTIONS_IN_DATASET = 2556
G = json.load(open("runs/heldout400/build_calibration/graph_build_calibration.json"))
R = json.load(open("runs/heldout400/build_calibration/raptor_build_calibration.json"))
EMB = 7728 * 290 * 0.02 / 1e6
BUILD = {"graphrag": G["list_usd_full_build"] + EMB, "rag": EMB,
         "hipporag": 3.896 + EMB * 1.5, "raptor": R["list_usd_full_build"] + EMB * 9190 / 7728}

verdicts = {}
# Sorted: "-strict-" files sort after and replace the earlier lenient evidence verdicts.
for f in sorted(glob.glob(str(OUT / "judge" / "verdicts-*.json"))):
    for v in json.load(open(f)):
        verdicts[v["id"]] = v
manifest = [json.loads(l) for l in open(OUT / "manifest.jsonl")]
corpus = load_corpus("multihop_rag")

def score_run(run_dir):
    run_dir = Path(run_dir)
    results = {r["record_id"]: r for r in map(json.loads, open(run_dir / "results.jsonl"))}
    rows = {}
    for saved in load_run(run_dir):
        r = results[saved.record_id]
        golds = locate_gold(saved.supporting_sentences, corpus)
        items = graphrag_items(saved, corpus) if saved.method == "graphrag" else ranked_items(saved, corpus)
        spans = [s for it in items for s in it.spans]
        covered = {i for i, g in enumerate(golds) if any_overlap(g, spans)}
        located = len(covered)
        for ev in unlocated_evidence(saved, corpus, golds):
            v = verdicts.get(ev.id)
            if v:
                covered |= {ev.open_gold[k] for k in v.get("supported", []) if k < len(ev.open_gold)}
        v = verdicts.get(answer_key(r["question"], r["answer"]))
        judged = bool(r["exact_match"]) or bool(v and v.get("correct"))
        rows[saved.record_id] = dict(
            type=r["question_type"], em=r["exact_match"] or 0.0, correct=float(judged),
            recall=len(covered) / len(golds) if golds else None,
            recall_located=located / len(golds) if golds else None,
            usd=r["list_question_cost_usd"], billed=r["estimated_question_cost_usd"],
            sec=r["retrieval_and_answer_seconds"], model_sec=r["model_seconds"],
            ctx=r["retrieved_tokens"], failed=bool(r.get("retrieval_error")),
            failed_calls=len(r["question_cost"]["failed_calls"]),
            abstain=normalize_answer(r["answer"]) == normalize_answer("Insufficient Information"))
    return rows

runs = defaultdict(dict)  # (tag, method) -> record_id -> row
for m in manifest:
    if m["run_dir"]:
        runs[(m["tag"], m["method"])].update(score_run(m["run_dir"]))
reps = sorted({t for t, _ in runs if t != "latency"})

def per_question(method, key):
    out = defaultdict(list)
    for t in reps:
        for q, row in runs[(t, method)].items():
            if row[key] is not None:
                out[q].append(row[key])
    return {q: st.mean(v) for q, v in out.items()}

def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None

summary = {"repeats": reps, "build_usd": BUILD, "methods": [], "paired": [], "by_type": []}
for m in NAMES:
    rows = [row for t in reps for row in runs[(t, m)].values()]
    if not rows:
        continue
    lat = list(runs[("latency", m)].values())
    acc = st.mean(r["correct"] for r in rows)
    usd = st.mean(r["usd"] for r in rows)
    build_q = BUILD[m] / QUESTIONS_IN_DATASET
    rec = [r["recall"] for r in rows if r["recall"] is not None]
    summary["methods"].append(dict(
        method=NAMES[m], answers=len(rows), questions=len({q for t in reps for q in runs[(t, m)]}),
        exact_match=st.mean(r["em"] for r in rows), accuracy=acc,
        accuracy_by_repeat=[st.mean(r["correct"] for r in runs[(t, m)].values()) for t in reps],
        retrieval_recall=st.mean(rec),
        retrieval_recall_located_only=st.mean(r["recall_located"] for r in rows if r["recall_located"] is not None),
        context_tokens=st.mean(r["ctx"] for r in rows),
        query_usd_list=usd, query_usd_billed=st.mean(r["billed"] or 0 for r in rows),
        build_usd=BUILD[m], total_usd_per_question=usd + build_q,
        usd_per_1000_correct=1000 * (usd + build_q) / acc if acc else None,
        latency_median=st.median(r["sec"] for r in lat) if lat else None,
        latency_p90=pct([r["sec"] for r in lat], 0.9),
        latency_model_share=(sum(r["model_sec"] for r in lat) / sum(r["sec"] for r in lat)) if lat else None,
        latency_questions=len(lat),
        abstain_rate=st.mean(r["abstain"] for r in rows),
        failed_questions=sum(r["failed"] for r in rows), failed_calls=sum(r["failed_calls"] for r in rows)))
    for qt in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == qt]
        summary["by_type"].append(dict(method=NAMES[m], type=qt, accuracy=st.mean(r["correct"] for r in sub),
                                       recall=st.mean([r["recall"] for r in sub if r["recall"] is not None] or [0])))

rnd = random.Random(1)
for key in ("correct", "recall"):
    ours = per_question("graphrag", key)
    for b in ("rag", "hipporag", "raptor"):
        other = per_question(b, key)
        qs = sorted(set(ours) & set(other))
        if not qs:
            continue
        d = [ours[q] - other[q] for q in qs]
        boots = sorted(st.mean(rnd.choice(d) for _ in d) for _ in range(5000))
        summary["paired"].append(dict(metric=key, vs=NAMES[b], n=len(qs), diff=st.mean(d),
                                      ci_low=boots[125], ci_high=boots[4874],
                                      wins=sum(x > 0 for x in d), losses=sum(x < 0 for x in d)))
json.dump(summary, open(OUT / "analysis.json", "w"), indent=1)
for row in summary["methods"]:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
for p in summary["paired"]:
    print(f"{p['metric']:8} vs {p['vs']:9} n={p['n']} diff={p['diff']:+.3f} 95%CI[{p['ci_low']:+.3f},{p['ci_high']:+.3f}] wins={p['wins']} losses={p['losses']}")
