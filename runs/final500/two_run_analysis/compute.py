"""Four poster sections for both runs, per MultiHop-RAG question type only (no pooled totals).

previous = runs/heldout400 (100 questions per type), latest = runs/final500 MultiHop-RAG (125 per type).
Sections: 1 significance, 3b null and yes/no bias, 5b latency, 5d total cost (+ query cost per question).
No model calls; reads saved answers and verdicts only. No gpt-6-luna verdict is read.

Answered right
  previous: exact match, else the Claude Haiku answer verdict in runs/heldout400/judge (as heldout_analysis.py,
            so it reproduces insights.json).
  latest:   exact match, else the Claude Haiku answer verdict in runs/final500/judge/answers (as report.py).
Query cost: list_question_cost_usd of each answer (results.jsonl); re-derived here from the per-call token counts
  in question_cost.calls and the pricing in each run folder's settings.json, and the script stops on any mismatch.
Latency: retrieval_and_answer_seconds of each answer (results.jsonl, wall clock per question inside the run).
Build cost: the latest run reused the previous MultiHop-RAG indexes, so both use heldout_analysis.py's BUILD.

Run from the repo root:
  PYTHONPATH=. .venv/bin/python runs/final500/two_run_analysis/compute.py
Writes two_run_analysis.json and SUMMARY.md next to this file.
"""
import glob
import json
import math
import statistics as st
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from evals.qa_metrics import normalize_answer
from evals.subagent_judge import ABSTENTION, _Result, answer_key, needs_answer_judge

H = Path("runs/heldout400")
F = Path("runs/final500")
OUT = Path(__file__).resolve().parent
M = ["rag", "raptor", "hipporag", "graphrag"]
NAMES = {"rag": "Normal RAG", "raptor": "RAPTOR", "hipporag": "HippoRAG", "graphrag": "Agentic Hybrid RAG"}
REPS = ["1", "2", "3"]
TYPES = ["comparison_query", "temporal_query", "inference_query", "null_query"]
TN = {t: t.replace("_query", "") for t in TYPES}
pct = lambda x: None if x is None else round(100 * x, 1)
mean = lambda xs: st.mean(xs) if xs else None

old_verdicts = {}
for f in sorted(glob.glob(str(H / "judge" / "verdicts-*.json"))):  # "-strict-" sorts last, as in heldout_analysis
    for v in json.load(open(f)):
        old_verdicts[v["id"]] = v
new_verdicts = {v["id"]: v["correct"] for p in glob.glob(str(F / "judge/answers/verdicts-answers-*.json")) for v in json.load(open(p))}


def right_old(r):
    v = old_verdicts.get(answer_key(r["question"], r["answer"]))
    return bool(r["exact_match"]) or bool(v and v.get("correct"))


def right_new(r):
    if r.get("exact_match") == 1:
        return True
    res = _Result.model_validate({k: r[k] for k in ("record_id", "question", "answer", "expected_answers", "exact_match")})
    if not needs_answer_judge(res):
        return False
    key = answer_key(r["question"], r["answer"])
    if key not in new_verdicts:
        raise SystemExit(f"no answer verdict for {r['record_id']}")
    return new_verdicts[key]


cost_checked = defaultdict(int)
folder_spans = defaultdict(list)  # (run label, method) -> [(start, end)] per run folder


def list_usd(r, pricing):
    """List price from per-call token counts: every input token at the full input rate (no cache discount)."""
    usd = 0.0
    for c in r["question_cost"]["calls"]:
        if c["kind"] == "embeddings":
            usd += c["input_tokens"] * pricing["embedding_per_million"] / 1e6
        else:
            usd += (c["input_tokens"] * pricing["input_per_million"] + c["output_tokens"] * pricing["output_per_million"]) / 1e6
    return usd


def load(run_dir, label, method, judge):
    pricing = json.load(open(Path(run_dir) / "settings.json"))["pricing"]
    rows, starts, ends = {}, [], []
    for r in map(json.loads, open(Path(run_dir) / "results.jsonl")):
        assert abs(list_usd(r, pricing) - r["list_question_cost_usd"]) < 1e-9, (run_dir, r["record_id"])
        cost_checked[label] += 1
        calls = r["question_cost"]["calls"]
        rows[r["record_id"]] = dict(
            type=r["question_type"], gold=r["expected_answers"], norm=normalize_answer(r["answer"]),
            correct=float(judge(r)), abstain=normalize_answer(r["answer"]) == ABSTENTION,
            usd=r["list_question_cost_usd"], sec=r["retrieval_and_answer_seconds"], model_sec=r["model_seconds"],
            input_tokens=sum(c["input_tokens"] for c in calls if c["kind"] != "embeddings"),
            output_tokens=sum(c["output_tokens"] for c in calls if c["kind"] != "embeddings"),
            calls=sum(c["kind"] == "responses" for c in calls), concurrency=r.get("concurrency"),
            failed_calls=len(r["question_cost"]["failed_calls"]))
        t0 = datetime.fromisoformat(r["started_at"].replace("Z", "+00:00")).timestamp()
        starts.append(t0)
        ends.append(t0 + r["retrieval_and_answer_seconds"])
    if starts:
        folder_spans[(label, method)].append((min(starts), max(ends)))
    return rows


prev, latest = defaultdict(dict), defaultdict(dict)
prev_latency = defaultdict(list)  # (type, method) -> seconds, the previous one-worker latency pass
for m in map(json.loads, open(H / "manifest.jsonl")):
    if not m["run_dir"]:
        continue
    if m["tag"] == "latency":
        for r in map(json.loads, open(Path(m["run_dir"]) / "results.jsonl")):
            prev_latency[(r["question_type"], m["method"])].append(r["retrieval_and_answer_seconds"])
    else:
        prev[(m["tag"][1:], m["method"])].update(load(m["run_dir"], "previous", m["method"], right_old))
registry = json.load(open(F / "run_registry.json"))["multihop_rag"]
for m in M:
    for rep in REPS:
        for d in registry[m][rep]:
            latest[(rep, m)].update(load(d, "latest", m, right_new))


def max_parallel(spans):
    ev = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans])
    cur = best = 0
    for _, x in ev:
        cur += x
        best = max(best, cur)
    return best


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


G = json.load(open(H / "build_calibration/graph_build_calibration.json"))
RC = json.load(open(H / "build_calibration/raptor_build_calibration.json"))
EMB = 7728 * 290 * 0.02 / 1e6
BUILD = {"rag": EMB, "raptor": RC["list_usd_full_build"] + EMB * 9190 / 7728,
         "hipporag": 3.896 + EMB * 1.5, "graphrag": G["list_usd_full_build"] + EMB}
BUILD_PARTS = {"rag": "embeddings only: 7,728 chunks x 290 tokens x $0.02/M (estimate)",
               "raptor": "build_calibration/raptor_build_calibration.json (1,462 summary calls, token counts measured) + embeddings of 9,190 nodes (estimate)",
               "hipporag": "$3.896 hard-coded in heldout_analysis.py (no calibration file or metered log found for it) + 1.5x chunk embeddings (estimate)",
               "graphrag": "build_calibration/graph_build_calibration.json (100 sampled chunks metered, scaled to 7,728) + chunk embeddings (estimate)"}


def analyse(runs, qt, n_type):
    qids = sorted(q_ for q_, r in runs[("1", "rag")].items() if r["type"] == qt)
    assert len(qids) == n_type and all(set(qids) <= set(runs[(t, m)]) for t in REPS for m in M), qt
    rows = lambda m: [runs[(t, m)][q_] for t in REPS for q_ in qids]
    mean_q = lambda m, q_: st.mean(runs[(t, m)][q_]["correct"] for t in REPS)
    maj = {m: {q_: mean_q(m, q_) >= 0.5 for q_ in qids} for m in M}
    S = {"questions": len(qids)}
    S["1_significance"] = []
    for other in ("rag", "raptor", "hipporag"):
        b = sum(maj["graphrag"][q_] and not maj[other][q_] for q_ in qids)
        c = sum(maj[other][q_] and not maj["graphrag"][q_] for q_ in qids)
        d = [mean_q("graphrag", q_) - mean_q(other, q_) for q_ in qids]
        S["1_significance"].append(dict(vs=NAMES[other], questions=len(qids),
            agentic_right_2of3=sum(maj["graphrag"].values()), other_right_2of3=sum(maj[other].values()),
            agentic_only_right=b, other_only_right=c, p=round(mcnemar(b, c), 6),
            wins=sum(x > 0 for x in d), ties=sum(x == 0 for x in d), losses=sum(x < 0 for x in d)))
    S["3b_null_yesno"] = []
    for m in M:
        rs = rows(m)
        yn = [r for r in rs if normalize_answer(r["gold"][0]) in ("yes", "no")]
        first = defaultdict(int)
        for r in yn:
            first[r["norm"].split()[0] if r["norm"] else ""] += 1
        d = dict(method=NAMES[m], answers=len(rs))
        if qt == "null_query":
            d.update(null_abstain_rate=pct(mean([r["abstain"] for r in rs])), null_answered_anyway=sum(not r["abstain"] for r in rs))
        else:
            d.update(false_insufficient_rate=pct(mean([r["abstain"] for r in rs])), false_insufficient=sum(r["abstain"] for r in rs))
        if yn:
            d.update(yesno_answers=len(yn), gold_yes_share=pct(mean([normalize_answer(r["gold"][0]) == "yes" for r in yn])),
                     said_yes=pct(first["yes"] / len(yn)), said_no=pct(first["no"] / len(yn)), refused=pct(mean([r["abstain"] for r in yn])),
                     accuracy_gold_yes=pct(mean([r["correct"] for r in yn if normalize_answer(r["gold"][0]) == "yes"])),
                     accuracy_gold_no=pct(mean([r["correct"] for r in yn if normalize_answer(r["gold"][0]) == "no"])))
        S["3b_null_yesno"].append(d)
    S["5b_latency"] = [dict(method=NAMES[m], answers=len(rows(m)), median_s=round(st.median(r["sec"] for r in rows(m)), 2),
                            p90_s=round(q([r["sec"] for r in rows(m)], 0.9), 2), mean_s=round(mean([r["sec"] for r in rows(m)]), 2),
                            model_share_of_time=pct(sum(r["model_sec"] for r in rows(m)) / sum(r["sec"] for r in rows(m))),
                            threads_per_process=sorted({r["concurrency"] for r in rows(m)} - {None})) for m in M]
    S["5d_cost"] = []
    for m in M:
        rs = rows(m)
        u = mean([r["usd"] for r in rs])
        S["5d_cost"].append(dict(method=NAMES[m], answers=len(rs), query_usd_per_question=round(u, 6),
            query_usd_per_1000_questions=round(1000 * u, 3), accuracy=pct(mean([r["correct"] for r in rs])),
            mean_input_tokens=round(mean([r["input_tokens"] for r in rs])), mean_output_tokens=round(mean([r["output_tokens"] for r in rs])),
            mean_model_calls=round(mean([r["calls"] for r in rs]), 2), failed_calls=sum(r["failed_calls"] for r in rs),
            build_usd_one_time=round(BUILD[m], 3), build_plus_this_types_questions_once=round(BUILD[m] + n_type * u, 3),
            build_plus_this_types_questions_all_repeats=round(BUILD[m] + 3 * n_type * u, 3),
            build_plus_10000_questions_of_this_type=round(BUILD[m] + 10000 * u, 2)))
    return S


P = {qt: analyse(prev, qt, 100) for qt in TYPES}
L = {qt: analyse(latest, qt, 125) for qt in TYPES}
prev400 = {k: {q_: v[q_] for q_ in prev[("1", "rag")]} for k, v in latest.items()}
L400 = {qt: analyse(prev400, qt, 100) for qt in TYPES}
for S in (P, L, L400):  # Holm across the 12 per-type tests of each run (no pooled test)
    tests = [t for qt in TYPES for t in S[qt]["1_significance"]]
    run_max = 0.0
    for rank, i in enumerate(sorted(range(len(tests)), key=lambda i: tests[i]["p"])):
        run_max = max(run_max, min(1.0, (len(tests) - rank) * tests[i]["p"]))
        tests[i]["p_holm_12"] = round(run_max, 6)
for qt in TYPES:
    P[qt]["5b_latency_pass_1worker"] = [dict(method=NAMES[m], questions=len(v), median_s=round(st.median(v), 2),
                                             p90_s=round(q(v, 0.9), 2)) for m in M for v in [prev_latency[(qt, m)]]]
parallel = {f"{lab} {NAMES[m]}": dict(run_folders=len(folder_spans[(lab, m)]), max_folders_running_at_once=max_parallel(folder_spans[(lab, m)]))
            for lab in ("previous", "latest") for m in M}

SRC = {
    "previous": "runs/heldout400/runs/<method>/<timestamp>/results.jsonl, the r1-r3 folders listed in runs/heldout400/manifest.jsonl (r2 Agentic is the r2resume folders plus the 20261101T231404 resume folders). Latency pass: the 4 'latency' folders in the same manifest (40 questions per method, 10 per type, 1 worker).",
    "latest": "the folders in runs/final500/run_registry.json['multihop_rag']: Normal RAG, RAPTOR and HippoRAG are the previous run's r1-r3 folders for the 400 plus 2 new folders per repeat (runs/<method>/20261113T0941*-0942*) for the 100 new questions; Agentic Hybrid RAG is 6 new folders per repeat (runs/graphrag/20261113T0948*-0949*). runs/final500/multihop_rag/<method>/r* hold no extra timing data. No one-worker latency pass exists for the latest run (searched runs/final500 jobs, logs, scripts, STATUS/DECISIONS).",
    "fields": "Latency = retrieval_and_answer_seconds (wall clock from the start of retrieval to the end of the answer, per question); model_seconds = time inside API calls. Tokens = question_cost.calls[*].input_tokens / output_tokens (API usage per call; output includes reasoning). Query cost = list_question_cost_usd, re-derived from those tokens at settings.json pricing (gpt-6-luna $0.10/M input, $0.50/M output, no cached-input discount; text-embedding-3-small $0.02/M); identical pricing block in both runs.",
}
result = {
    "definition": "Previous = held-out 400 (runs/heldout400, 100 per type); latest = final500 MultiHop-RAG (runs/final500, 125 per type). Per question type only; no pooled MultiHop-RAG numbers. 'latest_on_previous_400' = latest run restricted to the previous run's questions.",
    "notes": ("Latest Normal RAG, RAPTOR and HippoRAG reuse the previous run's answers (same run folders) on the 400 and answered 25 new questions per type; "
              "only Agentic Hybrid RAG was re-run (new version) on all 500. Previous answered-right uses the previous Haiku answer verdicts, latest the fresh Haiku answer judge. "
              "judge/excluded_records.json lists only 2 Natural Questions ids, so no MultiHop-RAG question is left out."),
    "data_sources": SRC,
    "cost_rows_rechecked_from_tokens": dict(cost_checked),
    "sections": {
        "1_significance": {
            "definition": "Per question, right = right in >=2 of 3 repeats. Exact two-sided McNemar on Agentic-only vs baseline-only right counts. p_holm_12 = Holm over the 12 per-type tests of each run (3 baselines x 4 types). Wins/ties/losses compare the mean over 3 repeats per question.",
            "notes": "The poster's Holm p used 15 tests (incl. a pooled 'all' row); with the pooled row dropped Holm is over 12, so previous Holm p differ from the poster (e.g. temporal vs Normal RAG 0.492 -> 0.422). Raw p, wins/ties/losses and discordant counts reproduce insights.json exactly. Agentic Hybrid RAG is a different version in each run; the baselines are the same answers on the 400.",
            "previous": {qt: P[qt]["1_significance"] for qt in TYPES}, "latest": {qt: L[qt]["1_significance"] for qt in TYPES},
            "latest_on_previous_400": {qt: L400[qt]["1_significance"] for qt in TYPES}},
        "3b_null_yesno": {
            "definition": "Answers = question x repeat. Null type: share of answers that are 'Insufficient Information'. Answerable types: share that abstain (false 'insufficient'). Yes/no questions (gold 'yes'/'no', comparison and temporal only): first word of the answer, refusals, accuracy by gold label.",
            "notes": "Inference and null questions have no yes/no golds.",
            "previous": {qt: P[qt]["3b_null_yesno"] for qt in TYPES}, "latest": {qt: L[qt]["3b_null_yesno"] for qt in TYPES},
            "latest_on_previous_400": {qt: L400[qt]["3b_null_yesno"] for qt in TYPES}},
        "5b_latency": {
            "definition": "retrieval_and_answer_seconds per answer over all 3 repeats of the main runs: median, p90, mean; model_share_of_time = sum(model_seconds)/sum(seconds). Previous also: its one-worker latency pass (what the poster showed), 10 questions per type.",
            "notes": "Main-run timings are taken under load. threads_per_process is each folder's 'concurrency' field; parallel_processes gives how many run folders of that method ran at the same time (from started_at + seconds). The latest run has no one-worker pass, so the only like-for-like latency comparison is main-run vs main-run, and load differs (see parallel_processes).",
            "parallel_processes": parallel, "sources": SRC,
            "previous": {qt: P[qt]["5b_latency"] for qt in TYPES}, "latest": {qt: L[qt]["5b_latency"] for qt in TYPES},
            "latest_on_previous_400": {qt: L400[qt]["5b_latency"] for qt in TYPES},
            "previous_latency_pass_1worker": {qt: P[qt]["5b_latency_pass_1worker"] for qt in TYPES}},
        "5d_total_cost": {
            "definition": "Query cost per question = mean list_question_cost_usd over the type's answers (3 repeats). Total = one-time build (whole corpus, list USD) + N x query cost per question, N = the type's questions once (100 / 125), all 3 repeats (300 / 375), or 10,000.",
            "notes": "The build is one index for the whole MultiHop-RAG corpus and cannot be split by type; it is the same figure in both runs because the latest run reused the previous indexes (SPEC 'multihop_indexes: no rebuild'; only one index folder per method exists under runs/<method>/indexes; the latest Agentic run folders' build_cost.json are $0). Build figures were not metered during the build; their sources are in build_sources.",
            "build_usd": {NAMES[m]: round(BUILD[m], 4) for m in M}, "build_sources": {NAMES[m]: BUILD_PARTS[m] for m in M}, "sources": SRC,
            "previous": {qt: P[qt]["5d_cost"] for qt in TYPES}, "latest": {qt: L[qt]["5d_cost"] for qt in TYPES},
            "latest_on_previous_400": {qt: L400[qt]["5d_cost"] for qt in TYPES}},
    },
}
(OUT / "two_run_analysis.json").write_text(json.dumps(result, indent=1, default=str))

# ---------------------------------------------------------------- SUMMARY.md
out = []
w = out.append
pl = lambda a, b: f"{'–' if a is None else a} / {'–' if b is None else b}"


def tbl(head, rows):
    w("| " + " | ".join(head) + " |"); w("|" + "---|" * len(head))
    for r in rows:
        w("| " + " | ".join("–" if x is None else str(x) for x in r) + " |")
    w("")


def per_type(key, head, fn):
    tbl(["type"] + head, [[TN[qt]] + fn(a, b, qt) for qt in TYPES for a, b in zip(P[qt][key], L[qt][key])])


w("# Two-run analysis: previous (held-out 400) vs latest (final500, MultiHop-RAG)\n")
w("Generated by compute.py. Per question type only (previous 100, latest 125 questions per type; 3 repeats); no pooled numbers. "
  "Cells are `previous / latest`. Answered right = exact match, else Haiku answer judge. "
  "Latest Normal RAG, RAPTOR and HippoRAG reuse the previous answers on the 400 and add 25 new questions per type; Agentic Hybrid RAG is a new version on all 500. "
  "two_run_analysis.json also has each section for the latest run on the previous 400 questions only.\n")
w("## 1. Is the gap real? Agentic Hybrid RAG vs each baseline (right = right in 2 of 3 repeats)\n")
per_type("1_significance", ["vs", "Agentic right / other right", "wins/ties/losses", "Agentic-only vs other-only", "exact McNemar p", "Holm p (12 tests)"],
    lambda a, b, qt: [a["vs"], pl(f"{a['agentic_right_2of3']}/{a['other_right_2of3']}", f"{b['agentic_right_2of3']}/{b['other_right_2of3']}"),
                      pl(f"{a['wins']}/{a['ties']}/{a['losses']}", f"{b['wins']}/{b['ties']}/{b['losses']}"),
                      pl(f"{a['agentic_only_right']}-{a['other_only_right']}", f"{b['agentic_only_right']}-{b['other_only_right']}"),
                      pl(a["p"], b["p"]), pl(a["p_holm_12"], b["p_holm_12"])])
w("Previous raw p and counts match insights.json; its Holm p differ from the poster because the pooled row is gone (12 tests, not 15).\n")
w("## 3b. Null questions and yes/no bias (% of answers = question x repeat)\n")
tbl(["method", "null: abstain (right)", "comparison: false 'insufficient'", "temporal: false 'insufficient'", "inference: false 'insufficient'"],
    [[NAMES[m]] + [pl(P["null_query"]["3b_null_yesno"][i]["null_abstain_rate"], L["null_query"]["3b_null_yesno"][i]["null_abstain_rate"])]
     + [pl(P[qt]["3b_null_yesno"][i]["false_insufficient_rate"], L[qt]["3b_null_yesno"][i]["false_insufficient_rate"]) for qt in TYPES[:3]] for i, m in enumerate(M)])
rows = []
for qt in ("comparison_query", "temporal_query"):
    for a, b in zip(P[qt]["3b_null_yesno"], L[qt]["3b_null_yesno"]):
        rows.append([TN[qt], a["method"], pl(a["yesno_answers"], b["yesno_answers"]), pl(a["said_yes"], b["said_yes"]), pl(a["said_no"], b["said_no"]),
                     pl(a["refused"], b["refused"]), pl(a["accuracy_gold_yes"], b["accuracy_gold_yes"]), pl(a["accuracy_gold_no"], b["accuracy_gold_no"])])
w("Yes/no questions (comparison and temporal only; inference and null have none):\n")
tbl(["type", "method", "answers", "said yes", "said no", "refused", "acc. gold yes", "acc. gold no"], rows)
w("Gold 'yes' share: comparison " + pl(P["comparison_query"]["3b_null_yesno"][0]["gold_yes_share"], L["comparison_query"]["3b_null_yesno"][0]["gold_yes_share"])
  + ", temporal " + pl(P["temporal_query"]["3b_null_yesno"][0]["gold_yes_share"], L["temporal_query"]["3b_null_yesno"][0]["gold_yes_share"]) + ".\n")
w("## 5b. Latency per question (seconds)\n")
w("Main runs, all 3 repeats: `retrieval_and_answer_seconds` from each run folder's results.jsonl. In brackets: the previous run's one-worker latency pass (10 questions per type), which the poster showed. The latest run has no such pass.\n")


def lat(a, b, qt):
    lp = {x["method"]: x for x in P[qt]["5b_latency_pass_1worker"]}[a["method"]]
    return [a["method"], f"{a['median_s']} [{lp['median_s']}] / {b['median_s']}", f"{a['p90_s']} [{lp['p90_s']}] / {b['p90_s']}",
            pl(a["model_share_of_time"], b["model_share_of_time"])]


per_type("5b_latency", ["method", "median", "p90", "% of time in API calls"], lat)
w("Load during the main runs (most run folders of the method running at once x threads per folder):\n")
tbl(["method", "previous", "latest"], [[NAMES[m], f"{parallel['previous ' + NAMES[m]]['max_folders_running_at_once']} x {P['comparison_query']['5b_latency'][i]['threads_per_process']}",
                                        f"{parallel['latest ' + NAMES[m]]['max_folders_running_at_once']} x {L['comparison_query']['5b_latency'][i]['threads_per_process']}"] for i, m in enumerate(M)])
w("## 5d. Cost (USD, list price)\n")
w("Query cost per question = mean `list_question_cost_usd` (re-derived from per-call token counts at $0.10/M input, $0.50/M output, $0.02/M embeddings; no cache discount). "
  "Build is one-time for the whole corpus, the same indexes in both runs: " + ", ".join(f"{NAMES[m]} ${BUILD[m]:.3f}" for m in M) + ". The build figures are estimates from calibration files (HippoRAG's $3.896 has no saved source).\n")
per_type("5d_cost", ["method", "query USD / question", "mean input / output tokens", "model calls", "build + this type once (100/125 q)", "build + 10,000 q"],
    lambda a, b, qt: [a["method"], pl(a["query_usd_per_question"], b["query_usd_per_question"]),
                      pl(f"{a['mean_input_tokens']}/{a['mean_output_tokens']}", f"{b['mean_input_tokens']}/{b['mean_output_tokens']}"),
                      pl(a["mean_model_calls"], b["mean_model_calls"]), pl(a["build_plus_this_types_questions_once"], b["build_plus_this_types_questions_once"]),
                      pl(a["build_plus_10000_questions_of_this_type"], b["build_plus_10000_questions_of_this_type"])])
(OUT / "SUMMARY.md").write_text("\n".join(out))
print("wrote", OUT / "two_run_analysis.json", "and SUMMARY.md; cost rows rechecked:", dict(cost_checked))
