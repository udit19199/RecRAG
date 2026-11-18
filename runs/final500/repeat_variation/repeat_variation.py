"""Why "answered right" varied more between the 3 repeats in the previous run than in the latest run.

previous = runs/heldout400 (100 per type), latest = runs/final500 MultiHop-RAG (125 per type).
No model calls. Reads saved answers, prompts, retrieval rows and Claude Haiku answer verdicts only.
No gpt-6-luna verdict is read (judge_support/ is never opened).

Loaders: the loader block of runs/final500/two_run_analysis/compute.py (everything above `def max_parallel`)
is executed as-is, so verdict dicts, right_old/right_new and the run-folder lists are exactly that script's.
compute.py is not imported, because importing it rewrites its own output files.

Per (method, type, run), each flip question (right in some repeats, wrong in others) is put in one cause:
  a  judge:      two repeats with opposite verdicts saw the same prompt and gave the same answer text
  b  answer:     two repeats with opposite verdicts saw the same prompt but gave different answer text
  c  retrieval:  every pair of repeats with opposite verdicts saw different prompts (different evidence)
"Evidence" = the full prompt the answer model received (results.jsonl `prompt`: context + question + instruction).
For Agentic Hybrid RAG the passages in retrieval.jsonl are split into graph rows (score -1.0) and text passages,
and the latest run's picked IDs are read from the select_evidence step of the saved trace.

Run from the repo root:
  PYTHONPATH=. .venv/bin/python runs/final500/repeat_variation/repeat_variation.py
Writes repeat_variation.json next to this file and prints the tables.
"""
import glob
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path("runs/final500/two_run_analysis/compute.py")
src = ROOT.read_text()
ns = {"__file__": str(ROOT.resolve()), "__name__": "compute_loaders"}
exec(compile(src[: src.index("def max_parallel")], str(ROOT), "exec"), ns)  # loaders only, no writes
H, F, M, REPS, TYPES, NAMES = ns["H"], ns["F"], ns["M"], ns["REPS"], ns["TYPES"], ns["NAMES"]
right_old, right_new, answer_key = ns["right_old"], ns["right_new"], ns["answer_key"]
normalize_answer = ns["normalize_answer"]
OUT = Path(__file__).resolve().parent

CASES = [("comparison_query", "graphrag"), ("comparison_query", "raptor"), ("comparison_query", "hipporag"),
         ("comparison_query", "rag"), ("temporal_query", "graphrag")]

# ---- which verdict file decided each answer key (same file order as compute.py)
old_src, old_all = {}, defaultdict(list)
for f in sorted(glob.glob(str(H / "judge" / "verdicts-*.json"))):
    for v in json.load(open(f)):
        if "correct" in v:
            old_src[v["id"]] = Path(f).name
            old_all[v["id"]].append((Path(f).name, v["correct"]))
new_src, new_all = {}, defaultdict(list)
for f in glob.glob(str(F / "judge/answers/verdicts-answers-*.json")):  # compute.py's (unsorted) order
    for v in json.load(open(f)):
        new_src[v["id"]] = Path(f).name
        new_all[v["id"]].append((Path(f).name, v["correct"]))
guide_of = {}
for f in glob.glob(str(H / "judge" / "batch*" / "items-*.json")) + glob.glob(str(F / "judge/answers/items-answers-*.json")):
    d = json.load(open(f))
    tag = "older wording (cb7391d7)" if "Topic overlap is not enough." in d["guide"] else "strict wording (81025957)"
    guide_of[f] = tag


def read_dir(d, judge, srcmap):
    rows = {}
    ret = {}
    for s in map(json.loads, open(Path(d) / "retrieval.jsonl")):
        ret[s["record_id"]] = s["retrieval"]
    for r in map(json.loads, open(Path(d) / "results.jsonl")):
        rt = ret.get(r["record_id"], {})
        ps, sc = rt.get("passages", []), rt.get("scores", [])
        picked = None
        if "trace" in rt:
            sel = [t for t in rt["trace"] if t["tool"] == "select_evidence"]
            picked = tuple(sel[-1]["output"].split()) if sel else ()
        key = answer_key(r["question"], r["answer"])
        rows[r["record_id"]] = dict(
            type=r["question_type"], answer=r["answer"], norm=normalize_answer(r["answer"]), em=r["exact_match"],
            prompt=r["prompt"], right=bool(judge(r)), key=key,
            vsrc="exact match" if r["exact_match"] == 1 else srcmap.get(key, "no verdict (abstain/empty: wrong)"),
            rows=frozenset(p for p, s in zip(ps, sc) if s == -1.0),
            text=frozenset(p for p, s in zip(ps, sc) if s != -1.0),
            picked=picked, has_trace="trace" in rt, dir=d)
    return rows


prev, latest = defaultdict(dict), defaultdict(dict)
for m in map(json.loads, open(H / "manifest.jsonl")):
    if m["run_dir"] and m["tag"] != "latency":
        prev[(m["tag"][1:], m["method"])].update(read_dir(m["run_dir"], right_old, old_src))
registry = json.load(open(F / "run_registry.json"))["multihop_rag"]
for m in M:
    for rep in REPS:
        for d in registry[m][rep]:
            latest[(rep, m)].update(read_dir(d, right_new, new_src))


def classify(runs, qt, meth):
    qids = sorted(q for q, r in runs[("1", meth)].items() if r["type"] == qt)
    acc = [sum(runs[(t, meth)][q]["right"] for q in qids) for t in REPS]
    out = dict(questions=len(qids), right_per_repeat=acc, spread_points=round(100 * (max(acc) - min(acc)) / len(qids), 1),
               flips=0, a_judge=0, b_answer=0, c_retrieval=0, c_detail=Counter(), examples=defaultdict(list),
               flip_ids=[], all_same_prompt=0)
    for q in qids:
        rr = [runs[(t, meth)][q] for t in REPS]
        out["all_same_prompt"] += len({x["prompt"] for x in rr}) == 1
        if len({x["right"] for x in rr}) == 1:
            continue
        out["flips"] += 1
        out["flip_ids"].append(q)
        disc = [(x, y) for x, y in combinations(rr, 2) if x["right"] != y["right"]]
        if any(x["prompt"] == y["prompt"] and x["norm"] == y["norm"] for x, y in disc):
            cause = "a_judge"
        elif any(x["prompt"] == y["prompt"] for x, y in disc):
            cause = "b_answer"
        else:
            cause = "c_retrieval"
            if meth == "graphrag":
                for x, y in disc:
                    d = []
                    d.append("rows differ" if x["rows"] != y["rows"] else "rows same")
                    d.append("text differs" if x["text"] != y["text"] else "text same")
                    if x["picked"] is None:
                        d.append("picked IDs: no trace saved")
                    else:
                        d.append("picked IDs differ" if x["picked"] != y["picked"] else "picked IDs same")
                    d.append("answer differs" if x["norm"] != y["norm"] else "answer same")
                    out["c_detail"][", ".join(d)] += 1
            else:
                for x, y in disc:
                    out["c_detail"]["passages differ" if x["text"] != y["text"] else "passages same, prompt differs"] += 1
        out[cause] += 1
        out["flips_with_abstain_in_a_wrong_repeat"] = out.get("flips_with_abstain_in_a_wrong_repeat", 0) + any(
            not x["right"] and x["norm"] == ns["ABSTENTION"] for x in rr)
        if len(out["examples"][cause]) < 3:
            out["examples"][cause].append(dict(id=q, repeats=[dict(rep=t, right=x["right"], answer=x["answer"][:160],
                                                                    verdict_from=x["vsrc"]) for t, x in zip(REPS, rr)]))
    out["c_detail"] = dict(out["c_detail"])
    out["examples"] = dict(out["examples"])
    return out


result = {"cases": {}, "judge_passes": {}, "baseline_reuse": {}}
for qt, meth in CASES:
    k = f"{qt.replace('_query', '')} | {NAMES[meth]}"
    result["cases"][k] = {"previous": classify(prev, qt, meth), "latest": classify(latest, qt, meth)}
    L = result["cases"][k]["latest"]
    shared = [q for q in L["flip_ids"] if q in prev[("1", meth)]]
    L["flips_on_400_shared"], L["flips_on_100_new"] = len(shared), L["flips"] - len(shared)

# ---- 3. baselines: are the latest answers identical to the previous ones, and do verdicts agree?
for meth in ("rag", "raptor", "hipporag"):
    for qt in ("comparison_query", "temporal_query"):
        same_ans = same_prompt = n = vdiff = 0
        moves = Counter()
        for t in REPS:
            for q, p in prev[(t, meth)].items():
                if p["type"] != qt:
                    continue
                l = latest[(t, meth)][q]
                n += 1
                same_ans += p["answer"] == l["answer"]
                same_prompt += p["prompt"] == l["prompt"]
                if p["right"] != l["right"]:
                    vdiff += 1
                    moves[f"{'right' if p['right'] else 'wrong'}->{'right' if l['right'] else 'wrong'}"] += 1
        changed = [dict(id=q, rep=t, answer=p["answer"][:120], previous=p["right"], previous_from=p["vsrc"],
                        latest=latest[(t, meth)][q]["right"], latest_from=latest[(t, meth)][q]["vsrc"])
                   for t in REPS for q, p in prev[(t, meth)].items()
                   if p["type"] == qt and p["right"] != latest[(t, meth)][q]["right"]]
        shared = [q for q, p in prev[("1", meth)].items() if p["type"] == qt]
        result["baseline_reuse"][f"{qt.replace('_query', '')} | {NAMES[meth]}"] = dict(
            rows=n, identical_answer_text=same_ans, identical_prompt=same_prompt,
            verdict_changed=vdiff, verdict_moves=dict(moves), changed_rows=changed,
            shared100_right_per_repeat_previous_verdicts=[sum(prev[(t, meth)][q]["right"] for q in shared) for t in REPS],
            shared100_right_per_repeat_latest_verdicts=[sum(latest[(t, meth)][q]["right"] for q in shared) for t in REPS],
            new25_right_per_repeat=[sum(r["right"] for q, r in latest[(t, meth)].items() if r["type"] == qt and q not in prev[("1", meth)]) for t in REPS])

# ---- spread is the uneven landing of flips across repeats, not the flip count: measure how likely each spread is
# Null: keep each flip question's number of right repeats (1 or 2 of 3), assign which repeats at random.
import random
rng = random.Random(20261118)
for qt, meth in CASES:
    k = f"{qt.replace('_query', '')} | {NAMES[meth]}"
    for run, runs in (("previous", prev), ("latest", latest)):
        x = result["cases"][k][run]
        per_rep_flip_right = [sum(runs[(t, meth)][q]["right"] for q in x["flip_ids"]) for t in REPS]
        ks = [sum(runs[(t, meth)][q]["right"] for t in REPS) for q in x["flip_ids"]]
        sims = []
        for _ in range(20000):
            c = [0, 0, 0]
            for kk in ks:
                for i in rng.sample(range(3), kk):
                    c[i] += 1
            sims.append(max(c) - min(c))
        obs = max(per_rep_flip_right) - min(per_rep_flip_right)
        x["flips_right_per_repeat"] = per_rep_flip_right
        x["spread_if_flips_land_at_random"] = dict(
            median_questions=sorted(sims)[len(sims) // 2], share_at_or_above_observed=round(sum(s >= obs for s in sims) / len(sims), 3),
            share_at_or_below_observed=round(sum(s <= obs for s in sims) / len(sims), 3))

# per-question view of the shared 400 for the baselines: previous flips that stopped flipping, and new ones
for qt, meth in CASES:
    if meth == "graphrag":
        continue
    k = f"{qt.replace('_query', '')} | {NAMES[meth]}"
    P = set(result["cases"][k]["previous"]["flip_ids"])
    Ls = set(q for q in result["cases"][k]["latest"]["flip_ids"] if q in prev[("1", meth)])
    result["cases"][k]["shared400_flip_overlap"] = dict(previous_only=sorted(P - Ls), latest_only=sorted(Ls - P),
                                                         both=len(P & Ls))

# ---- 4. one judge pass per repeat, or a mix?
for label, runs, all_ in (("previous", prev, old_all), ("latest", latest, new_all)):
    for meth in M:
        for t in REPS:
            c = Counter(r["vsrc"].split("-0")[0].replace("verdicts-", "").replace(".json", "")
                        if r["vsrc"].startswith("verdicts-batch") else
                        ("answers-" + r["vsrc"][-8:-5] if r["vsrc"].startswith("verdicts-answers") else r["vsrc"])
                        for r in runs[(t, meth)].values() if r["type"] in ("comparison_query", "temporal_query"))
            result["judge_passes"][f"{label} | {NAMES[meth]} | r{t}"] = dict(c) if label == "previous" else {
                "exact match": c.pop("exact match", 0), "no verdict": sum(v for k, v in c.items() if k.startswith("no verdict")),
                "judged, files": sorted(k for k in c if k.startswith("answers-"))}
    multi = {k: v for k, v in all_.items() if len(v) > 1}
    result["judge_passes"][f"{label} | keys judged more than once"] = len(multi)
    result["judge_passes"][f"{label} | keys judged more than once with different verdicts"] = sum(len({c for _, c in v}) > 1 for v in multi.values())

# share of a key's rows across repeats: how many answer keys are used by more than one repeat (so one verdict covers them)
for label, runs in (("previous", prev), ("latest", latest)):
    for qt, meth in CASES:
        keys = defaultdict(set)
        for t in REPS:
            for r in runs[(t, meth)].values():
                if r["type"] == qt and r["em"] != 1:
                    keys[r["key"]].add(t)
        result["judge_passes"][f"{label} | {qt.replace('_query', '')} {NAMES[meth]} | judged keys shared by >1 repeat"] = \
            f"{sum(len(v) > 1 for v in keys.values())} of {len(keys)}"

(OUT / "repeat_variation.json").write_text(json.dumps(result, indent=1, default=list))

for k, v in result["cases"].items():
    print(f"\n== {k}")
    for run in ("previous", "latest"):
        x = v[run]
        pt = 100 / x["questions"]
        print(f"  {run:8s} n={x['questions']} right/rep={x['right_per_repeat']} spread={x['spread_points']} "
              f"flips={x['flips']} ({x['flips'] * pt:.1f} pts) a={x['a_judge']} b={x['b_answer']} c={x['c_retrieval']} "
              f"same-prompt-all-3={x['all_same_prompt']} abstain-in-wrong-repeat={x.get('flips_with_abstain_in_a_wrong_repeat', 0)}")
        print(f"           right among flips per repeat {x['flips_right_per_repeat']}; random-landing spread "
              f"{x['spread_if_flips_land_at_random']}")
        if x["c_detail"]:
            print(f"           c detail (discordant pairs): {x['c_detail']}")
        if run == "latest":
            print(f"           flips on shared 400: {x['flips_on_400_shared']}, on 100 new: {x['flips_on_100_new']}")
    if "shared400_flip_overlap" in v:
        o = v["shared400_flip_overlap"]
        print(f"  shared-400 flips: previous only {len(o['previous_only'])}, latest only {len(o['latest_only'])}, both {o['both']}")
print("\n== baseline reuse (shared 400, all 3 repeats)")
for k, v in result["baseline_reuse"].items():
    print(" ", k, {a: b for a, b in v.items() if a != "changed_rows"})
    for c in v["changed_rows"]:
        print("     ", c)
print("\n== judge passes")
for k, v in result["judge_passes"].items():
    print(" ", k, v)
