"""Why does new Agentic GraphRAG vary between repeats? Comparison questions only.

Uses the 10 Nov runs (traces saved), the Haiku fact verdicts, and Normal RAG.
Run from the repo root: PYTHONPATH=. uv run python runs/heldout400/scripts/agentic_variability.py"""
import collections
import json
import statistics as st
from itertools import combinations
from pathlib import Path

from evals.haiku_fact_judge import JudgeTask, JudgeVerdict, _verified
from evals.retrieval_quality import normalize

NEW = {"r1": "runs/agentic_tune/graphrag/comparison100", "r2": "runs/agentic_tune/graphrag/comparison100_rep2",
       "r3": "runs/agentic_tune/graphrag/comparison100_rep3"}
J = Path("runs/agentic_tune/fact_judge")
H = Path("runs/heldout400")


def key(text):
    return normalize(text)[:400]


# --- judge data: per (run, question) which passages hold found facts
tasks = {t["id"]: JudgeTask.model_validate(t) for p in (J / "judge").glob("items-*.json") for t in json.loads(p.read_text())["items"]}
verdicts = {v["id"]: JudgeVerdict.model_validate(v) for p in (J / "judge").glob("verdict-*.json") for v in json.loads(p.read_text())}
links = [json.loads(line) for line in (J / "judge" / "links.jsonl").read_text().splitlines()]
share = {(l["run"], l["record_id"]): None for l in links}
for line in open(J / "per_question.jsonl"):
    r = json.loads(line)
    share[(r["run"], r["record_id"])] = r["share"]


def found_items(run, rid):
    """Passages (normalized key) whose verified quote supports a found fact, per fact label."""
    link = next(l for l in links if l["run"] == run and l["record_id"] == rid)
    task, verdict = tasks[link["task"]], verdicts[link["task"]]
    out = {}
    for f in verdict.facts:
        if f.status != "stated":
            continue
        ok = [q for q in f.evidence if _verified(q, task.items)]
        if ok:
            out[f.fact] = {key(task.items[q.item - 1]) for q in ok}
    return out, {l for g in task.gold for l in g["facts"]}


rag_dir = next(json.loads(l)["run_dir"] for l in open(H / "manifest.jsonl") if json.loads(l)["method"] == "rag" and json.loads(l)["tag"] == "r1")
rag_runs = [str(H / json.loads(l)["run_dir"]) if not json.loads(l)["run_dir"].startswith("/") else json.loads(l)["run_dir"]
            for l in open(H / "manifest.jsonl") if json.loads(l)["method"] == "rag" and json.loads(l)["tag"] == "r1" and json.loads(l)["exit"] == 0]
rag = {}
for d in rag_runs:
    for line in open(Path(d) / "retrieval.jsonl"):
        s = json.loads(line)
        if s["question_type"] == "comparison_query":
            rag[s["record_id"]] = (d, [key(p) for p in s["retrieval"]["passages"]])

new = {}
for t, d in NEW.items():
    for line in open(Path(d) / "retrieval.jsonl"):
        s = json.loads(line)
        r = s["retrieval"]
        # which search step returned each kept passage
        step_of = {}
        for step in r["trace"]:
            if step["tool"] != "vector_search":
                continue
            for row in step["output"].split("\n"):
                try:
                    ev = json.loads(row)
                except ValueError:
                    continue
                if isinstance(ev, dict) and "text" in ev:
                    step_of.setdefault(key(ev["text"]), step["step"])
        vs = [st_ for st_ in r["trace"] if st_["tool"] == "vector_search"]
        cy = [st_ for st_ in r["trace"] if st_["tool"] == "cypher_search"]
        kept = [key(p) for p in r["passages"]]
        new[(t, s["record_id"])] = dict(
            run=d, kept=kept, step_of=step_of,
            first_vs=vs[0]["step"] if vs else None,
            n_vs=len(vs), n_cy=len(cy),
            queries=[st_["args"].get("query_text", "") for st_ in vs],
            cy_status=[("error" if c["status"] == "error" else "empty" if c["results"] == 0 else "huge" if len(c["output"]) > 100_000 else "rows") for c in cy],
            rows_kept=sum(1 for p in r["passages"] if p.lstrip().startswith(("{", "["))),
        )
qs = sorted({rid for (_, rid) in new})

print("=== 1. How much each method changes between repeats (comparison, 100 questions)")
def ok(run, rid, th):
    return share[(run, rid)] is not None and share[(run, rid)] >= th - 1e-9
for th, label in [(0.5, ">=50%"), (0.75, ">=75%"), (1.0, "100%")]:
    flips = sum(len({ok(NEW[t], q, th) for t in NEW}) > 1 for q in qs)
    print(f"  {label}: questions whose found/not-found result differs across the 3 new repeats: {flips}")

print("\n=== 2. Does the agent search the same way each repeat?")
same_first = sum(len({new[(t, q)]["queries"][0] if new[(t, q)]["queries"] else "" for t in NEW}) == 1 for q in qs)
print(f"  first search text identical in all 3 repeats: {same_first} of 100")
jacc = []
for q in qs:
    for a, b in combinations(NEW, 2):
        A, B = set(new[(a, q)]["kept"]), set(new[(b, q)]["kept"])
        jacc.append(len(A & B) / max(1, len(A | B)))
print(f"  kept passages shared between two repeats (Jaccard): median {st.median(jacc):.2f}, mean {st.mean(jacc):.2f}")
same10 = sum(len({tuple(sorted(new[(t, q)]['kept'])) for t in NEW}) == 1 for q in qs)
print(f"  questions with the identical 10 kept passages in all 3 repeats: {same10}")
for t in NEW:
    rows = [new[(t, q)] for q in qs]
    c = collections.Counter(x for r in rows for x in r["cy_status"])
    print(f"  {t}: vector searches per question {st.mean(r['n_vs'] for r in rows):.2f} (dist {sorted(collections.Counter(r['n_vs'] for r in rows).items())}); cypher {dict(c)}; graph rows kept {sum(r['rows_kept'] for r in rows)}")

print("\n=== 3. Where kept passages come from")
for t in NEW:
    first = later = rows = unknown = 0
    for q in qs:
        r = new[(t, q)]
        for k in r["kept"]:
            s = r["step_of"].get(k)
            if s is None:
                rows += 1
            elif s == r["first_vs"]:
                first += 1
            else:
                later += 1
    print(f"  {t}: kept from first search {first}, from later searches {later}, graph rows/other {rows}")
ov = [len(set(new[(t, q)]["kept"]) & set(rag[q][1])) for t in NEW for q in qs]
print(f"  passages shared with Normal RAG's 10: mean {st.mean(ov):.1f}")

print("\n=== 4. Where the found facts come from (new runs)")
for t in NEW:
    src = collections.Counter()
    for q in qs:
        r = new[(t, q)]
        f, _ = found_items(r["run"], q)
        rag_set = set(rag[q][1])
        for label, items in f.items():
            steps = {r["step_of"].get(k) for k in items}
            in_rag = any(k in rag_set for k in items)
            if r["first_vs"] in steps:
                src["first search"] += 1
            elif steps - {None}:
                src["later search only"] += 1
            else:
                src["graph row only"] += 1
            src["also in Normal RAG's 10" if in_rag else "not in Normal RAG's 10"] += 1
    print(f"  {t}: {dict(src)}")

print("\n=== 5. Questions that flip at >=50% between repeats: why")
reasons = collections.Counter()
examples = []
for q in qs:
    res = {t: ok(NEW[t], q, 0.5) for t in NEW}
    if len(set(res.values())) == 1:
        continue
    good = [t for t in NEW if res[t]]
    bad = [t for t in NEW if not res[t]]
    for g in good:
        for b in bad:
            fg, labels = found_items(NEW[g], q)
            fb, _ = found_items(NEW[b], q)
            lost = [l for l in fg if l not in fb]
            kept_b = set(new[(b, q)]["kept"])
            for l in lost:
                if any(k in kept_b for k in fg[l]):
                    reasons["passage was kept in the worse repeat too (judge did not credit it)"] += 1
                else:
                    gs = {new[(g, q)]["step_of"].get(k) for k in fg[l]}
                    via_later = new[(g, q)]["first_vs"] not in gs
                    reasons["passage not kept in the worse repeat" + (" (better repeat got it from a later search)" if via_later else " (better repeat got it from its first search)")] += 1
            if len(examples) < 4:
                examples.append((q, g, b, new[(g, q)]["queries"][:3], new[(b, q)]["queries"][:3], new[(g, q)]["cy_status"], new[(b, q)]["cy_status"]))
for k, v in reasons.most_common():
    print(f"  {v:4d}  {k}")
print("\n  examples (question, better repeat, worse repeat):")
for q, g, b, qg, qb, cg, cb in examples:
    print(f"  {q}: {g} vs {b}")
    print(f"     {g} searches: {[x[:90] for x in qg]} cypher {cg}")
    print(f"     {b} searches: {[x[:90] for x in qb]} cypher {cb}")

print("\n=== 6. Same question, repeats with 1 search vs 2+ searches")
better = worse = same = 0
for q in qs:
    one = [t for t in NEW if new[(t, q)]["n_vs"] == 1]
    more = [t for t in NEW if new[(t, q)]["n_vs"] >= 2]
    for a in one:
        for b in more:
            sa, sb = share[(NEW[a], q)], share[(NEW[b], q)]
            if sa is None or sb is None:
                continue
            better += sb > sa; worse += sb < sa; same += sb == sa
print(f"  pairs where the 2+ search repeat found more facts {better}, fewer {worse}, same {same}")
print("  share of questions with a 2nd search:", {t: sum(new[(t, q)]['n_vs'] >= 2 for q in qs) for t in NEW})

print("\n=== 7. Answer-model randomness alone (Normal RAG: identical passages every repeat)")
import glob
from evals.subagent_judge import answer_key
verd = {}
for f in sorted(glob.glob(str(H / "judge/verdicts-*.json"))) + sorted(glob.glob("runs/agentic_tune/judge/verdicts-*.json")):
    for v in json.load(open(f)):
        verd[v["id"]] = v
def right_runs(method, tags):
    out = {}
    for line in open(H / "manifest.jsonl"):
        m = json.loads(line)
        if m["method"] == method and m["tag"] in tags and m["exit"] == 0 and m["run_dir"]:
            d = m["run_dir"]; d = d if d.startswith("/") else str(H / d)
            for l in open(Path(d) / "results.jsonl"):
                r = json.loads(l)
                if r["question_type"] == "comparison_query":
                    out[(m["tag"], r["record_id"])] = bool(r["exact_match"]) or bool(verd.get(answer_key(r["question"], r["answer"]), {}).get("correct"))
    return out
rr = right_runs("rag", ("r1", "r2", "r3"))
print("  Normal RAG questions whose Answered right differs across repeats:", sum(len({rr[(t, q)] for t in NEW}) > 1 for q in qs))
nr = {}
for t, d in NEW.items():
    for l in open(Path(d) / "results.jsonl"):
        r = json.loads(l)
        nr[(t, r["record_id"])] = bool(r["exact_match"]) or bool(verd.get(answer_key(r["question"], r["answer"]), {}).get("correct"))
print("  New Agentic questions whose Answered right differs across repeats:", sum(len({nr[(t, q)] for t in NEW}) > 1 for q in qs))
