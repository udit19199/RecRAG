"""Test the reasoning claims on deck slides 6 and 7 against the held-out 400 data, per type and repeat.

Uses Answered right and gold facts found (full_status), plus refusals ("Insufficient information")
and whether a gold article was retrieved (any retrieved passage title equals a gold evidence title).
Run from the repo root: PYTHONPATH=. uv run python runs/heldout400/scripts/check_slide6_7_claims.py
"""
import glob, json
from collections import defaultdict
from pathlib import Path

from evals.qa_metrics import normalize_answer
from evals.subagent_judge import answer_key

H = Path("runs/heldout400")
DS = json.load(open("datasets/multihop_rag/MultiHopRAG.json"))
Q = json.load(open(H / "questions.json"))
qtype = {f"multihop-rag-{i}": t for t, ids in Q["by_type"].items() for i in ids}
gold_titles = {f"multihop-rag-{i}": {e["title"] for e in DS[i]["evidence_list"]} for i in range(len(DS))}
REPS, M = ["r1", "r2", "r3"], ["rag", "raptor", "hipporag", "graphrag"]
ABST = normalize_answer("Insufficient Information")

verdicts = {}
for f in sorted(glob.glob(str(H / "judge" / "verdicts-*.json"))):
    for v in json.load(open(f)):
        verdicts[v["id"]] = v
row = defaultdict(dict)
for m in map(json.loads, open(H / "manifest.jsonl")):
    if m["tag"] not in REPS or not m["run_dir"]:
        continue
    d = Path(m["run_dir"])
    titles = {}
    for r in map(json.loads, open(d / "retrieval.jsonl")):
        t = set()
        for p in r["retrieval"].get("passages", []):
            try:
                head = json.loads(p)["title"]
            except (ValueError, TypeError, KeyError):
                head = p.split("\n", 1)[0]
            if head:
                t.add(head.rsplit(" (", 1)[0])
        titles[r["record_id"]] = t
    for r in map(json.loads, open(d / "results.jsonl")):
        v = verdicts.get(answer_key(r["question"], r["answer"]))
        row[(m["tag"], m["method"])][r["record_id"]] = dict(
            right=bool(r["exact_match"]) or bool(v and v.get("correct")),
            refuse=normalize_answer(r["answer"]) == ABST,
            article_hit=bool(gold_titles[r["record_id"]] & titles.get(r["record_id"], set())),
            all_articles=gold_titles[r["record_id"]] <= titles.get(r["record_id"], set()),
            titles_known=bool(titles.get(r["record_id"])))
for t in REPS:
    for s in range(4):
        for r in map(json.loads, open(H / "judge_support" / f"{t}-s{s}" / "per_question.jsonl")):
            c = [x for g in r["gold"] for x in g["claims"]]
            row[(t, r["method"])][r["record_id"]]["facts"] = sum(x["full_status"] == "stated" for x in c) / len(c) if c else None

for qt in ["comparison_query", "temporal_query"]:
    qs = [q for q in qtype if qtype[q] == qt]
    print(f"\n===== {qt}")
    for m in M:
        for t in REPS:
            R = row[(t, m)]
            none = sum(R[q]["facts"] == 0 for q in qs)
            full = [q for q in qs if R[q]["facts"] == 1]
            some = [q for q in qs if 0 < R[q]["facts"] < 1]
            nof = [q for q in qs if R[q]["facts"] == 0]
            inc = [q for q in qs if R[q]["facts"] < 1]
            ref = [q for q in inc if R[q]["refuse"]]
            ref_all = [q for q in qs if R[q]["refuse"] and R[q]["facts"] == 1]
            ref_noart = [q for q in qs if R[q]["refuse"] and not R[q]["all_articles"]]
            allref = [q for q in qs if R[q]["refuse"]]
            wrong = [q for q in qs if not R[q]["right"]]
            wrong_noart = [q for q in wrong if not R[q]["all_articles"]]
            print(f"{m:9}{t} no facts {none:3} | 100% {len(full):3} right {sum(R[q]['right'] for q in full):3} | "
                  f"some {len(some):3} right {sum(R[q]['right'] for q in some):3} | none right {sum(R[q]['right'] for q in nof):3}/{len(nof):3} | "
                  f"refusals on <100%: {len(ref)}/{len(inc)} | refusals w/ 100% facts {len(ref_all)} | "
                  f"refusals missing a gold article {len(ref_noart)}/{len(allref)} | wrong missing a gold article {len(wrong_noart)}/{len(wrong)} "
                  f"| titles parsed {sum(R[q]['titles_known'] for q in qs)}")
