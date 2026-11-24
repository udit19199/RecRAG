"""Why B4 (top 50 direct) behaves as it does. No model calls.

Questions answered (250 MultiHop-RAG comparison + temporal questions, per type):
1. How many of the 50 chunks hold a gold sentence? (is most of the 50 noise?)
2. What did B2's picking keep: how many gold chunks in its 10, vs the top 10 by score vs all 50?
3. For questions whose top 50 lacks some gold sentence: is the whole article missing, or only
   the right chunk of an article that was retrieved? And do those questions still get answered?

Text match: evals/retrieval_quality (locate_gold, ranked_items, any_overlap), as ceiling.py.
Answered right: exact match, else the Claude Haiku answer verdict (as runs/judge_paper/slide_data.py).

Usage (repo root, light): PYTHONPATH=. .venv/bin/python runs/selection/b4_analysis/analysis.py
  -> runs/selection/b4_analysis/analysis.json
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

from benchmark import SavedRetrieval
from evals.retrieval_quality import any_overlap, load_corpus, locate_gold, ranked_items
from evals.subagent_judge import _Result, answer_key, needs_answer_judge

OUT = Path("runs/selection/b4_analysis")
QS = {f"multihop-rag-{i}" for i in json.load(open("runs/nograph/questions.json"))["indices"]}
reg = json.load(open("runs/selection/run_registry.json"))

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


def saved(label, rep):
    rows = {}
    for d in reg[label][rep]:
        for line in open(loc(d) / "retrieval.jsonl"):
            s = SavedRetrieval.model_validate_json(line)
            rows[s.record_id] = s
    return rows


def results(label, rep):
    rows = {}
    for d in reg[label][rep]:
        for line in open(loc(d) / "results.jsonl"):
            r = json.loads(line)
            rows[r["record_id"]] = right(r)
    return rows


corpus = load_corpus("multihop_rag")
b4, b2 = saved("B4", "1"), saved("B2", "1")
right_b4 = {rep: results("B4", rep) for rep in "123"}
right_b2 = {rep: results("B2", rep) for rep in "123"}


def pages(items):
    return {corpus.page_of(sp.start) for it in items for sp in it.spans}


def covered(golds, items):
    spans = [sp for it in items for sp in it.spans]
    return [any_overlap(g, spans) for g in golds]


stats = defaultdict(lambda: defaultdict(float))
cases = defaultdict(list)
for q, s in b4.items():
    if q not in QS:
        continue
    qt = s.question_type
    o = stats[qt]
    items50 = ranked_items(s, corpus)
    golds = locate_gold(s.supporting_sentences, corpus)
    gold_chunks = [it for it in items50 if any(any_overlap(g, it.spans) for g in golds)]
    o["questions"] += 1
    o["gold_sentences"] += len(golds)
    o["chunks_in_top50"] += len(items50)
    o["gold_chunks_in_top50"] += len(gold_chunks)
    o["gold_chunks_in_top10"] += sum(it.rank < 10 for it in gold_chunks)
    # B2: which 10 did the picking keep? (its saved passages are the picked 10)
    picked = ranked_items(b2[q], corpus)
    pick_gold = [it for it in picked if any(any_overlap(g, it.spans) for g in golds)]
    o["b2_gold_chunks_in_picks"] += len(pick_gold)
    o["b2_picks_total"] += len(picked)
    o["b2_gold_sentences_covered_by_picks"] += sum(covered(golds, picked))
    o["gold_sentences_covered_by_top10"] += sum(covered(golds, items50[:10]))
    o["gold_sentences_covered_by_top50"] += sum(covered(golds, items50))
    top50_idx = {it.text: it.rank for it in items50}
    o["b2_picks_from_top10"] += sum(top50_idx.get(it.text, 99) < 10 for it in picked)
    # incomplete questions: some gold sentence has no overlapping chunk in the top 50
    hit = covered(golds, items50)
    complete = all(hit)
    o["complete_questions" if complete else "incomplete_questions"] += 1
    got_pages = pages(items50)
    if not complete:
        miss = [g for g, h in zip(golds, hit) if not h]
        whole_article_missing = [bool(g.pages) and not (g.pages & got_pages) for g in miss]
        o["incomplete_missing_article_entirely"] += any(whole_article_missing)
        o["incomplete_article_present_chunk_missing"] += not any(whole_article_missing)
        o["missing_gold_sentences"] += len(miss)
        o["missing_gold_in_absent_article"] += sum(whole_article_missing)
        cases[qt].append({"record_id": q, "question": s.question, "gold_sentences": len(golds), "missing": len(miss),
                          "article_absent": sum(whole_article_missing)})
    for rep in "123":
        k = "complete" if complete else "incomplete"
        o[f"b4_right_{k}"] += right_b4[rep][q] / 3
        o[f"b2_right_{k}"] += right_b2[rep][q] / 3

out = {qt: dict(o) for qt, o in stats.items()}
out["examples_incomplete"] = {qt: v[:6] for qt, v in cases.items()}
(OUT / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
for qt, o in stats.items():
    n = o["questions"]
    print(qt, int(n), "questions")
    print(f"  chunks per question in top 50 that hold a gold sentence: {o['gold_chunks_in_top50'] / n:.2f} of 50 (gold sentences per question {o['gold_sentences'] / n:.2f})")
    print(f"  ... of those, in the top 10: {o['gold_chunks_in_top10'] / n:.2f}")
    print(f"  B2 picks: {o['b2_gold_chunks_in_picks'] / n:.2f} gold chunks of {o['b2_picks_total'] / n:.1f} picked; from the score top 10: {o['b2_picks_from_top10'] / n:.2f}")
    print(f"  gold sentences covered: top10 {o['gold_sentences_covered_by_top10']:.0f}, B2 picks {o['b2_gold_sentences_covered_by_picks']:.0f}, top50 {o['gold_sentences_covered_by_top50']:.0f} of {o['gold_sentences']:.0f}")
    print(f"  complete {o['complete_questions']:.0f}, incomplete {o['incomplete_questions']:.0f}"
          f" (whole article missing {o['incomplete_missing_article_entirely']:.0f}, article there but right chunk missing {o['incomplete_article_present_chunk_missing']:.0f})")
    print(f"  answered right (mean of 3 repeats): B4 complete {100 * o['b4_right_complete'] / o['complete_questions']:.0f}% / incomplete {100 * o['b4_right_incomplete'] / max(o['incomplete_questions'], 1):.0f}%;"
          f" B2 complete {100 * o['b2_right_complete'] / o['complete_questions']:.0f}% / incomplete {100 * o['b2_right_incomplete'] / max(o['incomplete_questions'], 1):.0f}%")
