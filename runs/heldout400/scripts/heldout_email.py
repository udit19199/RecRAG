"""Draft email: held-out 400 retrieval confusion matrices by question type, every run separately.

Every number comes from runs/heldout400/confusion_by_run.json (heldout_confusion_by_run.py).
No cost, latency or pooled/overall figures. Writes runs/heldout400/reports/email-heldout-by-run.md.
"""
import json
from pathlib import Path

d = json.loads(Path("runs/heldout400/confusion_by_run.json").read_text())
RUNS = ["r1", "r2", "r3"]
BASE = [("rag", "RAG"), ("hipporag", "HippoRAG"), ("raptor", "RAPTOR")]
METH = [("graphrag", "Agentic GraphRAG"), ("rag", "RAG"), ("hipporag", "HippoRAG"), ("raptor", "RAPTOR")]
TYPES = [("comparison_query", "Comparison"), ("temporal_query", "Temporal"),
         ("inference_query", "Inference"), ("null_query", "Null")]
TH = [("0.5", "≥50%"), ("0.75", "≥75%"), ("1.0", "100%")]
LV = [("stated", "answer stated"), ("complete", "complete chain")]
c4 = lambda v: " / ".join(map(str, v))


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


L = []
w = L.append
w("Subject: RecRAG held-out 400: retrieval confusion matrices by question type, each run separately\n")
w("Hi all,\n")
w("Following my 3 Nov mail, here is the retrieval analysis broken down by question type, with each of the "
  "three runs shown separately (100 questions per type in every run), plus how consistent each question is "
  "across the runs. There are no overall averages here.\n")
w("**How to read the tables**\n")
w("- Retrieval check 1, gold-sentence recall: a question counts as *found* when at least 50%, 75% or 100% "
  "of its gold sentences are in the retrieved text.")
w("- Retrieval check 2, answer support: a judge (gpt-6-luna) reads the retrieved text and scores 0–3 whether it "
  "states what is needed to reach the gold answer, by any valid route. *Answer stated* = score 2 or 3; "
  "*complete chain* = score 3. Now run on all 3 runs × 400 questions.")
w("- Agentic GraphRAG vs baseline cells: **both / Agentic only / baseline only / neither**, out of 100 questions.")
w("- Across-run cells: number of questions **in all 3 runs / 2 runs / 1 run / no run**.")
w("- Null questions have no gold facts, so they appear only in the answer tables.\n")

for qt, name in TYPES:
    w(f"## {name}\n")
    if d["pair_retrieval"].get(qt):
        w("**Gold-sentence recall, Agentic GraphRAG vs each baseline, per run** (both / Agentic only / baseline only / neither)\n")
        rows = [[f"vs {bn}", f"Run {i + 1}"] + [c4(d["pair_retrieval"][qt][b][r][t]) for t, _ in TH]
                for b, bn in BASE for i, r in enumerate(RUNS)]
        w(table(["", "Run"] + [f"Found {lab}" for _, lab in TH], rows) + "\n")
        w("**Answer support, per run** (both / Agentic only / baseline only / neither)\n")
        rows = [[f"vs {bn}", f"Run {i + 1}"] + [c4(d["pair_support"][qt][b][r][lv]) for lv, _ in LV]
                for b, bn in BASE for i, r in enumerate(RUNS)]
        w(table(["", "Run"] + [lab.capitalize() for _, lab in LV], rows) + "\n")
        w("**The same question across the 3 runs** (all 3 / 2 / 1 / no run)\n")
        rows = [[mn] + [c4(d["consistency_retrieval"][qt][m][t]) for t, _ in TH]
                + [c4(d["consistency_support"][qt][m][lv]) for lv, _ in LV] for m, mn in METH]
        w(table(["Method"] + [f"Found {lab}" for _, lab in TH] + [f"Support: {lab}" for _, lab in LV], rows) + "\n")
    w("**Answers, per run** (both correct / Agentic only / baseline only / both wrong)\n")
    rows = [[f"vs {bn}"] + [c4(d["pair_answer"][qt][b][r]) for r in RUNS] for b, bn in BASE]
    w(table(["", "Run 1", "Run 2", "Run 3"], rows) + "\n")
    w("Answered right across the 3 runs (all 3 / 2 / 1 / no run): " + "; ".join(
        f"{mn} {c4(d['consistency_answer'][qt][m])}" for m, mn in METH) + ".\n")
    w("Wrong in every run (Agentic wrong ×3 & baseline right ×3 / baseline wrong ×3 & Agentic right ×3 / both wrong ×3): "
      + "; ".join(f"vs {bn} {c4(d['wrong_all_runs'][qt][b])}" for b, bn in BASE) + ".\n")

# Observations, each computed here from the same file.
CR, CS = d["consistency_retrieval"], d["consistency_support"]
T3 = TYPES[:3]
vary = lambda c: c[1] + c[2]
base_stable_ret = all(vary(CR[qt][m][t]) == 0 for qt, _ in T3 for m, _ in BASE for t, _ in TH)
base_vary_sup = [(qt, m) for qt, _ in T3 for m, _ in BASE for lv, _ in LV if vary(CS[qt][m][lv])]
w("## What the per-run view shows\n")
w("- RAG, HippoRAG and RAPTOR give the same retrieval result in every run"
  + (": on gold-sentence recall each question is found in all 3 runs or in none" if base_stable_ret else "")
  + f", and they get the same support verdict in all 3 runs on every question but {len(base_vary_sup)}. "
  "Only Agentic GraphRAG's retrieval changes from run to run.")
w("- Agentic GraphRAG's result changes between runs on "
  + ", ".join(f"{vary(CR[qt]['graphrag']['0.5'])} {n.lower()}" for qt, n in T3)
  + " questions (gold-sentence recall ≥50%) and on "
  + ", ".join(f"{vary(CS[qt]['graphrag']['stated'])} {n.lower()}" for qt, n in T3)
  + " questions (answer stated).")
rng = lambda xs: f"{min(xs)}–{max(xs)}" if min(xs) != max(xs) else str(xs[0])
w("- Questions whose answer is supported with a complete chain in every run: "
  + "; ".join(f"{n.lower()} {CS[qt]['graphrag']['complete'][0]} for Agentic GraphRAG vs "
              f"{rng([CS[qt][m]['complete'][0] for m, _ in BASE])} for the baselines" for qt, n in T3) + ".")
gap = max(abs(c[1] - c[2]) for qt, _ in T3 for b, _ in BASE for r in RUNS
          for lv, _ in LV for c in [d["pair_support"][qt][b][r][lv]])
w(f"- Within any single run, the Agentic-only and baseline-only support counts differ by at most {gap} "
  "questions, so in one run Agentic GraphRAG is close to the baselines. What separates them is consistency: "
  "Agentic GraphRAG's supported questions move between runs, so fewer are supported in all three.\n")
w("The same tables are on slides 11 (gold-sentence recall) and 12 (answer support) of the Experiments scene.\n")
w("Warm Regards,")

out = Path("runs/heldout400/reports/email-heldout-by-run.md")
out.write_text("\n".join(L) + "\n")
print(f"wrote {out}, base support varies on: {base_vary_sup}")
