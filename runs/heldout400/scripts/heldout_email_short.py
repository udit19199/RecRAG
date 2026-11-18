"""Short email: held-out 400 retrieval, by question type, from runs/heldout400/confusion_by_run.json.

Full per-run confusion matrices stay in the long draft (heldout_email.py) and on slides 11-12.
Writes runs/heldout400/reports/email-heldout-short.md.
"""
import json
from pathlib import Path

d = json.loads(Path("runs/heldout400/confusion_by_run.json").read_text())
CR, CS, CA = d["consistency_retrieval"], d["consistency_support"], d["consistency_answer"]
T3 = [("comparison_query", "Comparison"), ("temporal_query", "Temporal"), ("inference_query", "Inference")]
M = [("graphrag", "Agentic GraphRAG"), ("rag", "RAG"), ("hipporag", "HippoRAG"), ("raptor", "RAPTOR")]
vary = lambda c: c[1] + c[2]

L = ["Subject: RecRAG held-out 400: retrieval by question type, across 3 runs\n", "Hi all,\n",
     "Following my 3 Nov mail: retrieval broken down by question type, 100 questions per type, 3 runs. "
     "Both retrieval checks are now complete for all runs.\n",
     "Questions that pass in all 3 runs (out of 100):\n",
     "| Type | Method | Gold sentences ≥50% | Gold sentences 100% | Answer supported | Answered right |",
     "|---|---|---|---|---|---|"]
for qt, tn in T3:
    for m, mn in M:
        L.append(f"| {tn if m == 'graphrag' else ''} | {mn} | {CR[qt][m]['0.5'][0]} | {CR[qt][m]['1.0'][0]} | "
                 f"{CS[qt][m]['complete'][0]} | {CA[qt][m][0]} |")
ag = lambda C, k: ", ".join(f"{vary(C[qt]['graphrag'][k])} {tn.lower()}" for qt, tn in T3)
L += ["",
      "- The baselines give the same retrieval result in every run. Agentic GraphRAG does not: its result changes "
      f"between runs on {ag(CR, '0.5')} questions (gold sentences ≥50%) and {ag(CS, 'stated')} (answer support).",
      "- In any single run Agentic GraphRAG is close to the baselines; it falls behind on questions it gets right in every run.",
      "- Answer support is judged by gpt-6-luna: does the retrieved text state what is needed to reach the answer, by any route.\n",
      "Full per-run confusion matrices (Agentic GraphRAG vs each baseline, every threshold) are on slides 11 and 12 "
      "of the Experiments scene.\n",
      "Warm Regards,"]
out = Path("runs/heldout400/reports/email-heldout-short.md")
out.write_text("\n".join(L) + "\n")
print(out)
