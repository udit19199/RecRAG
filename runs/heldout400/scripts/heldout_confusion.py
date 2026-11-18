"""Confusion matrices for the final held-out run (400 questions x 3 repeats).

Uses the same scoring as heldout_analysis (Haiku-judged answers and evidence).
Per method and question type:
* evidence vs answer: an answer counts as "evidence found" when the share of
  gold sentences (facts) retrieved reaches the threshold (50%, 75%, 100%),
  crossed with the answer being correct. Null questions have no gold facts and
  are left out.
* exact match vs judge: exact match crossed with the Haiku-judged answer.
Importing heldout_analysis re-runs it and rewrites analysis.json with the same
values. Writes runs/heldout400/confusion.json."""
import json
from collections import Counter

import heldout_analysis as h

THRESHOLDS = (0.5, 0.75, 1.0)
rows = {m: [r for t in h.reps for r in h.runs[(t, m)].values()] for m in h.NAMES}
types = sorted({r["type"] for rs in rows.values() for r in rs})
out = {"repeats": h.reps, "thresholds": THRESHOLDS, "evidence_vs_answer": [], "exact_vs_judge": []}
for m, name in h.NAMES.items():
    for qt in types:
        sub = [r for r in rows[m] if r["type"] == qt]
        c = Counter((bool(r["em"]), bool(r["correct"])) for r in sub)
        out["exact_vs_judge"].append(dict(method=name, type=qt, answers=len(sub),
            em_right_judge_right=c[(True, True)], em_right_judge_wrong=c[(True, False)],
            em_wrong_judge_right=c[(False, True)], em_wrong_judge_wrong=c[(False, False)]))
        ev = [r for r in sub if r["recall"] is not None]
        for th in THRESHOLDS if ev else ():
            c = Counter((r["recall"] >= th - 1e-9, bool(r["correct"])) for r in ev)
            out["evidence_vs_answer"].append(dict(method=name, type=qt, threshold=th, answers=len(ev),
                found_right=c[(True, True)], found_wrong=c[(True, False)],
                missed_right=c[(False, True)], missed_wrong=c[(False, False)]))
json.dump(out, open(h.OUT / "confusion.json", "w"), indent=1)
