"""Spec for deck slides 3-5, version 2: one tall slide per question type, one number per cell.

Slide 3 comparison, slide 4 temporal, slide 5 inference + null. Each slide: dataset facts, then
"All 4 methods" (one table per measure: rows methods, columns repeats), then "Agentic GraphRAG vs
Normal RAG" (one table per measure: rows repeats, columns both / Agentic only / Normal RAG only /
neither), a fails-in-all-3 table with the undercount caveat, and found-in-3/2/1/0-repeats tables.
Numbers come from runs/heldout400/gold_facts_by_run.json (heldout_gold_facts.py).
Run from the repo root: python3 runs/heldout400/scripts/deck_slides_3_5_v2_spec.py
"""
import json
from pathlib import Path

D = json.load(open("runs/heldout400/gold_facts_by_run.json"))
REPS = D["repeats"]
MEAS = ["answered_right", "0.5", "0.75", "1.0"]
LAB = {"answered_right": "Answered right", "0.5": "Gold facts found ≥50%",
       "0.75": "Gold facts found ≥75%", "1.0": "Gold facts found 100%"}
METHODS = [("rag", "Normal RAG"), ("raptor", "RAPTOR"), ("hipporag", "HippoRAG"), ("graphrag", "Agentic GraphRAG")]
W, GAP = 2872, 80
HALF = (W - GAP) / 2
HEAD, BODY, LABEL = ("#e9ecef", "#adb5bd"), ("#FFFFFF", "#ced4da"), ("#f8f9fa", "#ced4da")
AGC, RAGC, FAIL = ("#e5dbff", "#ced4da"), ("#dbe4ff", "#ced4da"), ("#fff3bf", "#ced4da")
FR = {3: "7fNjDMQmsNw4M-_kyqhx3", 4: "_LGLLt7ziA8Nwhqwofiam", 5: "X0d0Z-rbTxg2KzkX889Sz"}
FX = {3: 10050, 4: 13400, 5: 16750}
ROW, CAP = 72, 58

spec = {"delete_tag": "2026-11-09", "frames": {}, "titles": {}, "moves": [], "add": []}


def cell(f, x, y, w, h, text, col, fs=36):
    spec["add"].append({"kind": "cell", "frame": f, "x": x, "y": y, "w": w, "h": h,
                        "bg": col[0], "stroke": col[1], "text": str(text), "fs": fs})


def text(f, x, y, s, fs=36, maxw=W):
    spec["add"].append({"kind": "text", "frame": f, "x": x, "y": y, "text": s, "fs": fs, "maxw": maxw})


def table(f, x, y, caption, corner, cols, rows, widths, head_h=ROW, head_fs=32, row_col=None):
    """rows: list of (label, [values], colors). Returns bottom y."""
    text(f, x, y, caption, 40, sum(widths))
    y += CAP
    cx = x
    for i, h in enumerate([corner] + cols):
        cell(f, cx, y, widths[i], head_h, h, HEAD, head_fs)
        cx += widths[i]
    y += head_h
    for label, vals, col in rows:
        cx = x
        cell(f, cx, y, widths[0], ROW, label, col or LABEL, 34)
        cx += widths[0]
        for i, v in enumerate(vals):
            cell(f, cx, y, widths[i + 1], ROW, v, BODY, 40)
            cx += widths[i + 1]
        y += ROW
    return y


def type_slide(n, qt, takeaway):
    f, x0 = FR[n], FX[n] + 64
    xr = x0 + HALF + GAP
    y = 6000
    text(f, x0, y, "All 4 methods: questions out of 100, each repeat separately", 50)
    y += 90
    w_m = [520, (HALF - 520) / 3, (HALF - 520) / 3, (HALF - 520) / 3]
    reps_cols = [f"Repeat {t[1:]}" for t in REPS]
    for r, pair in enumerate([MEAS[:2], MEAS[2:]]):
        bottoms = []
        for c, ms in enumerate(pair):
            rows = [(name, [D["per_method"][qt][m][t][ms] for t in REPS],
                     AGC if m == "graphrag" else RAGC if m == "rag" else None) for m, name in METHODS]
            bottoms.append(table(f, x0 if c == 0 else xr, y, LAB[ms], "Method", reps_cols, rows, w_m))
        y = max(bottoms) + 50
    y += 30
    text(f, x0, y, "Agentic GraphRAG vs Normal RAG: questions out of 100, each repeat separately", 50)
    y += 90
    w_p = [256] + [(HALF - 256) / 4] * 4
    pcols = ["Both", "Agentic GraphRAG only", "Normal RAG only", "Neither"]
    for pair in (MEAS[:2], MEAS[2:]):
        bottoms = []
        for c, ms in enumerate(pair):
            rows = [(f"Repeat {t[1:]}", D["pair"][qt]["rag"][t][ms], None) for t in REPS]
            bottoms.append(table(f, x0 if c == 0 else xr, y, LAB[ms], "Repeat", pcols, rows, w_p, head_h=100, head_fs=30))
        y = max(bottoms) + 50
    # fails in all 3 + caveat beside it
    w_f = [520] + [(HALF - 520) / 3] * 3
    rows = [(LAB[ms], D["fails_all3"][qt]["rag"][ms], None) for ms in MEAS]
    yb = table(f, x0, y, "Fails in all 3 repeats", "Measure", ["Both", "Agentic GraphRAG only", "Normal RAG only"],
               rows, w_f, head_h=100, head_fs=30)
    text(f, xr, y + CAP + 10, "Agentic GraphRAG only: Agentic GraphRAG fails in all 3 repeats while Normal RAG "
         "succeeds in all 3. Normal RAG only: the reverse. "
         "This table undercounts Agentic GraphRAG, which varies between repeats. Normal RAG "
         "retrieves the same text every repeat, so each question is found in 3 repeats or in 0. Read it together "
         "with the two tables below: they show in how many of the 3 repeats each method gets a question.", 38, HALF)
    y = yb + 50
    w_r = [520] + [(HALF - 520) / 4] * 4
    bottoms = []
    for c, (m, name, col) in enumerate((("graphrag", "Agentic GraphRAG", AGC), ("rag", "Normal RAG", RAGC))):
        rows = [(LAB[ms], D["repeats_found"][qt][m][ms], col) for ms in MEAS]
        bottoms.append(table(f, x0 if c == 0 else xr, y, f"{name}: questions right or found in N of the 3 repeats",
                             "Measure", ["3 repeats", "2 repeats", "1 repeat", "0 repeats"], rows, w_r))
    y = max(bottoms) + 60
    text(f, x0, y, takeaway, 50)
    return y


fails100 = lambda qt: D["fails_all3"][qt]["rag"]["1.0"][0]
ar = [D["per_method"]["inference_query"][m][t]["answered_right"] for m, _ in METHODS for t in REPS]

spec["titles"] = {FR[3]: "Comparison questions", FR[4]: "Temporal questions", FR[5]: "Inference and null questions"}
spec["frames"] = {FR[3]: {"name": "03 Comparison questions"}, FR[4]: {"name": "04 Temporal questions"},
                  FR[5]: {"name": "05 Inference and null questions"}}
# dataset-fact blocks (y ranges as they are in the scene now)
spec["moves"] = [
    {"from": FR[4], "ymin": 5690, "ymax": 5960, "to": FR[3], "dx": -3350, "target_y": 5700},        # comparison facts
    {"from": FR[4], "ymin": 6740, "ymax": 7020, "to": FR[4], "dx": 0, "target_y": 5700},         # temporal facts -> 5700
    {"from": FR[3], "ymin": 7200, "ymax": 7440, "to": FR[5], "dx": 6700, "target_y": None},      # null facts
]

type_slide(3, "comparison_query",
           "The two methods are level. Gold facts found are within 5 questions of each other at every threshold in "
           "every repeat, and Answered right has no consistent winner: Agentic GraphRAG is ahead in repeat 1, behind in "
           f"repeat 2 and level in repeat 3. {fails100('comparison_query')} questions never reach 100% of gold "
           "facts for either method in any repeat.")
type_slide(4, "temporal_query",
           "Agentic GraphRAG leads on Answered right and on gold facts found ≥50% and ≥75% in every repeat. At "
           "100% the two are level in repeats 1 and 3 (15 each), and Agentic GraphRAG is ahead in repeat 2 "
           f"(19 vs 15). {fails100('temporal_query')} questions never reach 100% of gold facts for "
           "either method in any repeat.")
y = type_slide(5, "inference_query",
               f"Every method answers {min(ar)} to {max(ar)} of 100 inference questions right in every repeat, so the "
               "method hardly matters here. Gold facts found are level between Agentic GraphRAG and Normal RAG, "
               "within 4 questions of each other at every threshold in every repeat, and "
               f"{fails100('inference_query')} questions never reach 100% of gold facts for either method in any repeat.")
y += 4 * 63 + 80          # leave room for the takeaway (up to 4 lines)
spec["moves"][2]["target_y"] = y  # null facts block (two boxes, 220 tall) moves here
spec["null_y"] = y
f, x0 = FR[5], FX[5] + 64
y += 220 + 40
w_m = [520, (HALF - 520) / 3, (HALF - 520) / 3, (HALF - 520) / 3]
rows = [(name, [D["per_method"]["null_query"][m][t]["answered_right"] for t in REPS],
         AGC if m == "graphrag" else RAGC if m == "rag" else None) for m, name in METHODS]
yb = table(f, x0, y, "Null: Answered right, out of 100", "Method", [f"Repeat {t[1:]}" for t in REPS], rows, w_m)
text(f, x0 + HALF + GAP, y + CAP + 10, "Null questions have no gold facts, so they have no retrieval numbers.", 38, HALF)

Path("runs/heldout400/deck_slides_3_5_v2_spec.json").write_text(json.dumps(spec, ensure_ascii=False))
print(len(spec["add"]), "items; null_y", spec["null_y"])
