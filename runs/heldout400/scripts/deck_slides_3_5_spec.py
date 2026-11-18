"""Build the element spec for rebuilding deck slides 3-5 (and the run-map fixes on slide 1).

Every number comes from runs/heldout400/gold_facts_by_run.json (heldout_gold_facts.py).
Writes runs/heldout400/deck_slides_3_5_spec.json, which a browser script applies to the
Excalidraw scene 5KJVp4wI32J. Run from the repo root: python3 runs/heldout400/scripts/deck_slides_3_5_spec.py
"""
import json
from pathlib import Path

D = json.load(open("runs/heldout400/gold_facts_by_run.json"))
REPS = D["repeats"]
MEAS = ["answered_right", "0.5", "0.75", "1.0"]
MEAS_LABEL = ["Answered right", "Gold facts found ≥50%", "Gold facts found ≥75%", "Gold facts found 100%"]
METHODS = [("rag", "Normal RAG"), ("raptor", "RAPTOR"), ("hipporag", "HippoRAG"), ("graphrag", "Agentic GraphRAG")]
W = 2872
HEAD = ("#e9ecef", "#adb5bd")
BODY = ("#FFFFFF", "#ced4da")
LABEL = ("#f8f9fa", "#ced4da")
AGC = ("#e5dbff", "#ced4da")
RAGC = ("#dbe4ff", "#ced4da")
FAIL = ("#fff3bf", "#ced4da")

FR3, FR4, FR5, FR1 = "7fNjDMQmsNw4M-_kyqhx3", "_LGLLt7ziA8Nwhqwofiam", "X0d0Z-rbTxg2KzkX889Sz", "qWfPUNxVx8sJFI1rgXoJJ"
spec = {"frames": {}, "keep": {}, "move": [], "edit_text": [], "add": []}


def cell(frame, x, y, w, h, text, colors, fs=36):
    spec["add"].append({"kind": "cell", "frame": frame, "x": x, "y": y, "w": w, "h": h,
                        "bg": colors[0], "stroke": colors[1], "text": text, "fs": fs})


def text(frame, x, y, s, fs=36):
    spec["add"].append({"kind": "text", "frame": frame, "x": x, "y": y, "text": s, "fs": fs})


def slash(v):
    return " / ".join(str(n) for n in v)


def pair_block(frame, x0, y, qt):
    """Agentic GraphRAG vs Normal RAG: per-repeat cells, fails in all 3, repeats found. Returns new y."""
    text(frame, x0, y, "Agentic GraphRAG vs Normal RAG. Each cell: both / Agentic GraphRAG only / "
         "Normal RAG only / neither, out of 100 questions", 32)
    y += 56
    lw, cw = 940, (W - 940) / 4
    cell(frame, x0, y, lw, 72, "", HEAD)
    for i, lab in enumerate(MEAS_LABEL):
        cell(frame, x0 + lw + i * cw, y, cw, 72, lab, HEAD, 34)
    y += 72
    for t in REPS:
        cell(frame, x0, y, lw, 72, f"Repeat {t[1:]}", LABEL)
        for i, ms in enumerate(MEAS):
            cell(frame, x0 + lw + i * cw, y, cw, 72, slash(D["pair"][qt]["rag"][t][ms]), BODY, 40)
        y += 72
    cell(frame, x0, y, lw, 100, "Fails in all 3 repeats\n(both / Agentic GraphRAG only / Normal RAG only)", FAIL, 30)
    for i, ms in enumerate(MEAS):
        cell(frame, x0 + lw + i * cw, y, cw, 100, slash(D["fails_all3"][qt]["rag"][ms]), FAIL, 40)
    y += 100
    for m, name, col in (("graphrag", "Agentic GraphRAG", AGC), ("rag", "Normal RAG", RAGC)):
        cell(frame, x0, y, lw, 100, f"{name}: questions right or found\nin 3 / 2 / 1 / 0 repeats", col, 30)
        for i, ms in enumerate(MEAS):
            cell(frame, x0 + lw + i * cw, y, cw, 100, slash(D["repeats_found"][qt][m][ms]), col, 40)
        y += 100
    text(frame, x0, y + 14, "Fails in all 3 repeats undercounts Agentic GraphRAG, which varies between repeats. "
         "Normal RAG retrieves the same\ntext every repeat, so each question is found in 3 repeats or in 0. "
         "Read the fails row together with the 3 / 2 / 1 / 0 rows.", 30)
    return y + 14 + 90


# ---------- slide 3: all 4 methods, per type, per repeat ----------
x0 = 10050 + 64
spec["keep"][FR3] = {"title_y_max": 5600, "blocks": [[7590, 7820, None]]}
text(FR3, x0, 5690, "Questions out of 100, repeats 1 / 2 / 3. Answered right, then gold facts found at ≥50% / ≥75% / 100% "
     "of a question's gold facts.\nEach type's dataset facts, and Agentic GraphRAG vs Normal RAG, are on slides 4 and 5.", 34)
y = 5810
TYPE_NAME = {"comparison_query": "Comparison", "temporal_query": "Temporal", "inference_query": "Inference"}
lw, cw = 620, (W - 620) / 4
for qt in ["comparison_query", "temporal_query", "inference_query"]:
    text(FR3, x0, y, f"{TYPE_NAME[qt]}: 100 questions per repeat", 40)
    y += 64
    cell(FR3, x0, y, lw, 72, "Method", HEAD, 34)
    for i, lab in enumerate(MEAS_LABEL):
        cell(FR3, x0 + lw + i * cw, y, cw, 72, lab, HEAD, 34)
    y += 72
    for m, name in METHODS:
        col = AGC if m == "graphrag" else (RAGC if m == "rag" else LABEL)
        cell(FR3, x0, y, lw, 72, name, col)
        for i, ms in enumerate(MEAS):
            cell(FR3, x0 + lw + i * cw, y, cw, 72, slash(D["per_method"][qt][m][t][ms] for t in REPS), BODY, 40)
        y += 72
    y += 44
# null block: existing facts text moves here, then one Answered right row
spec["keep"][FR3]["blocks"][0][2] = y - 7600  # null facts (two boxes, 7600-7820) move here
spec["edit_text"].append({"frame": FR3, "text_prefix": "Null: 100 per run", "strip_suffix": " HotpotQA has none"})
y += 220 + 20
cell(FR3, x0, y, lw, 72, "Null questions", HEAD, 34)
for i, (_, name) in enumerate(METHODS):
    cell(FR3, x0 + lw + i * cw, y, cw, 72, name, HEAD, 34)
y += 72
cell(FR3, x0, y, lw, 100, "Answered right,\nrepeats 1 / 2 / 3", LABEL, 32)
for i, (m, _) in enumerate(METHODS):
    cell(FR3, x0 + lw + i * cw, y, cw, 100, slash(D["per_method"]["null_query"][m][t]["answered_right"] for t in REPS), BODY, 40)
y += 100
text(FR3, x0, y + 14, "Null questions have no gold facts, so they have no retrieval numbers.", 30)
spec["frames"][FR3] = {"height": round(y + 14 + 60 + 70 - 5450)}

# ---------- slide 4: comparison and temporal ----------
x0 = 13400 + 64
spec["keep"][FR4] = {"title_y_max": 5600, "blocks": [[5690, 5960, 0], [6580, 6850, None]]}
y = pair_block(FR4, x0, 5990, "comparison_query")
dy = y + 40 - 6610
spec["keep"][FR4]["blocks"][1][2] = dy
y = pair_block(FR4, x0, 6890 + dy, "temporal_query")
take4 = ("Comparison: the two methods are level. Gold facts found are within a few questions at every threshold\n"
         "in every repeat, and Answered right has no consistent winner: Agentic GraphRAG is ahead in repeat 1,\n"
         "behind in repeat 2 and level in repeat 3. Temporal: Agentic GraphRAG leads on Answered right and on\n"
         "gold facts found ≥50% and ≥75% in every repeat; at 100% the two are level. For either method, "
         f"{D['fails_all3']['comparison_query']['rag']['1.0'][0]} comparison\nand {D['fails_all3']['temporal_query']['rag']['1.0'][0]} "
         "temporal questions never reach 100% of gold facts in any repeat.")
text(FR4, x0, y + 30, take4, 50)
spec["frames"][FR4] = {"height": round(y + 30 + 5 * 63 + 80 - 5450)}

# ---------- slide 5: inference only ----------
x0 = 16750 + 64
spec["keep"][FR5] = {"title_y_max": 5600, "blocks": [[5690, 5960, 0]]}
spec["frames"][FR5] = {"name": "05 Inference questions"}
spec["edit_text"].append({"frame": FR5, "text_prefix": "Inference questions and HotpotQA", "new": "Inference questions"})
spec["edit_text"].append({"frame": FR5, "text_prefix": "Held-out 400 inference: 100 per run",
                          "replace": ["Held-out 400 inference: ", "Inference: "]})
y = pair_block(FR5, x0, 5990, "inference_query")
ar = [D["per_method"]["inference_query"][m][t]["answered_right"] for m, _ in METHODS for t in REPS]
take5 = (f"Every method answers {min(ar)} to {max(ar)} of 100 inference questions right in every repeat, so the method\n"
         "hardly matters here. Gold facts found are level between Agentic GraphRAG and Normal RAG, within a few\n"
         f"questions at every threshold in every repeat, and {D['fails_all3']['inference_query']['rag']['1.0'][0]} "
         "questions never reach 100% of gold facts for either method in any repeat.")
text(FR5, x0, y + 30, take5, 50)
spec["frames"][FR5]["height"] = round(y + 30 + 3 * 63 + 80 - 5450)

# ---------- slide 1: run map fixes ----------
spec["edit_text"] += [
    {"frame": FR1, "text_prefix": "Exact match, then Claude Haiku judge",
     "new": "Answered right: exact\nmatch, then Claude\nHaiku judge. Gold\nfacts found:\ngpt-6-luna judge"},
    {"frame": FR1, "text_prefix": "Exact match and contains answer. gpt-6-luna",
     "new": "Exact match and\ncontains answer, plus\nolder evidence\nmeasures. Not on\nthe result slides"},
    {"frame": FR1, "exact": "4, 5", "new": "Not shown"},
    {"frame": FR1, "exact": "5", "new": "Not shown"},
]

Path("runs/heldout400/deck_slides_3_5_spec.json").write_text(json.dumps(spec, ensure_ascii=False))
print(len(spec["add"]), "new elements;", {k: v for k, v in spec["frames"].items()})
