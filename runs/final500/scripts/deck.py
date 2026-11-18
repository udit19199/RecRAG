"""Build the Experiments-scene additions for the final 500-question study.

Usage (repo root): python3 runs/final500/scripts/deck.py BACKUP.excalidraw
Reads runs/final500/report.json (and the 38-question check), and writes
runs/final500/deck_ops.json: new elements (tagged customData.recragDeck =
"2026-11-13-final500") and new frame heights. Nothing old is removed or
changed except frame heights, which grow to fit the new section at the bottom
of each slide. Styles copy the existing tables: grey header row, white cells,
green/red matrix columns, Excalifont, 44 px cell text.
"""

import json
import random
import secrets
import sys
import textwrap
import time
from pathlib import Path

F = Path("runs/final500")
backup = json.loads(Path(sys.argv[1]).read_text())
frames = {e["id"]: e for e in backup["elements"] if e["type"] == "frame"}
report = json.loads((F / "report.json").read_text())
TAG = {"recragDeck": "2026-11-13-final500"}
rng = random.Random(20261113)
NOW = int(time.time() * 1000)
elements: list[dict] = []
heights: dict[str, float] = {}

FRAME = {
    "map": "qWfPUNxVx8sJFI1rgXoJJ",
    "methods": "9fu1vowfab2t4oldptecF",
    "changes": "g6q4hxbfmwbrwzlpeq1",
    "comparison": "mf54m550mwbrm8logx3",
    "temporal": "7f11rr27mwbrm8mbmsc",
    "infnull": "X0d0Z-rbTxg2KzkX889Sz",
    "nq": "69f3c7efmwbqah9isqz",
    "hotpot": "zr6kwlddmwbr4jlki7z",
    "backup": "9E77qFFB20jsaawsDwI80",
}
NAMES = {"rag": "Normal RAG", "graphrag": "Agentic Hybrid RAG", "hipporag": "HippoRAG", "raptor": "RAPTOR"}
ORDER = ["rag", "graphrag", "hipporag", "raptor"]
MEASURES = ["Answered right", "Gold facts ≥50%", "Gold facts ≥75%", "Gold facts 100%"]


def uid() -> str:
    return secrets.token_hex(10)


def base(kind: str, fid: str, x: float, y: float, w: float, h: float) -> dict:
    return {
        "id": uid(), "type": kind, "x": x, "y": y, "width": w, "height": h, "angle": 0,
        "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
        "groupIds": [], "frameId": fid, "roundness": None, "seed": rng.randint(1, 2**31),
        "version": 1, "versionNonce": rng.randint(1, 2**31), "isDeleted": False,
        "boundElements": [], "updated": NOW, "link": None, "locked": False,
        "customData": dict(TAG),
    }


def text(fid: str, x: float, y: float, s: str, size: int = 44, width: float | None = None,
         align: str = "left", color: str = "#1e1e1e") -> float:
    """Place text; wraps to width. Returns its height."""
    chars = int(width / (size * 0.52)) if width else 10**6
    lines = []
    for para in s.split("\n"):
        lines += textwrap.wrap(para, chars) or [""]
    body = "\n".join(lines)
    w = width or max(len(line) for line in lines) * size * 0.55
    h = len(lines) * size * 1.25
    e = base("text", fid, x, y, w, h)
    e.update({
        "strokeColor": color, "text": body, "originalText": body, "fontSize": size,
        "fontFamily": 5, "textAlign": align, "verticalAlign": "top", "containerId": None,
        "autoResize": False, "lineHeight": 1.25,
    })
    if body.strip():  # Excalidraw deletes empty text elements on its own
        elements.append(e)
    return h


def rect(fid: str, x: float, y: float, w: float, h: float, fill: str, stroke: str) -> None:
    e = base("rectangle", fid, x, y, w, h)
    e.update({"backgroundColor": fill, "strokeColor": stroke})
    elements.append(e)


def table(fid: str, x: float, y: float, title: str, header: list[str], rows: list[list],
          widths: list[int], colors: dict[int, str] | None = None, size: int = 44) -> float:
    """A titled table, one number per cell. Returns the height used."""
    colors = colors or {}
    used = text(fid, x, y, title, 36, sum(widths)) + 14
    row_h = size * 1.25 + 36
    hs = 36
    head_lines = max(len(textwrap.wrap(h, max(4, int((w - 20) / (hs * 0.52))))) or 1 for h, w in zip(header, widths))
    head_h = head_lines * hs * 1.25 + 30
    cx = x
    for h, w in zip(header, widths):
        rect(fid, cx, y + used, w, head_h, "#e9ecef", "#adb5bd")
        text(fid, cx + 10, y + used + 15, h, hs, w - 20, "center" if cx > x else "left")
        cx += w
    used += head_h
    for row in rows:
        rect(fid, x, y + used, sum(widths), row_h, "#FFFFFF", "#ced4da")
        cx = x
        for i, (value, w) in enumerate(zip(row, widths)):
            if i in colors:
                rect(fid, cx, y + used, w, row_h, colors[i], "#ced4da")
            text(fid, cx + 10, y + used + 18, str(value), size, w - 20, "left" if i == 0 else "center")
            cx += w
        used += row_h
    return used


def grid(fid: str, x: float, y: float, blocks: list, per_row: int = 2, gap: int = 100) -> float:
    """Lay out table() calls two per row; blocks are kwargs dicts. Returns height."""
    used = 0.0
    for start in range(0, len(blocks), per_row):
        tallest = 0.0
        cx = x
        for b in blocks[start : start + per_row]:
            tallest = max(tallest, table(fid, cx, y + used, **b))
            cx += sum(b["widths"]) + gap
        used += tallest + 70
    return used


def section(fid: str, dataset: str, kind: str, label: str, facts: str, note: str, y: float) -> float:
    """One question type of one dataset: all methods, then each method vs Normal RAG."""
    x = frames[fid]["x"] + 64
    full = frames[fid]["width"] - 128
    t = report[dataset]["types"][kind]
    y0 = y
    y += text(fid, x, y, label, 72, full) + 20
    y += text(fid, x, y, facts, 40, full, color="#343a40") + 40
    ms = [m for m in MEASURES if m in t["measures"]]
    y += text(fid, x, y, "Closed-book: the same answer model with no retrieved text, to show what it already knows.", 40, full, color="#343a40") + 30
    y += text(fid, x, y, f"All 4 methods, questions out of {t['questions']} in each repeat (gold facts: out of {t['with_gold_facts']})", 56, full) + 30
    blocks = []
    for m in ms:
        e = t["measures"][m]
        rows = [[NAMES[k]] + e["methods"][k]["per_repeat"] + [e["methods"][k]["fail_all_3"]] for k in ORDER]
        if m == "Answered right":
            rows.append(["Closed-book"] + e["closed_book_per_repeat"] + ["–"])
        head = ["Method", "Repeat 1", "Repeat 2", "Repeat 3", "Wrong in all 3" if m == "Answered right" else "Not found in all 3"]
        blocks.append({"title": f"{m}, out of {e['questions']}", "header": head, "rows": rows, "widths": [560, 190, 190, 190, 300]})
    y += grid(fid, x, y, blocks)
    for k in ["graphrag", "hipporag", "raptor"]:
        y += text(fid, x, y, f"{NAMES[k]} vs Normal RAG, each repeat", 56, full) + 30
        blocks = []
        for m in ms:
            e = t["measures"][m]
            cells = e["methods"][k]["vs_normal_rag_per_repeat"]
            rows = [[f"Repeat {r + 1}"] + cells[r] for r in range(3)]
            word = "right" if m == "Answered right" else "found"
            blocks.append({
                "title": f"{m}, out of {e['questions']}",
                "header": ["", "Both " + word, f"{NAMES[k].split()[0]} only", "Normal RAG only", "Neither"],
                "rows": rows, "widths": [280, 270, 330, 330, 210],
                "colors": {2: "#d3f9d8", 3: "#ffe3e3"},
            })
        y += grid(fid, x, y, blocks)
    u0 = t["measures"][ms[0]]["agentic_never_tuned"]
    y += text(fid, x, y, f"Questions never used to tune Agentic Hybrid RAG: {u0['questions']} of {t['questions']}", 56, full) + 30
    blocks = []
    for m in ms:
        u = t["measures"][m]["agentic_never_tuned"]
        rows = [["Agentic Hybrid RAG"] + u["graphrag_per_repeat"], ["Normal RAG"] + u["rag_per_repeat"]]
        blocks.append({"title": f"{m}, out of {u['questions']}", "header": ["", "Repeat 1", "Repeat 2", "Repeat 3"], "rows": rows, "widths": [560, 280, 280, 280]})
    y += grid(fid, x, y, blocks)
    if note:
        y += text(fid, x, y, note, 48, full) + 40
    return y - y0


def start(fid: str) -> float:
    f = frames[fid]
    y = f["y"] + heights.get(fid, f["height"]) + 60
    rect(fid, f["x"] + 64, y, f["width"] - 128, 8, "#1e1e1e", "#1e1e1e")
    return y + 80


def finish(fid: str, y: float) -> None:
    heights[fid] = y + 80 - frames[fid]["y"]


def tokens_line(dataset: str) -> str:
    c = report[dataset]["context_tokens"]
    return "Tokens passed to the answer model, median and largest: " + "; ".join(
        f"{NAMES[k]} {int(c[k]['median']):,} and {int(c[k]['max']):,}" for k in ORDER
    ) + "."


MH_FACTS = {
    "comparison_query": ("Comparison", "Comparison questions compare two articles or entities. 125 per repeat (the held-out 100 plus 25 new), out of 856 in the dataset."),
    "temporal_query": ("Temporal", "Temporal questions ask about the order of events or whether two dates agree. 125 per repeat (the held-out 100 plus 25 new), out of 583 in the dataset."),
    "inference_query": ("Inference", "Inference questions name an entity by combining facts. 125 per repeat (the held-out 100 plus 25 new), out of 816 in the dataset."),
    "null_query": ("Null", "Null questions cite articles not in the collection, so the only right reply is \"Insufficient information\". 125 per repeat (the held-out 100 plus 25 new), out of 301. No gold facts."),
}
COMMON = (" Agentic Hybrid RAG: final 13 Nov version (the agent picks the evidence; graph rows carry graph facts only). "
          "Normal RAG, RAPTOR and HippoRAG: the 100 held-out answers are reused from 2-7 Nov, the 25 new questions answered on 13 Nov with the same code and indexes. "
          "Scored on 13 Nov: Answered right = exact match, then a fresh Claude Haiku judge; gold facts = Claude Haiku judge on facts frozen once per question, every retrieval judged again.")
SLIDE = {"comparison_query": "comparison", "temporal_query": "temporal", "inference_query": "infnull", "null_query": "infnull"}
for kind in ["comparison_query", "temporal_query", "inference_query", "null_query"]:
    fid = FRAME[SLIDE[kind]]
    y = start(fid)
    label, facts = MH_FACTS[kind]
    used = section(fid, "multihop_rag", kind, f"{label} questions · 500-question run · 13 Nov 2026",
                   "MultiHop-RAG, 609 news articles, 3 repeats. " + facts + COMMON + " " + tokens_line("multihop_rag"),
                   "", y)
    finish(fid, y + used)

if "hotpotqa" in report:
    fid = FRAME["hotpot"]
    for kind, label, facts in [
        ("bridge", "HotpotQA bridge questions (closest MultiHop-RAG type: inference)", "250 bridge questions: the answer needs a fact from one page to find the second page."),
        ("comparison", "HotpotQA comparison questions (MultiHop-RAG type: comparison)", "250 comparison questions: compare two entities, each on its own page."),
    ]:
        y = start(fid)
        used = section(fid, "hotpotqa", kind, f"{label} · 500-question run · 13 Nov 2026",
                       "HotpotQA distractor validation split, 500 questions picked with seed 42 (250 bridge, 250 comparison) out of 7,405; "
                       "one shared collection of all 4,860 paragraphs of the 500 questions (5,122 chunks); 3 repeats; all 4 methods built and answered on 13 Nov. "
                       + facts + " Agentic Hybrid RAG: final 13 Nov version. 27 of the 500 questions appeared in older HotpotQA runs and are left out of the never-tuned tables. "
                       + tokens_line("hotpotqa"), "", y)
        finish(fid, y + used)

if "natural_questions" in report:
    fid = FRAME["nq"]
    y = start(fid)
    used = section(fid, "natural_questions", "all", "Natural Questions · 500-question run · 14 Nov 2026 (no question types)",
                   "Validation rows 0-499 in order, no filtering; one shared collection of their 497 distinct Wikipedia pages (21,554 chunks); 3 repeats; all 4 methods. "
                   "Gold facts: 356 questions, the 358 with an evidence paragraph minus 2 whose gold answer is a large table (965 and 1,525 facts, too many to judge). Gold \"No answer\": \"Insufficient Information\" counts as right. "
                   "Rows 0-99 are the NQ 100 pilot (one index per page there; not pooled with this run) and are left out of the never-tuned tables. "
                   + tokens_line("natural_questions"), "", y)
    finish(fid, y + used)


# ---------- slide 01: run map rows ----------
COLS = [(3414, 340), (3754, 840), (4594, 180), (4774, 290), (5064, 560), (5624, 382), (6006, 280)]


def map_row(fid: str, y: float, cells: list[str], color: str) -> float:
    chars = [max(4, int((w - 30) / (40 * 0.52))) for _, w in COLS]
    lines = max(len(textwrap.wrap(c, n)) for c, n in zip(cells, chars))
    h = lines * 40 * 1.25 + 50
    for i, ((cx, w), c) in enumerate(zip(COLS, cells)):
        rect(fid, cx, y, w, h, color if i == 0 else "#FFFFFF", "#ced4da")
        text(fid, cx + 15, y + 25, c, 40, w - 30, "left" if i in (0, 1, 4) else "center")
    return h


fid = FRAME["map"]
y = frames[fid]["y"] + frames[fid]["height"] - 66
SCORE = "Answered right: exact match, then a fresh Claude Haiku judge. Gold facts found: Claude Haiku judge, facts frozen once per question"
rows = [
    (["Agentic 13 Nov check, 38 questions", "MultiHop-RAG: the 38 comparison questions whose gold facts swung most across the 10 Nov repeats. Agentic Hybrid RAG only: 10 Nov vs agent picks vs final (graph facts only)", "3", "13 Nov 2026", "Answered right: exact match, then Claude Haiku judge. Gold facts found: Claude Haiku judge", "Chose the final version. Tuning set, not a result", "03"], "#ffd8a8"),
    (["MultiHop-RAG 500", "609 news articles. The held-out 400 plus 25 new per type never used before (500: 125 per type). Agentic final version answers all 500; the other 3 reuse their held-out answers and answer the 100 new", "3", "13 Nov 2026 (400 reused from 2 to 7 Nov)", SCORE + "; every retrieval judged again", "Main result", "04, 05, 06"], "#d3f9d8"),
    (["HotpotQA 500", "Distractor validation: 250 bridge + 250 comparison (seed 42) out of 7,405; one shared collection of their 4,860 paragraphs", "3", "13 Nov 2026", SCORE, "Second dataset", "08"], "#fff3bf"),
    (["Natural Questions 500", "Validation rows 0 to 499; one shared collection of their 497 Wikipedia pages (21,554 chunks)", "3", "13 to 14 Nov 2026", SCORE + "; gold \"No answer\": \"Insufficient Information\" counts as right", "Control: single-page questions", "07"], "#d0ebff"),
]
for cells, color in rows:
    if "Natural Questions" in cells[0] and "natural_questions" not in report:
        continue
    y += map_row(fid, y, cells, color)
y += 40
y += text(fid, 3414, y, "Rows above the 13 Nov rows are older runs. The 500-question runs replace them as the main result; the older results stay on their slides. "
          "Every 500-question run: gpt-6-luna (medium reasoning) answers, text-embedding-3-small, 256-token chunks, 10 items to the answer model, "
          "closed-book baseline (no retrieval) 3 repeats. Index builds sent at OpenAI's flex tier (Batch API price), same prompts.", 36, 2872) + 40
heights[fid] = y - frames[fid]["y"]

# ---------- slide 03: the 13 Nov change and the 38-question check ----------
cmp38 = json.loads(Path("runs/agentic_tune/graph_facts/compare.json").read_text())
fid = FRAME["changes"]
y = start(fid)
x = frames[fid]["x"] + 64
full = frames[fid]["width"] - 128
y += text(fid, x, y, "What changed: 10 Nov → 13 Nov (final version)", 110, full) + 30
for head, body in [
    ("13 Nov, step 1: the agent picks the evidence",
     "Before, code kept the 10 vector passages with the highest similarity scores and graph rows only filled empty places, so no graph row ever reached the answer model "
     "(0 of 300 comparison runs), and in 59 of 111 runs missing a named article the graph lookup had found it. Now every result line carries a short ID and the agent "
     "ends by listing the IDs the answer needs, up to 10, from either tool; code puts those exact texts in the answer prompt and fills any free places by the old rule."),
    ("13 Nov, step 2: graph rows carry graph facts only",
     "Some graph rows had collected every chunk of an article (up to 14,600 tokens). Now the passage text field is hidden from the Cypher writer, a query that returns it "
     "is refused, and lists in a row keep at most 20 items. Passage text comes only from vector_search."),
]:
    y += text(fid, x, y, head, 56, full) + 20
    y += text(fid, x, y, body, 44, full) + 40
y += text(fid, x, y, "The 38-question check: MultiHop-RAG comparison questions whose gold facts swung most across the 10 Nov repeats, 3 repeats per version. "
          "These questions were used to choose the final version, so they are a tuning set, not a result.", 44, full, color="#343a40") + 40
blocks = []
for m in MEASURES:
    e = cmp38[m]
    rows = [[label] + e[key]["per_repeat"] + [e[key]["in_all_3"], e[key]["in_none"]]
            for label, key in [("10 Nov version", "10nov"), ("Agent picks", "picks"), ("Final: graph facts only", "facts")]]
    word = "right" if m == "Answered right" else "found"
    blocks.append({"title": f"{m}, out of 38", "header": ["Version", "Repeat 1", "Repeat 2", "Repeat 3", f"{word.title()} in all 3", f"{word.title()} in none"],
                   "rows": rows, "widths": [470, 170, 170, 170, 200, 200]})
y += grid(fid, x, y, blocks)
c = cmp38["context_tokens"]
y += text(fid, x, y, f"Tokens passed to the answer model, largest: 10 Nov {c['10nov']['max']:,}; agent picks {c['picks']['max']:,}; final {c['facts']['max']:,} "
          f"(medians {int(c['10nov']['median']):,}, {int(c['picks']['median']):,}, {int(c['facts']['median']):,}).", 44, full) + 60
finish(fid, y)

# ---------- slide 02: final prompts (verbatim) ----------
prompts = json.loads(Path("runs/final500/prompts_final.json").read_text())
fid = FRAME["methods"]
y = start(fid)
x = frames[fid]["x"] + 64
full = frames[fid]["width"] - 128
y += text(fid, x, y, "Agentic Hybrid RAG final version (13 Nov): prompts changed, exact text", 72, full) + 30
for label, body in prompts:
    y += text(fid, x, y, label, 30, full, color="#343a40") + 10
    y += text(fid, x, y, body, 26, full) + 40
finish(fid, y)

# ---------- slide 11: judges of the 500-question runs (verbatim) ----------
judge = json.loads(Path("runs/final500/judge_prompts.json").read_text())
fid = FRAME["backup"]
y = start(fid)
x = frames[fid]["x"] + 64
full = frames[fid]["width"] - 128
y += text(fid, x, y, "500-question runs (13 Nov): scoring and every judge prompt", 72, full) + 30
y += text(fid, x, y, judge["rules"], 40, full) + 40
for label, body in judge["prompts"]:
    y += text(fid, x, y, label, 30, full, color="#343a40") + 10
    y += text(fid, x, y, body, 26, full) + 40
if judge.get("audit"):
    a = judge["audit"]
    o = a["overall"]
    names = {"multihop_rag": "MultiHop-RAG", "hotpotqa": "HotpotQA", "natural_questions": "Natural Questions"}
    per = "; ".join(f"{names[d]} {v['agree']} of {v['n']}" for d, v in a["by_dataset"].items())
    pending = "" if "natural_questions" in a["by_dataset"] else " Natural Questions (66 facts) is added after its answers are judged."
    y += text(
        fid, x, y,
        f"Claude Opus audit of Haiku's gold-fact verdicts: {o['agree']} of {o['n']} agree (kappa {o['kappa']}); {per}. "
        f"Sample stratified by dataset, method and Haiku found / not found; Opus saw no method name and no Haiku verdict; "
        f"the same in-code quote check. Disagreements: {a['disagreements_note']}{pending}",
        40, full,
    ) + 40
finish(fid, y)

only = sys.argv[2] if len(sys.argv) > 2 else None
if only:
    elements = [e for e in elements if e["frameId"] == FRAME[only]]
    heights = {k: v for k, v in heights.items() if k == FRAME[only]}
(F / "deck_ops.json").write_text(json.dumps({"frames": heights, "elements": elements}))
print(len(elements), "new elements;", {k: round(v) for k, v in heights.items()})
