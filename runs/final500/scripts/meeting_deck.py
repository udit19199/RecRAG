"""Meeting slides for the 18 Nov HPE meeting, added to the Experiments scene.

Usage (repo root): python3 runs/final500/scripts/meeting_deck.py BACKUP.json OUT.json
Reads a scene backup (excalidraw scenes content get), copies the text and box
styles of slide 09, and writes a patch with three new frames:
  12  which RAG to use for each question type (summary, words only; numbers
      stay on the results slides)
  13  next phase, proposal A
  14  next phase, options B and C
Every new element carries customData.recragDeck = "2026-11-14-meeting", so the
results-slide scripts (tag 2026-11-13-final500) never touch them. Nothing
existing is changed. Apply with: excalidraw scenes content patch <id> --file OUT.json
"""

import copy
import json
import random
import secrets
import sys
import textwrap
import time

TAG = "2026-11-14-meeting"
backup = json.load(open(sys.argv[1]))
els = [e for e in backup["elements"] if not e.get("isDeleted")]
rng = random.Random(20261114)
NOW = int(time.time() * 1000)

# Style templates from slide 09 ("What explains the results").
f9 = sorted([e for e in els if e.get("frameId") == "ryf2jgl9mwbrmqpnktd"], key=lambda e: e["y"])
T_TITLE = next(e for e in f9 if e["type"] == "text" and e["fontSize"] == 110)
T_SUB = next(e for e in f9 if e["type"] == "text" and e["fontSize"] == 44)
T_BOX = next(e for e in f9 if e["type"] == "rectangle")
T_HEAD = next(e for e in f9 if e["type"] == "text" and e["fontSize"] == 48)
T_FRAME = next(e for e in els if e["id"] == "ryf2jgl9mwbrmqpnktd")

max_index = max(e["index"] for e in backup["elements"] if e.get("index"))
B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
counter = 0
out: list[dict] = []


def next_index() -> str:
    # Keys after every existing key: max_index + two base62 chars, last never "0".
    global counter
    counter += 1
    hi, lo = divmod(counter, 61)
    return max_index + B62[hi + 1] + B62[lo + 1]


def stamp(e: dict, fid: str | None) -> dict:
    e.update(
        id=secrets.token_hex(10), frameId=fid, groupIds=[], boundElements=[],
        seed=rng.randint(1, 2**31), versionNonce=rng.randint(1, 2**31), version=1,
        updated=NOW, isDeleted=False, locked=False, link=None, index=next_index(),
        customData={"recragDeck": TAG},
    )
    out.append(e)
    return e


def wrap(text: str, size: int, width: float) -> str:
    chars = max(10, int(width / (size * (0.6 if size >= 100 else 0.52))))
    return "\n".join(
        "\n".join(textwrap.wrap(p, chars)) if p else "" for p in text.split("\n")
    )


def text(fid, x, y, s, size=40, width=1300, tpl=None, color=None, family=None):
    e = copy.deepcopy(tpl or T_SUB)
    s = wrap(s, size, width)
    lines = s.count("\n") + 1
    lh = e.get("lineHeight", 1.25)
    longest = max(len(l) for l in s.split("\n"))
    e.update(x=x, y=y, fontSize=size, text=s, originalText=s, containerId=None,
             width=min(width, longest * size * 0.52), height=lines * size * lh)
    if color:
        e["strokeColor"] = color
    if family:
        e["fontFamily"] = family
    stamp(e, fid)
    return e


def box(fid, x, y, w, h, stroke=None, fill=None):
    e = copy.deepcopy(T_BOX)
    e.update(x=x, y=y, width=w, height=h)
    if stroke:
        e["strokeColor"] = stroke
    if fill:
        e["backgroundColor"] = fill
    return stamp(e, fid)


def frame(x, y, name):
    e = copy.deepcopy(T_FRAME)
    e.update(x=x, y=y, width=3000, height=1000, name=name)
    return stamp(e, None)


def header(fid, fx, fy, title, sub):
    tt = text(fid, fx + 64, fy + 64, title, 110, 2860, T_TITLE)
    t = text(fid, fx + 64, tt["y"] + tt["height"] + 24, sub, 44, 2860, T_SUB)
    return t["y"] + t["height"] + 60


GREY, WARN_S, WARN_F, PICK_S, PICK_F = "#868e96", "#e8590c", "#fff4e6", "#2f9e44", "#ebfbee"


def cell_rows(fid, x0, y, cols, rows, size=40, head_fill="#e9ecef"):
    """cols: list of (title, width). rows: list of lists of strings. Returns bottom y."""
    pad = 28
    for i, row in enumerate([[c for c, _ in cols]] + rows):
        wrapped = [wrap(v, size, w - 2 * pad) for v, (_, w) in zip(row, cols)]
        h = max(v.count("\n") + 1 for v in wrapped) * size * 1.25 + 2 * pad
        x = x0
        for v, (_, w) in zip(row, cols):
            box(fid, x, y, w, h, "#ADB5BD", head_fill if i == 0 else "#ffffff")
            text(fid, x + pad, y + pad, v, size, w - 2 * pad, T_HEAD if i == 0 else T_SUB,
                 color="#343a40" if i == 0 else "#1e1e1e", family=None if i == 0 else 5)
            x += w
        y += h
    return y


def note(fid, x, y, w, title, body, stroke, fill, size=40):
    pad = 36
    tb = wrap(body, size, w - 2 * pad)
    h = 48 * 1.15 + 20 + (tb.count("\n") + 1) * size * 1.25 + 2 * pad
    box(fid, x, y, w, h, stroke, fill)
    text(fid, x + pad, y + pad, title, 48, w - 2 * pad, T_HEAD, color=stroke)
    text(fid, x + pad, y + pad + 75, body, size, w - 2 * pad, T_SUB, color="#1e1e1e", family=5)
    return y + h


FY = 5450

# ---------- 00 summary ----------
f0 = frame(40200, FY, "12 Summary: which RAG to use for each question type")
y = header(f0["id"], 40200, FY, "Which RAG to use for each question type",
           "Final 500-question study: 500 questions per dataset, 4 methods, 3 repeats. "
           "Normal RAG is the baseline. The numbers are on the slide named in each row.")
rows = [
    ["MultiHop-RAG: comparison", "Agentic Hybrid RAG",
     "Finds the most gold facts on every measure, in all 3 repeats, and answers more right than Normal RAG. "
     "Also ahead on the 25 new questions never used in tuning. Slide 04."],
    ["MultiHop-RAG: temporal", "Agentic Hybrid RAG",
     "Ahead on Answered right and gold facts in all 3 repeats, and on the 25 new questions. "
     "HippoRAG is second on Answered right. Slide 05."],
    ["MultiHop-RAG: inference", "Normal RAG is enough",
     "All four methods answer almost every question right. Agentic Hybrid RAG finds more of the "
     "gold facts, but the answers do not change. Slide 06."],
    ["MultiHop-RAG: null (no answer exists)", "Any method",
     "All four say \"not enough information\" on about the same number of questions. Slide 06."],
    ["HotpotQA: bridge", "Agentic Hybrid RAG or HippoRAG",
     "Agentic Hybrid RAG answers the most right; HippoRAG finds the most gold facts. Both are ahead "
     "of Normal RAG in all 3 repeats. Slide 08."],
    ["HotpotQA: comparison", "Normal RAG is enough",
     "All four methods are within a few questions of each other on every measure. Slide 08."],
    ["Natural Questions", "Normal RAG is enough",
     "All four answer about as many right. Agentic Hybrid RAG finds a few more gold facts, "
     "but no more right answers. Slide 07."],
]
y = cell_rows(f0["id"], 40264, y, [("Question type", 720), ("Use", 760), ("Why", 1392)], rows)
y = note(f0["id"], 40264, y + 60, 2872, "Read with care: the model already knows many MultiHop-RAG answers",
         "With no retrieval at all (closed-book), the answer model gets more MultiHop-RAG comparison "
         "questions right than any method, and about as many temporal questions as Agentic Hybrid RAG. "
         "So on MultiHop-RAG, Answered right mostly measures what the model already knows. The "
         "gold-facts measures are the fair test of retrieval there, and Agentic Hybrid RAG leads on them. "
         "On HotpotQA bridge questions, retrieval clearly helps. Closed-book results are on the results slides.",
         WARN_S, WARN_F)
y = note(f0["id"], 40264, y + 40, 2872, "In one line",
         "Agentic Hybrid RAG finds the evidence best when a question links several documents by comparison "
         "or by time. For the other question types, Normal RAG does as well and is simpler to run.",
         PICK_S, PICK_F)
f0["height"] = y - FY + 64

# ---------- 12 next phase A ----------
X12 = 43550
f12 = frame(X12, FY, "13 Next phase (proposal): network agents that learn safely")
y = header(f12["id"], X12, FY, "Next phase, proposal: network agents that learn without gaining power",
           "Draft for the team to confirm. Options B and C on the next slide can replace it. "
           "Combines the two areas the team picked: self-evolving agents and agent security.")
A = [
    ["Area", "Self-evolving agents, with agent security built in."],
    ["Sub-area", "Safe skill learning for network and IT operations agents."],
    ["Problem", "An agent that fixes network incidents can save what worked as a reusable skill, so it "
     "gets better over time. Two things go wrong. It learns to blame a fault even when the network is "
     "healthy, because every real fault rewards a confident diagnosis. And a poisoned ticket or log can "
     "teach it a bad skill, which it then runs with full permissions."],
    ["Method", "Grow a skill library on a network-incident test set. Every new skill starts as advice "
     "only. Permission to act is a separate grant, earned by checks: where the skill came from, a replay "
     "in a sandbox, and no rise in false alarms on healthy networks. Compare against no checks and "
     "against accuracy-only checks."],
    ["Data", "NIKA, a network troubleshooting benchmark with 54 issue types. FaulT-Bench, which sends "
     "tickets that claim faults on healthy networks. Poisoned skills built from published attack patterns."],
    ["Success", "Same or better fault finding, no rise in false alarms, 5% or fewer poisoned skills "
     "reaching action while 85% or more of good skills are kept. The test setup is released for others."],
    ["Why HPE", "Networking (Juniper, Mist, Marvis). HPE's pitch for agents is that they act under "
     "governance. In one sentence: the agent gets better without getting more permission."],
    ["Closest work", "FaulT-Bench found the false-alarm problem but did not study learning agents. "
     "EvoMal and SkillJack show skill poisoning, but their defences screen content or assume a human "
     "reviews every skill. We found no study of learning agents on NIKA or FaulT-Bench."],
    ["Resources", "[ TO FILL: model and compute budget. Not settled yet. ]"],
]
y = cell_rows(f12["id"], X12 + 64, y, [("", 520), ("Proposal A", 2352)], A, head_fill="#ebfbee")
# Mark the blank: the last row's value cell gets the warning colours.
blank = [e for e in out if e.get("text", "").startswith("[ TO FILL")][0]
blank["strokeColor"] = WARN_S
cell = [e for e in out if e["type"] == "rectangle" and e["frameId"] == f12["id"]][-1]
cell["strokeColor"], cell["backgroundColor"] = WARN_S, WARN_F
f12["height"] = y - FY + 64

# ---------- 13 options B and C ----------
X13 = 46900
f13 = frame(X13, FY, "14 Next phase: options B and C")
y = header(f13["id"], X13, FY, "Next phase: the other two options",
           "Either can replace the proposal on slide 13. Same format, so the swap is one slide.")
BC = [
    ["Title", "The approved-change tax: how much real work does an agent's safety boundary block?",
     "A steadier search agent: let our agent improve its own search habits"],
    ["Area", "Agent security. Prompt-injection defences for agents that change infrastructure.",
     "Agentic retrieval, with self-evolution."],
    ["Problem", "A boundary that stops injected actions (a command hidden in a syslog line or a Wi-Fi "
     "client name) also blocks legitimate changes. NetInjectBench: undefended agents took unsafe "
     "actions in 82.5% of attacks; a static allowlist cut that to 5% but blocked every approved change. "
     "No one has measured this trade against an attacker who adapts.",
     "Our Agentic Hybrid RAG gave different answers from run to run and wasted long search loops. "
     "Self-evolution papers report average accuracy, not whether the same question gets the same answer."],
    ["Method", "200 IT and network scenarios. Four boundary designs: allowlist, policy gate, "
     "planner-reader split, LLM checker. An attacker who knows the policy.",
     "Keep the retriever and graph fixed. The agent rewrites its own search instructions from its "
     "failures. Scored with this phase's judges, plus whether all 3 repeats agree."],
    ["Success", "A public test setup, and one boundary that keeps 90% or more of approved changes "
     "while unsafe actions stay at 5% or less.",
     "Same or better answers with more repeats agreeing. About 4 to 5 months, reusing this phase's code."],
    ["Covers both team picks", "Security only", "Self-evolving only"],
    ["Main risk", "Fast-moving field; close work exists", "Smaller contribution"],
]
y = cell_rows(f13["id"], X13 + 64, y, [("", 520), ("Option B", 1176), ("Option C", 1176)], BC)
f13["height"] = y - FY + 64

old = [dict(e, isDeleted=True, version=e["version"] + 1, updated=NOW)
       for e in els if (e.get("customData") or {}).get("recragDeck") == TAG]
json.dump({"elements": old + out}, open(sys.argv[2], "w"))
print("removing", len(old), "earlier meeting elements")
print(len(out), "elements;", {f["name"]: round(f["height"]) for f in (f0, f12, f13)})
