"""Edit list for the 9 Nov text pass on the Experiments deck (truth check, clutter, plain wording).

Numbers in new text come from gold_facts_by_run.json and check_slide6_7_claims.py output.
Writes runs/heldout400/deck_text_pass_ops.json for the browser script.
"""
import json
from pathlib import Path

F1, F3, F4, F5 = "qWfPUNxVx8sJFI1rgXoJJ", "7fNjDMQmsNw4M-_kyqhx3", "_LGLLt7ziA8Nwhqwofiam", "X0d0Z-rbTxg2KzkX889Sz"
ops = {"text": [], "text_match": [], "delete_ids": [], "delete_region": [], "move": [], "frames": {}}

def t(idp, s, grow=False, maxw=None):
    ops["text"].append({"id": idp, "text": s, "grow": grow, "maxw": maxw})

def tm(frames, starts, s, exact=False):
    ops["text_match"].append({"frames": frames, "starts": starts, "text": s, "exact": exact})

# ---------- slide 1 ----------
t("_tSuey", "MultiHop-RAG, 609 news articles. 400 questions not used in any earlier run, 100 of each type, out of 2,556.")
t("gFsiam", "Exact match and contains answer")

# ---------- slides 3-5 ----------
T345 = [F3, F4, F5]
tm(T345, "All 4 methods: questions out of 100", "All four methods, out of 100 questions in each repeat")
tm(T345, "Agentic GraphRAG vs Normal RAG: questions out of 100", "Agentic GraphRAG vs Normal RAG, out of 100 questions in each repeat")
tm(T345, "Agentic GraphRAG: questions right or found in N", "Agentic GraphRAG, questions right or found in 3, 2, 1 or 0 repeats")
tm(T345, "Normal RAG: questions right or found in N", "Normal RAG, questions right or found in 3, 2, 1 or 0 repeats")
tm(T345, "Repeat", "", exact=True)
tm(T345, "Agentic GraphRAG only: Agentic GraphRAG fails",
   "Agentic GraphRAG only means Agentic GraphRAG fails in all 3 repeats and Normal RAG gets it in all 3. "
   "Normal RAG only is the reverse. This table undercounts Agentic GraphRAG, because its retrieval changes "
   "between repeats and Normal RAG's does not. The two tables below show how many repeats each method gets "
   "each question in.")
tm(T345, "Comparison: 100 per run",
   "Comparison questions compare two articles or entities. The same 100 in every repeat, out of 856 in the dataset.")
tm(T345, "Temporal: 100 per run",
   "Temporal questions ask about the order of events or whether two dates agree. The same 100 in every repeat, out of 583 in the dataset.")
tm(T345, "Inference: 100 per run",
   "Inference questions name an entity by combining facts. The same 100 in every repeat, out of 816 in the dataset.")
tm(T345, "Null: 100 per run",
   "Null questions cite articles that are not in the collection, so the only right reply is \"Insufficient information\". "
   "The same 100 in every repeat, out of 301 in the dataset.")
tm(T345, "Answers in the 100: Yes (63)", "Gold answers: 63 Yes, 33 No, and 4 other words such as Same or Different")
tm(T345, "Answers in the 100: Yes (48)", "Gold answers: 48 Yes, 41 No, 8 Consistent, and 3 other answers")
tm(T345, "Answers in the 100: one name", "Gold answers are names, most often Sam Bankman-Fried (42), Google (24) and Sam Altman (9)")
tm(T345, "Null: Answered right, out of 100", "Null questions, Answered right out of 100")
tm(T345, "The two methods are level.",
   "Agentic GraphRAG and Normal RAG are level here. On gold facts they are never more than 5 questions apart. "
   "On Answered right, Agentic GraphRAG is 3 ahead in repeat 1, 6 behind in repeat 2 and level in repeat 3. "
   "67 questions never reach 100% of gold facts for either method in any repeat.")
tm(T345, "Agentic GraphRAG leads on Answered right",
   "Agentic GraphRAG is ahead of Normal RAG in every repeat on Answered right, by 3 to 11 questions, and on gold "
   "facts found at ≥50% and ≥75%. At 100% they are level in repeats 1 and 3, and Agentic GraphRAG is 4 ahead in "
   "repeat 2. 78 questions never reach 100% of gold facts for either method in any repeat.")
tm(T345, "Every method answers 95 to 98",
   "Every method gets 95 to 98 of 100 right in every repeat, so the method barely matters here. On gold facts, "
   "Agentic GraphRAG and Normal RAG are never more than 4 questions apart. 85 questions never reach 100% of gold "
   "facts for either method in any repeat.")

# ---------- slide 6: drop the pooled sentence-level charts, keep the three points with gold-fact numbers ----------
F6 = "DA7MQHPcmU1W8YxDicf9M"
ops["delete_region"].append({"frame": F6, "ymin": 5690, "ymax": 6980, "xmin": 0, "xmax": 1e9})
for idp in ("FTFTOu", "nky4sH", "JIKVn9", "Lu1385", "vwBmZu", "QcWlZr"):
    ops["move"].append({"id": idp, "dy": -1290})
t("mJaxOv", "Found facts and right answers")
t("Lu1385", "1. Agentic GraphRAG mostly finds some of the facts, not all of them\n\n"
  "Comparison: it finds no gold facts on 8 to 12 questions per repeat, against 19 for Normal RAG, but all of "
  "them on 16 to 20, against 20. Temporal: no facts on 9 to 13 against 13, all facts on 15 to 19 against 15.\n\n"
  "With all gold facts found, both methods get 73 to 93% of those questions right. With only some found, 42 to 54%.",
  maxw=860)
t("vwBmZu", "2. \"Insufficient information\" usually means a missing article\n\n"
  "When Normal RAG or Agentic GraphRAG replies \"Insufficient information\" to a comparison or temporal question, "
  "at least one article the question needs was not retrieved in 94 to 100% of cases. Only 0 to 3 of these replies "
  "per repeat came with all gold facts found.", maxw=860)
t("QcWlZr", "3. On temporal questions, guessing pays\n\n"
  "Most temporal answers are Yes or No. HippoRAG replies \"Insufficient information\" to 32 or 33 of them per "
  "repeat, Normal RAG to 44 to 48. Of the 24 temporal questions where HippoRAG finds no gold facts, it still "
  "gets 10 to 12 right.", maxw=860)
ops["frames"][F6] = {"name": "06 Found facts and right answers", "fit_boxes": ["FTFTOu", "nky4sH", "JIKVn9"]}

# ---------- slide 7: drop pooled accuracy, cost and time; fix false claims ----------
F7 = "VpVINQ_RoLBET5_02fBC7"
ops["delete_region"].append({"frame": F7, "ymin": 5690, "ymax": 6300, "xmin": 24940, "xmax": 1e9})
ops["delete_region"].append({"frame": F7, "ymin": 6330, "ymax": 6800, "xmin": 0, "xmax": 1e9})
ops["delete_ids"] += ["PCYXlP"]
t("NRjZWs", "Is it a temporal question?")
t("fpXrnl", "HippoRAG\n59 to 60 right,\nNormal RAG 47 to 51")
t("jO3pFA", "Normal RAG\nlevel with the\nothers on comparison\nand inference")
t("LZDswo", "An idea, not built yet: send temporal questions to HippoRAG and the rest to Normal RAG. It needs a way to spot temporal questions.", maxw=1360)
for idp in ("Klh9ck", "l32BvY", "XA9xfJ", "8r9WmM", "i68TlD", "uNbWbH", "MZyHna", "dWEhaw", "NXT4MG",
            "XvN5Ip", "C-OdpO", "viczyT", "aWazZd", "O9e6Ys", "_BMZ2s", "gMdI4x", "QiCGYI", "9ppuLn", "K7BaDu", "YkvQ0S"):
    ops["move"].append({"id": idp, "dy": -440})
t("uNbWbH", "1. The question\n\nDo knowledge graphs make multi-hop RAG better? We compare Normal RAG, RAPTOR, "
  "HippoRAG and Agentic GraphRAG one question type at a time.", maxw=654)
t("MZyHna", "2. What we found\n\nOn comparison and inference questions, Normal RAG is as accurate as the graph "
  "methods. Graphs help on temporal questions: HippoRAG gets 59 to 60 of 100 right and Agentic GraphRAG 54 to 58, "
  "against 47 to 51 for Normal RAG. On comparison and temporal questions, most wrong answers come with an article "
  "missing from the retrieved text.", maxw=654)
t("dWEhaw", "3. What we have\n\n400 questions no method was tuned on, 3 repeats, the same answer model and prompt "
  "for every method, and the check of found facts against right answers on slide 6.", maxw=654)
t("NXT4MG", "4. Still needed\n\nRerun HotpotQA with 3 repeats. Build and test a way to spot temporal questions "
  "for the router. Add one more dataset. Have a judge check the refusals.", maxw=654)
t("viczyT", "Yes. All four methods use the same articles, chunks, answer model and prompt. Only retrieval is different.")
t("O9e6Ys", "No. Its tool order was picked on 100 earlier questions, and the 400 test questions leave out every question used before.")
t("gMdI4x", "Only for temporal questions, where it gets 54 to 58 of 100 right against 47 to 51 for Normal RAG. Elsewhere it is level.")
t("9ppuLn", "Inference and null questions are easy. Most inference answers are one of a few names, and every null answer is \"Insufficient information\".")
t("YkvQ0S", "It was an early check with one run. Rerunning it with 3 repeats is still on the list.")
ops["frames"][F7] = {"fit_boxes_same": ["l32BvY", "XA9xfJ", "8r9WmM", "i68TlD"]}

# ---------- slide 8: describe the measures actually shown; drop the unused answer-support prompt ----------
ops["delete_ids"] += ["L-2ivD", "5ufvQc"]
for idp in ("ec2CtV", "S6eglb", "8jFcGV", "jayOS4"):
    ops["move"].append({"id": idp, "dy": -1130})
t("ec2CtV", "GPT-6-LUNA GOLD-FACT JUDGE, STEP 1: split each gold sentence into claims, the gold facts")
t("rf2AJF", "Answered right: the answer matches a gold answer after lowercasing and removing punctuation and a, an, the. "
  "If not, the Claude Haiku judge on the left decides whether it names the same entity, value or yes or no. "
  "An \"Insufficient information\" reply that does not match counts as wrong. Gold facts found: the gpt-6-luna "
  "judge on the right splits each gold sentence into claims, then checks each claim against the retrieved text. "
  "A claim counts as found only when the judge marks it stated and its quote appears in the retrieved text.")

Path("runs/heldout400/deck_text_pass_ops.json").write_text(json.dumps(ops, ensure_ascii=False))
print(len(json.dumps(ops, ensure_ascii=False)), "bytes")
