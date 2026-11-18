"""Freeze the question sets of the final 500-question study (13 Nov 2026).

Run from the repo root, before any question of the study runs:
PYTHONPATH=. uv run python runs/final500/scripts/select_questions.py

MultiHop-RAG: the 400 held-out questions (runs/heldout400/questions.json) plus
25 new per type, drawn with seed 20261113 from questions whose id appears in
no file anywhere under runs/ (any run, tuning run, judge file or question
list). HotpotQA: distractor validation split, 250 bridge + 250 comparison by
the dataset "type" field, seed 42; the 500 rows are saved as-is and their
paragraphs are the shared corpus. Natural Questions: validation rows 0-499 in
order, no filtering.
"""

import collections
import glob
import json
import random
import re
from hashlib import sha256
from pathlib import Path

from dataset import _multihop_queries, load_multihop_rag_record

OUT = Path("runs/final500")
MULTIHOP_SEED = 20261113
HOTPOT_SEED = 42

# MultiHop-RAG ---------------------------------------------------------------
heldout = json.loads(Path("runs/heldout400/questions.json").read_text())
old400 = list(heldout["order"])
used: set[int] = set()
scanned = 0
for path in glob.glob("runs/**/*", recursive=True):
    if not path.endswith((".json", ".jsonl", ".md", ".log")) or path.startswith(
        str(OUT)
    ):
        continue
    try:
        text = Path(path).read_text(errors="ignore")
    except (IsADirectoryError, OSError):
        continue
    scanned += 1
    used.update(int(m) for m in re.findall(r"multihop-rag-(\d+)", text))
used.update(old400)
by_type: dict[str, list[int]] = collections.defaultdict(list)
for index in range(len(_multihop_queries())):
    if index not in used:
        by_type[load_multihop_rag_record(index).question_type].append(index)
rnd = random.Random(MULTIHOP_SEED)
new = {kind: sorted(rnd.sample(pool, 25)) for kind, pool in sorted(by_type.items())}
new_order = [index for group in zip(*new.values()) for index in group]
assert not set(new_order) & used and len(set(new_order)) == 100
multihop = {
    "seed": MULTIHOP_SEED,
    "files_scanned": scanned,
    "excluded_used": len(used),
    "available": {kind: len(pool) for kind, pool in sorted(by_type.items())},
    "old400": old400,
    "new100_by_type": new,
    "new100": new_order,
    "all500": old400 + new_order,
}

# HotpotQA -------------------------------------------------------------------
from datasets import load_dataset  # noqa: E402

table = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
types = table["type"]
rnd = random.Random(HOTPOT_SEED)
picked = {
    kind: sorted(
        rnd.sample([i for i, value in enumerate(types) if value == kind], 250)
    )
    for kind in ("bridge", "comparison")
}
hotpot_indices = [i for group in zip(*picked.values()) for i in group]
rows = [table[i] for i in hotpot_indices]
rows_path = OUT / "hotpotqa" / "questions_rows.json"
rows_path.parent.mkdir(parents=True, exist_ok=True)
rows_path.write_text(json.dumps(rows, ensure_ascii=False))
hotpot = {
    "split": "distractor/validation",
    "seed": HOTPOT_SEED,
    "dataset_rows": len(table),
    "fingerprint": table._fingerprint,
    "by_type_source_index": picked,
    "order_source_index": hotpot_indices,
    "rows_file": str(rows_path),
    "rows_file_sha256": sha256(rows_path.read_bytes()).hexdigest(),
    "record_indices": list(range(len(rows))),
    "ids": [row["id"] for row in rows],
}

# Natural Questions ----------------------------------------------------------
nq = {
    "split": "default/validation",
    "rows": "0-499",
    "record_indices": list(range(500)),
    "note": "rows 0-99 are the NQ 100 pilot",
}

(OUT / "questions.json").write_text(
    json.dumps(
        {"multihop_rag": multihop, "hotpotqa": hotpot, "natural_questions": nq},
        indent=1,
    )
)
print(
    "multihop: excluded",
    len(used),
    "available",
    multihop["available"],
    "| hotpot rows",
    len(rows),
    "| nq 500",
)
