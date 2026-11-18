"""Export the claims stage of the gold-fact judge for every question of the study.

One frozen fact split per question, from its gold evidence, shared by every
method and repeat. Built straight from the frozen question sets (the same
supporting sentences every run saves), so it does not wait for any run.
Questions with no gold evidence (MultiHop-RAG null questions, NQ questions
without a long answer) get no facts and no retrieval numbers.
Writes runs/final500/judge/claims/items-*.json (10 questions per file, the
same guide and prompt version as evals/haiku_fact_judge.py). Questions are
shuffled across datasets with seed 20261113.
"""

import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ["RECRAG_HOTPOTQA_ROWS"] = str(ROOT / "runs/final500/hotpotqa/questions_rows.json")
os.environ["RECRAG_NQ_RECORDS"] = str(ROOT / "runs/final500/natural_questions/records.json")

from dataset import get_source  # noqa: E402
from evals.haiku_fact_judge import CLAIMS_GUIDE, ClaimsTask, _write_batches  # noqa: E402

questions = json.loads((ROOT / "runs/final500/questions.json").read_text())
sets = {
    "multihop_rag": questions["multihop_rag"]["all500"],
    "hotpotqa": questions["hotpotqa"]["record_indices"],
    "natural_questions": questions["natural_questions"]["record_indices"],
}
tasks = []
counts = {}
for dataset, indices in sets.items():
    with_gold = 0
    for index in indices:
        record = get_source(dataset).load_record(index)
        sentences = list(record.supporting_sentences())
        if sentences:
            tasks.append(ClaimsTask(id=record.id, sentences=sentences))
            with_gold += 1
    counts[dataset] = with_gold
assert len({t.id for t in tasks}) == len(tasks)
random.Random(20261113).shuffle(tasks)
out = ROOT / "runs/final500/judge"
_write_batches(out / "claims", CLAIMS_GUIDE, tasks, 10)
(out / "claims_questions.json").write_text(json.dumps(counts, indent=1))
print(counts, len(tasks), "questions")
