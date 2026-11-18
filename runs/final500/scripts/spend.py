"""Metered OpenAI spend of the final 500-question study so far (billed USD).

Sums finished index builds (build/build.json) and every question row
(results.jsonl, estimated_question_cost_usd, which includes retrieval and
answer calls) under runs/final500, plus closed-book rows. Builds still running
are not metered until they finish.
"""

import json
from pathlib import Path

R = Path("runs/final500")
builds = sum(
    json.loads(path.read_text()).get("billed_usd") or 0
    for path in R.glob("*/*/build/build.json")
)
questions = 0.0
rows = 0
seen: set[Path] = set()
for path in [*R.glob("*/*/r*/shard*/results.jsonl"), *R.glob("closed_book/**/results.jsonl")]:
    real = path.resolve()
    if real in seen:
        continue
    seen.add(real)
    for line in real.open():
        row = json.loads(line)
        questions += row.get("estimated_question_cost_usd") or 0
        rows += 1
print(f"${builds + questions:.2f} (builds ${builds:.2f}, {rows} question rows ${questions:.2f})")
