"""Create a reproducible 1,000-question HotpotQA hard-bridge sample."""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

SAMPLE_SIZE = 1000
SAMPLE_SEED = 42
DATASET_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets/hotpotqa/hotpotqa_distractor_validation_hard_bridge_1000_seed_42.json"
)


def main() -> None:
    rows = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        revision=DATASET_REVISION,
    )
    candidate_indices = [
        index
        for index in range(len(rows))
        if rows[index]["type"] == "bridge" and rows[index]["level"] == "hard"
    ]
    if len(candidate_indices) < SAMPLE_SIZE:
        raise ValueError(
            f"Expected at least {SAMPLE_SIZE} hard bridge questions, "
            f"found {len(candidate_indices)}."
        )
    selected_indices = sorted(
        random.Random(SAMPLE_SEED).sample(candidate_indices, SAMPLE_SIZE)
    )
    selected_rows = [rows[index] for index in selected_indices]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(selected_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(selected_rows)} questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
