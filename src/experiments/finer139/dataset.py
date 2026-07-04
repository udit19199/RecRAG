"""Load and sample the FiNER-139 validation split.

FiNER-139 (``nlpaueb/finer-139``) is IOB2 token classification where the gold
"entities" are numeric tokens tagged with one of 139 XBRL concept types. We use
the validation split (its role in the GraphRAG analysis doc) and derive gold
entity spans from contiguous non-``O`` tag runs.

The upstream HF dataset ships as ``finer139.zip`` (JSONL) because the legacy
loading script is no longer supported by ``datasets`` 5.x. We load directly from
the zip via ``huggingface_hub`` (lazy import).
"""

from __future__ import annotations

import json
import random
import zipfile
from functools import lru_cache
from typing import Any

from experiments.finer139.types import Sentence, Span

DATASET_ID = "nlpaueb/finer-139"
SPLIT = "validation"
MAX_SAMPLE_SIZE = 500

_SPLIT_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}


def _download_zip_path() -> str:
    from huggingface_hub import hf_hub_download  # lazy

    return hf_hub_download(DATASET_ID, "finer139.zip", repo_type="dataset")


@lru_cache(maxsize=1)
def get_label_names() -> list[str]:
    """Return the IOB2 label names from ``dataset_infos.json``."""
    from huggingface_hub import hf_hub_download  # lazy

    path = hf_hub_download(DATASET_ID, "dataset_infos.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        infos = json.load(f)
    info = infos.get("finer-139") or next(iter(infos.values()))
    return list(info["features"]["ner_tags"]["feature"]["names"])


def concept_names_from_labels(label_names: list[str]) -> list[str]:
    """Extract the unique XBRL concept names (139) from IOB2 label names."""
    concepts: list[str] = []
    seen: set[str] = set()
    for name in label_names:
        if name == "O":
            continue
        _, _, concept = name.partition("-")
        if concept and concept not in seen:
            seen.add(concept)
            concepts.append(concept)
    return concepts


def _load_split_rows(split: str = SPLIT) -> list[dict[str, Any]]:
    """Load all rows for a split from the cached zip JSONL."""
    filename = _SPLIT_FILES.get(split)
    if filename is None:
        raise ValueError(f"Unknown split: {split}")

    zip_path = _download_zip_path()
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(filename) as handle:
            for raw in handle:
                line = raw.decode("utf-8").strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _gold_spans(tags: list[str]) -> list[Span]:
    """Derive entity spans from IOB2 string tags."""
    spans: list[Span] = []
    start: int | None = None
    label: str | None = None

    for i, name in enumerate(tags):
        if name == "O":
            if start is not None:
                spans.append(Span(start, i, label))
                start, label = None, None
            continue
        prefix, _, concept = name.partition("-")
        if start is None or prefix == "B" or concept != label:
            if start is not None:
                spans.append(Span(start, i, label))
            start, label = i, concept

    if start is not None:
        spans.append(Span(start, len(tags), label))
    return spans


def _reconstruct(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join tokens with single spaces and record each token's char offsets."""
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(tok)
        cursor += len(tok)
        offsets.append((start, cursor))
    return "".join(parts), offsets


def _span_text(tokens: list[str], span: Span) -> str:
    return " ".join(tokens[span.start : span.end])


def load_sample(
    sample_size: int = 100,
    seed: int = 42,
    require_gold: bool = True,
) -> list[Sentence]:
    """Return a seeded random sample of validation sentences.

    Args:
        sample_size: Number of sentences to return (capped at ``MAX_SAMPLE_SIZE``).
        seed: RNG seed for reproducible sampling.
        require_gold: When True, only include sentences with >=1 gold entity so
            recall is measurable (the natural split is dominated by ``O``).
    """
    size = max(1, min(sample_size, MAX_SAMPLE_SIZE))
    rows = _load_split_rows(SPLIT)

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)

    out: list[Sentence] = []
    for idx in order:
        if len(out) >= size:
            break
        row = rows[idx]
        tags = list(row["ner_tags"])
        gold = _gold_spans(tags)
        if require_gold and not gold:
            continue
        tokens = list(row["tokens"])
        text, offsets = _reconstruct(tokens)
        gold = [Span(s.start, s.end, s.label, _span_text(tokens, s)) for s in gold]
        out.append(
            Sentence(
                index=int(row.get("id", idx)),
                tokens=tokens,
                text=text,
                token_offsets=offsets,
                gold_spans=gold,
            )
        )
    return out
