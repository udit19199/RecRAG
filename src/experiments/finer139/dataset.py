"""Load and sample the FiNER-139 dataset (train / validation / test).

FiNER-139 (``nlpaueb/finer-139``) is IOB2 token classification where the gold
"entities" are numeric tokens tagged with one of 139 XBRL concept types.

The upstream HF dataset ships as ``finer139.zip`` (JSONL). Loaded lazily via
``huggingface_hub``. For offline CI, use ``load_fixture_corpus`` or a frozen
suite JSON that references inline fixture rows.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from experiments.finer139.types import Sentence, Span

DATASET_ID = "nlpaueb/finer-139"
SPLIT = "validation"
MAX_SAMPLE_SIZE = 500
VALID_SPLITS = frozenset({"train", "validation", "test"})

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
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split}")

    filename = _SPLIT_FILES[split]
    zip_path = _download_zip_path()
    rows: list[dict[str, Any]] = []
    import zipfile

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


def row_to_sentence(row: dict[str, Any], row_idx: int = 0) -> Sentence:
    """Convert a raw FiNER JSONL row into a ``Sentence``."""
    tags = list(row["ner_tags"])
    tokens = list(row["tokens"])
    text, offsets = _reconstruct(tokens)
    gold = _gold_spans(tags)
    gold = [Span(s.start, s.end, s.label, _span_text(tokens, s)) for s in gold]
    return Sentence(
        index=int(row.get("id", row_idx)),
        tokens=tokens,
        text=text,
        token_offsets=offsets,
        gold_spans=gold,
    )


def load_fixture_corpus(path: str | Path) -> list[Sentence]:
    """Load sentences from a local JSON fixture (offline / CI)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("sentences", data)
    return [row_to_sentence(row, i) for i, row in enumerate(rows)]


def load_frozen_suite(path: str | Path) -> tuple[list[Sentence], dict[str, Any]]:
    """Load a fixed sentence set from a suite spec JSON file.

    Suite format::

        {
          "version": 1,
          "split": "validation",
          "source": "hf" | "inline",
          "sentence_ids": [123, 456],   # when source=hf
          "sentences": [...]            # when source=inline
        }
    """
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    source = spec.get("source", "hf")
    if source == "inline":
        sentences = [
            row_to_sentence(row, i) for i, row in enumerate(spec.get("sentences", []))
        ]
        return sentences, spec

    split = spec.get("split", SPLIT)
    rows = _load_split_rows(split)
    id_map: dict[int, dict[str, Any]] = {}
    for i, row in enumerate(rows):
        id_map[int(row.get("id", i))] = row

    sentences: list[Sentence] = []
    for sid in spec["sentence_ids"]:
        row = id_map.get(int(sid))
        if row is None:
            raise ValueError(f"sentence id {sid} not found in split {split}")
        sentences.append(row_to_sentence(row, int(sid)))
    return sentences, spec


def _stratified_indices(
    rows: list[dict[str, Any]],
    sample_size: int,
    seed: int,
    require_gold: bool,
) -> list[int]:
    """Pick row indices with proportional allocation per primary gold concept."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        gold = _gold_spans(list(row["ner_tags"]))
        if require_gold and not gold:
            continue
        primary = gold[0].label if gold else "__none__"
        buckets[primary].append(i)

    if not buckets:
        return []

    rng = random.Random(seed)
    total_eligible = sum(len(v) for v in buckets.values())
    size = min(sample_size, total_eligible)
    # Proportional allocation with at least one from each non-empty bucket when possible.
    allocation: dict[str, int] = {}
    remaining = size
    for concept, indices in sorted(buckets.items(), key=lambda x: -len(x[1])):
        share = max(1, round(size * len(indices) / total_eligible)) if size >= len(buckets) else 0
        share = min(share, len(indices), remaining)
        if share > 0:
            allocation[concept] = share
            remaining -= share
    # Distribute any leftover to largest buckets.
    while remaining > 0:
        for concept in sorted(buckets, key=lambda c: -len(buckets[c])):
            if allocation.get(concept, 0) < len(buckets[concept]):
                allocation[concept] = allocation.get(concept, 0) + 1
                remaining -= 1
                if remaining <= 0:
                    break

    chosen: list[int] = []
    for concept, count in allocation.items():
        pool = buckets[concept][:]
        rng.shuffle(pool)
        chosen.extend(pool[:count])
    rng.shuffle(chosen)
    return chosen[:size]


def load_sample(
    sample_size: int = 100,
    seed: int = 42,
    require_gold: bool = True,
    split: str = SPLIT,
    stratified: bool = False,
    suite_path: str | Path | None = None,
) -> list[Sentence]:
    """Return a reproducible sample of FiNER-139 sentences.

    Args:
        sample_size: Number of sentences (capped at ``MAX_SAMPLE_SIZE``).
        seed: RNG seed for random sampling.
        require_gold: Skip sentences with no gold entities when True.
        split: ``train``, ``validation``, or ``test``.
        stratified: Proportional sampling by primary XBRL concept.
        suite_path: If set, load fixed sentences from a suite JSON (ignores size/seed).
    """
    if suite_path is not None:
        sentences, _ = load_frozen_suite(suite_path)
        return sentences

    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split: {split}")

    size = max(1, min(sample_size, MAX_SAMPLE_SIZE))
    rows = _load_split_rows(split)

    if stratified:
        indices = _stratified_indices(rows, size, seed, require_gold)
        return [row_to_sentence(rows[i], i) for i in indices]

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)

    out: list[Sentence] = []
    for idx in order:
        if len(out) >= size:
            break
        row = rows[idx]
        sentence = row_to_sentence(row, idx)
        if require_gold and not sentence.gold_spans:
            continue
        out.append(sentence)
    return out
