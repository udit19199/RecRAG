"""Shared value types for the FiNER-139 benchmark.

Kept in a dedicated module so ``dataset``, ``methods`` and ``scoring`` can share
them without import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Span:
    """An entity span expressed as a half-open token index range.

    ``start`` is inclusive, ``end`` is exclusive (``tokens[start:end]``).
    ``label`` holds the gold XBRL concept for gold spans and is ``None`` for
    predictions (detection is type-agnostic). ``text`` is the surface string.
    """

    start: int
    end: int
    label: str | None = None
    text: str | None = None


@dataclass
class Sentence:
    """A single FiNER-139 example, pre-tokenized with reconstructed text."""

    index: int
    tokens: list[str]
    text: str
    #: Character (start, end) offset of each token within ``text``.
    token_offsets: list[tuple[int, int]]
    gold_spans: list[Span] = field(default_factory=list)
