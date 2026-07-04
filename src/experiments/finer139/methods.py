"""The five graph-construction entity extractors, each in its ideal config.

Every extractor returns predicted entity spans (token index ranges) for a
sentence. Detection is type-agnostic, so predicted spans carry no label. Fair
scoring (numeric-only universe) is applied later in ``scoring``.

Methods:
  1. LLM-Based (open-ended prompting)
  2. NLP / OpenIE (spaCy)
  3. Ontology / Schema-Driven (XBRL gazetteer + numeric regex)
  4. Hybrid (schema-guided LLM + numeric rule filter)
  5. Dynamic / Incremental (memory-augmented, streaming wrapper over Hybrid)
"""

from __future__ import annotations

import json
import re
from typing import Any

from experiments.finer139.schema import NUMERIC_REGEX, is_numeric
from experiments.finer139.types import Sentence, Span

METHOD_DISPLAY_NAMES: dict[str, str] = {
    "llm": "LLM-Based (open-ended)",
    "nlp": "NLP / OpenIE (spaCy)",
    "ontology": "Ontology / Schema-Driven",
    "hybrid": "Hybrid (Schema-Guided LLM)",
    "dynamic": "Dynamic / Incremental (memory-augmented)",
}

ALL_METHODS: list[str] = ["llm", "nlp", "ontology", "hybrid", "dynamic"]

OPEN_PROMPT = (
    "You are an information-extraction system building a knowledge graph from "
    "financial filings.\n"
    "Extract every entity and value from the sentence below: monetary amounts, "
    "percentages, rates, share counts, dates, financial metrics, organizations, "
    "and instruments.\n"
    "Return ONLY a JSON array of the exact substrings as they appear in the "
    "sentence, with no extra commentary.\n\n"
    "Sentence:\n{sentence}"
)

HYBRID_PROMPT = (
    "You are extracting XBRL-tagged numeric facts from a financial-filing "
    "sentence to build a knowledge graph.\n"
    "The graph schema uses these XBRL concept types:\n{concepts}\n\n"
    "Identify the NUMERIC values in the sentence that correspond to any of these "
    "concepts (monetary amounts, percentages, rates, share counts, etc.).\n"
    "Return ONLY a JSON array of the exact numeric substrings as they appear in "
    "the sentence, with no extra commentary.\n\n"
    "Sentence:\n{sentence}"
)


# ── Alignment helpers ─────────────────────────────────────────────────────────


def align_char_span_to_tokens(
    char_start: int, char_end: int, token_offsets: list[tuple[int, int]]
) -> Span | None:
    """Map a character range onto the smallest covering token index range."""
    first: int | None = None
    last: int | None = None
    for i, (ts, te) in enumerate(token_offsets):
        if ts < char_end and char_start < te:  # overlap
            if first is None:
                first = i
            last = i
    if first is None or last is None:
        return None
    return Span(first, last + 1)


def align_surface_string_to_tokens(
    surface: str, sentence: Sentence, used: list[tuple[int, int]]
) -> Span | None:
    """Locate ``surface`` in the sentence text and map it to a token span.

    Prefers occurrences whose character range does not overlap one already
    consumed (tracked in ``used``) so repeated values map to distinct spans.
    """
    surface = surface.strip()
    if not surface:
        return None
    text_low = sentence.text.lower()
    needle = surface.lower()
    start = 0
    while True:
        pos = text_low.find(needle, start)
        if pos == -1:
            return None
        cs, ce = pos, pos + len(needle)
        if not any(cs < ue and us < ce for us, ue in used):
            span = align_char_span_to_tokens(cs, ce, sentence.token_offsets)
            if span is not None:
                used.append((cs, ce))
                return Span(span.start, span.end, None, surface)
        start = pos + 1


def dedup_spans(spans: list[Span]) -> list[Span]:
    """Drop duplicate spans that share the same token range."""
    seen: set[tuple[int, int]] = set()
    out: list[Span] = []
    for s in spans:
        key = (s.start, s.end)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# ── LLM output parsing ────────────────────────────────────────────────────────


def parse_json_list(raw: str) -> Any:
    """Best-effort parse of an LLM response into a Python list."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            return json.loads(match.group(0))
        except Exception:
            return []


def surfaces_from_items(data: Any) -> list[str]:
    """Normalize a parsed LLM list into a list of surface strings."""
    surfaces: list[str] = []
    if not isinstance(data, list):
        return surfaces
    for item in data:
        if isinstance(item, str):
            surfaces.append(item)
        elif isinstance(item, dict):
            for key in ("entity", "value", "text", "surface", "span", "token", "name"):
                val = item.get(key)
                if isinstance(val, str):
                    surfaces.append(val)
                    break
    return surfaces


def _align_surfaces(surfaces: list[str], sentence: Sentence) -> list[Span]:
    used: list[tuple[int, int]] = []
    spans: list[Span] = []
    for surface in surfaces:
        span = align_surface_string_to_tokens(surface, sentence, used)
        if span is not None:
            spans.append(span)
    return spans


# ── Extractors ────────────────────────────────────────────────────────────────


class BaseExtractor:
    name: str = "base"
    uses_llm: bool = False

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def display_name(self) -> str:
        return METHOD_DISPLAY_NAMES.get(self.name, self.name)

    def predict(self, sentence: Sentence) -> list[Span]:
        raise NotImplementedError


class LLMExtractor(BaseExtractor):
    """Open-ended prompting: ask the LLM for all entities/values."""

    name = "llm"
    uses_llm = True

    def __init__(self, llm: Any) -> None:
        super().__init__()
        self.llm = llm

    def predict(self, sentence: Sentence) -> list[Span]:
        raw = self.llm.generate(OPEN_PROMPT.format(sentence=sentence.text))
        self.call_count += 1
        surfaces = surfaces_from_items(parse_json_list(raw))
        return dedup_spans(_align_surfaces(surfaces, sentence))


class SpacyExtractor(BaseExtractor):
    """Traditional NLP/OpenIE: spaCy NER, numeric-yielding entity labels."""

    name = "nlp"
    KEEP = {"MONEY", "PERCENT", "CARDINAL", "QUANTITY", "DATE", "ORDINAL"}

    def __init__(self, model: str = "en_core_web_sm") -> None:
        super().__init__()
        import spacy  # lazy: optional 'experiments' extra

        try:
            self.nlp = spacy.load(model, disable=["lemmatizer"])
        except OSError as exc:  # model not downloaded
            raise RuntimeError(
                f"spaCy model '{model}' not installed. "
                f"Run: python -m spacy download {model}"
            ) from exc

    def predict(self, sentence: Sentence) -> list[Span]:
        doc = self.nlp(sentence.text)
        spans: list[Span] = []
        for ent in doc.ents:
            if ent.label_ not in self.KEEP:
                continue
            span = align_char_span_to_tokens(
                ent.start_char, ent.end_char, sentence.token_offsets
            )
            if span is not None:
                spans.append(Span(span.start, span.end, None, ent.text))
        return dedup_spans(spans)


class OntologyExtractor(BaseExtractor):
    """Schema-driven: if XBRL keywords are present, take numeric expressions."""

    name = "ontology"

    def __init__(self, gazetteer_regex: re.Pattern[str]) -> None:
        super().__init__()
        self.gazetteer_regex = gazetteer_regex

    def predict(self, sentence: Sentence) -> list[Span]:
        if not self.gazetteer_regex.search(sentence.text):
            return []
        spans: list[Span] = []
        for match in NUMERIC_REGEX.finditer(sentence.text):
            span = align_char_span_to_tokens(
                match.start(), match.end(), sentence.token_offsets
            )
            if span is not None:
                spans.append(Span(span.start, span.end, None, match.group(0)))
        return dedup_spans(spans)


class HybridExtractor(BaseExtractor):
    """Schema-guided LLM + numeric rule filter."""

    name = "hybrid"
    uses_llm = True

    def __init__(self, llm: Any, concept_names: list[str]) -> None:
        super().__init__()
        self.llm = llm
        self.concepts_block = ", ".join(concept_names)

    def predict(self, sentence: Sentence) -> list[Span]:
        raw = self.llm.generate(
            HYBRID_PROMPT.format(concepts=self.concepts_block, sentence=sentence.text)
        )
        self.call_count += 1
        surfaces = [
            s for s in surfaces_from_items(parse_json_list(raw)) if is_numeric(s)
        ]
        return dedup_spans(_align_surfaces(surfaces, sentence))


class DynamicExtractor(BaseExtractor):
    """Memory-augmented incremental extractor layered on the Hybrid base.

    Streams sentences in order. From each high-confidence base hit it learns the
    surrounding context words; it then also flags numeric expressions that sit
    near remembered context, so recognition can improve as more data is seen.
    Reuses the Hybrid base predictions, so it adds no extra LLM calls when
    Hybrid is also selected.
    """

    name = "dynamic"
    uses_llm = True
    WINDOW = 3

    def __init__(self) -> None:
        super().__init__()
        self.context_memory: set[str] = set()

    def predict_incremental(
        self, sentence: Sentence, base_spans: list[Span]
    ) -> list[Span]:
        for span in base_spans:
            self._learn_context(sentence, span)
        memory_spans = self._apply_memory(sentence)
        return dedup_spans(base_spans + memory_spans)

    def _learn_context(self, sentence: Sentence, span: Span) -> None:
        lo = max(0, span.start - self.WINDOW)
        hi = min(len(sentence.tokens), span.end + self.WINDOW)
        for i in range(lo, hi):
            if span.start <= i < span.end:
                continue
            word = sentence.tokens[i].lower()
            if word.isalpha() and len(word) >= 3:
                self.context_memory.add(word)

    def _apply_memory(self, sentence: Sentence) -> list[Span]:
        spans: list[Span] = []
        for match in NUMERIC_REGEX.finditer(sentence.text):
            span = align_char_span_to_tokens(
                match.start(), match.end(), sentence.token_offsets
            )
            if span is None:
                continue
            lo = max(0, span.start - self.WINDOW)
            hi = min(len(sentence.tokens), span.end + self.WINDOW)
            context = {sentence.tokens[i].lower() for i in range(lo, hi)}
            if context & self.context_memory:
                spans.append(Span(span.start, span.end, None, match.group(0)))
        return spans
