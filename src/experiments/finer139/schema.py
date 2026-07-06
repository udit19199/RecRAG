"""XBRL ontology helpers: a keyword gazetteer and a numeric-expression regex.

The gazetteer is derived from the 139 XBRL concept names (CamelCase split into
keywords). The numeric regex is used both by the ontology/schema method (to find
candidate numeric expressions) and by the fair scoring universe (numeric-only).
"""

from __future__ import annotations

import re

# Financial numeric expressions: optional currency, grouped/decimal digits,
# optional trailing percent. Matches "100", "7.00", "102.25", "$100", "1,234.56",
# "50%". Currency/scale words are usually separate tokens in FiNER.
NUMERIC_REGEX = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")

_DIGIT_RE = re.compile(r"\d")

# Very common English/finance words that carry no discriminating signal as
# ontology keywords (they would match almost every sentence).
_STOPWORDS = {
    "and",
    "or",
    "of",
    "the",
    "to",
    "in",
    "for",
    "per",
    "by",
    "at",
    "on",
    "as",
    "an",
    "a",
}


def _split_camel_case(name: str) -> list[str]:
    """Split a CamelCase XBRL concept into lowercased word tokens."""
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", name)
    return [p.lower() for p in parts]


def build_gazetteer(concept_names: list[str]) -> set[str]:
    """Build a keyword set from XBRL concept names (min length 3, no stopwords)."""
    keywords: set[str] = set()
    for concept in concept_names:
        for word in _split_camel_case(concept):
            if len(word) >= 3 and word not in _STOPWORDS and not word.isdigit():
                keywords.add(word)
    return keywords


def build_gazetteer_regex(gazetteer: set[str]) -> re.Pattern[str]:
    """Compile a word-boundary alternation over the gazetteer keywords."""
    if not gazetteer:
        # Never matches.
        return re.compile(r"(?!x)x")
    alternation = "|".join(re.escape(kw) for kw in sorted(gazetteer, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


def is_numeric(text: str) -> bool:
    """A span counts as numeric (in the scoring universe) if it contains a digit."""
    return bool(_DIGIT_RE.search(text))
