"""BenchmarkSuite generation (D7, D8, D24 — no industry templates v1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from orchestration.models import BenchmarkSuite, RagArchitecture, Requirements

ARCHITECTURE_PROBES: dict[str, list[str]] = {
    "naive": [
        "Summarize the main topic covered in these documents.",
        "What are the key steps described in the procedures?",
    ],
    "citation": [
        "Quote the exact passage that defines the compliance requirement.",
        "Which page and section describe the approval process?",
    ],
    "multimodal": [
        "Describe what the chart or diagram in the document shows.",
        "What data is presented in the visual figure?",
    ],
}


def _load_rules() -> dict[str, Any]:
    path = Path("data/recommendation_rules.toml")
    return toml.load(path) if path.exists() else {}


def _domain_queries(requirements: Requirements, count: int) -> list[str]:
    base = requirements.use_case.strip()
    templates = [
        f"What does the documentation say about {base[:80]}?",
        f"List the requirements related to {base[:60]}.",
        f"How should staff handle scenarios involving {base[:60]}?",
        f"What are the risks or exceptions mentioned for {base[:50]}?",
    ]
    return templates[:count]


def build_benchmark_suite(
    requirements: Requirements,
    architectures: list[RagArchitecture],
) -> BenchmarkSuite:
    rules = _load_rules()
    bench = rules.get("benchmark_suite", {})
    domain_count = int(bench.get("domain_query_count", 4))
    probe_count = int(bench.get("architecture_probe_count", 2))

    probes: dict[str, list[str]] = {}
    for arch in architectures:
        key = arch.value
        pool = ARCHITECTURE_PROBES.get(key, ARCHITECTURE_PROBES["naive"])
        probes[key] = pool[:probe_count]

    return BenchmarkSuite(
        domain_queries=_domain_queries(requirements, domain_count),
        user_queries=list(requirements.user_queries),
        architecture_probes=probes,
    )
