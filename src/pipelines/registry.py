"""RAG architecture registry (phase 5)."""

from __future__ import annotations

from orchestration.models import RagArchitecture

# D15 — graph/agentic commented until implemented
ACTIVE_ARCHITECTURES: frozenset[RagArchitecture] = frozenset(
    {
        RagArchitecture.NAIVE,
        RagArchitecture.CITATION,
        RagArchitecture.MULTIMODAL,
    }
)


def is_active(architecture: RagArchitecture) -> bool:
    return architecture in ACTIVE_ARCHITECTURES
