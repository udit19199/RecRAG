"""Hard constraint filters for pipeline shortlisting (D16)."""

from __future__ import annotations

from orchestration.models import DocumentModality, RagArchitecture, Requirements


def eligible_architectures(requirements: Requirements) -> list[RagArchitecture]:
    """Return architectures that pass hard gates — no model IDs (D16)."""
    allowed: set[RagArchitecture] = {
        RagArchitecture.NAIVE,
        RagArchitecture.CITATION,
        RagArchitecture.MULTIMODAL,
    }

    if requirements.document_modality in (
        DocumentModality.IMAGE_HEAVY,
        DocumentModality.MIXED,
    ):
        allowed.add(RagArchitecture.MULTIMODAL)
    else:
        # Text-heavy paths don't require multimodal-only stack
        pass

    if requirements.citations_required:
        allowed.discard(RagArchitecture.NAIVE)
        allowed.add(RagArchitecture.CITATION)

    if requirements.document_modality == DocumentModality.VIDEO:
        # No video loader v1 — only naive/citation on transcripts if provided
        allowed.discard(RagArchitecture.MULTIMODAL)

    return sorted(allowed, key=lambda a: a.value)


def metric_weight_key(requirements: Requirements) -> str:
    if requirements.citations_required:
        return "citations_required"
    if requirements.latency_sensitive:
        return "latency_sensitive"
    return "default"
