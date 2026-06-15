"""Bind tiers to live provider models via price ranking (D25, D26)."""

from __future__ import annotations

import logging
from typing import Literal

from orchestration.models import AdapterRef, ModelTier
from orchestration.pricing import rank_models_by_price

logger = logging.getLogger(__name__)

Category = Literal["embedding", "llm", "vision"]


def _pick_tier(ranked: list[tuple[str, float]], tier: ModelTier) -> str | None:
    if not ranked:
        return None
    priced = [(m, p) for m, p in ranked if p != float("inf")]
    if not priced:
        return ranked[0][0]
    if tier == ModelTier.FLAGSHIP:
        return priced[0][0]
    if tier == ModelTier.ECONOMY:
        return priced[-1][0]
    # mid — median by price (TC-5: one step below flagship when 3+ models)
    mid_idx = len(priced) // 2
    return priced[mid_idx][0]


def resolve_adapter(
    providers: dict[str, list[str]],
    provider: str,
    category: Category,
    tier: ModelTier,
) -> AdapterRef | None:
    models = providers.get(provider, [])
    if not models:
        logger.warning("No models for provider %s", provider)
        return None
    kind = "input" if category == "embedding" else "input"
    ranked = rank_models_by_price(provider, models, kind)
    model = _pick_tier(ranked, tier)
    if not model:
        return None
    return AdapterRef(provider=provider, model=model)


def build_provider_model_map(
    embedders: dict[str, object],
    llms: dict[str, object],
    vision: dict[str, object] | None = None,
) -> dict[str, list[str]]:
    """Flatten ProvidersResponse-like dicts to provider -> model names."""
    result: dict[str, list[str]] = {}
    for name, info in embedders.items():
        models = getattr(info, "models", None) or info.get("models", [])  # type: ignore[union-attr]
        if getattr(info, "available", True) or info.get("available", True):  # type: ignore[union-attr]
            result.setdefault(name, []).extend(models)
    for name, info in llms.items():
        models = getattr(info, "models", None) or info.get("models", [])  # type: ignore[union-attr]
        if getattr(info, "available", True) or info.get("available", True):  # type: ignore[union-attr]
            if name not in result:
                result[name] = []
            result[name].extend(models)
    if vision:
        for name, info in vision.items():
            models = getattr(info, "models", None) or info.get("models", [])  # type: ignore[union-attr]
            if getattr(info, "available", True) or info.get("available", True):  # type: ignore[union-attr]
                result.setdefault(f"vision_{name}", []).extend(models)
    return result
