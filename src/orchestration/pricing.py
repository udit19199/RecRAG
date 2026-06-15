"""Load model pricing from data/model_pricing.toml (D26)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import toml


@lru_cache(maxsize=1)
def load_pricing(path: Path | None = None) -> dict[str, dict[str, dict[str, float]]]:
    pricing_path = path or Path("data/model_pricing.toml")
    if not pricing_path.exists():
        return {}
    return toml.load(pricing_path)


def model_unit_price(provider: str, model: str, kind: str = "input") -> float | None:
    """Return USD per 1M tokens for input or output."""
    data = load_pricing()
    provider_key = provider.lower()
    models = data.get(provider_key, {})
    entry = models.get(model)
    if not entry:
        return None
    key = "input_per_million" if kind == "input" else "output_per_million"
    return float(entry.get(key, 0.0))


def rank_models_by_price(
    provider: str, models: list[str], kind: str = "input"
) -> list[tuple[str, float]]:
    """Rank models; unknown price sorts last with inf."""
    ranked: list[tuple[str, float]] = []
    for model in models:
        price = model_unit_price(provider, model, kind)
        ranked.append((model, price if price is not None else float("inf")))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
