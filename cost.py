"""Question-time API usage. Index builders suspend this meter explicitly."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace

from openai import DefaultAsyncHttpxClient, DefaultHttpxClient
from pydantic import BaseModel, Field


class Pricing(BaseModel):
    # Standard direct-API USD rates, verified 2026-10-04. Snapshot with each run.
    source: str = "https://developers.openai.com/api/docs/pricing"
    verified_date: str = "2026-10-04"
    llm_model: str = "gpt-6-luna"
    input_per_million: float = Field(default=0.10, ge=0)
    cached_input_per_million: float = Field(default=0.01, ge=0)
    cache_write_per_million: float = Field(default=0.125, ge=0)
    output_per_million: float = Field(default=0.50, ge=0)
    embedding_model: str = "text-embedding-3-small"
    embedding_per_million: float = Field(default=0.02, ge=0)


class APICall(BaseModel):
    model: str
    kind: str
    input_tokens: int
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    service_tier: str = "default"
    estimated_usd: float | None


class QuestionCost(BaseModel):
    calls: list[APICall] = Field(default_factory=list)
    missing_usage_calls: int = 0

    @property
    def estimated_usd(self) -> float | None:
        if self.missing_usage_calls or any(
            call.estimated_usd is None for call in self.calls
        ):
            return None
        return sum(call.estimated_usd or 0 for call in self.calls)


class Meter(BaseModel):
    pricing: Pricing
    cost: QuestionCost = Field(default_factory=QuestionCost)


_meter: ContextVar[Meter | None] = ContextVar("question_cost", default=None)


@contextmanager
def measure(pricing: Pricing):
    meter = Meter(pricing=pricing)
    token = _meter.set(meter)
    try:
        yield meter.cost
    finally:
        _meter.reset(token)


@contextmanager
def indexing():
    token = _meter.set(None)
    try:
        yield
    finally:
        _meter.reset(token)


def record_response(response) -> None:
    meter = _meter.get()
    if meter is None or not response.is_success:
        return
    kind = response.request.url.path.rstrip("/").rsplit("/", 1)[-1]
    if kind not in {"responses", "embeddings"}:
        return
    try:
        payload = json.loads(
            response.content, object_hook=lambda fields: SimpleNamespace(**fields)
        )
        usage = getattr(payload, "usage", None)
        if usage is None:
            meter.cost.missing_usage_calls += 1
            return
        price = meter.pricing
        request = json.loads(
            response.request.content,
            object_hook=lambda fields: SimpleNamespace(**fields),
        )
        model = getattr(request, "model", "unknown")
        input_tokens = getattr(
            usage, "input_tokens", getattr(usage, "prompt_tokens", 0)
        )
        output_tokens = getattr(usage, "output_tokens", 0)
        details = getattr(usage, "input_tokens_details", SimpleNamespace())
        cached = getattr(details, "cached_tokens", 0) or 0
        writes = (
            getattr(
                details,
                "cache_write_tokens",
                getattr(details, "cache_creation_tokens", 0),
            )
            or 0
        )
        reasoning = (
            getattr(
                getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0
            )
            or 0
        )
        tier = getattr(payload, "service_tier", "default") or "default"
        usd = None
        if kind == "embeddings" and model == price.embedding_model:
            usd = input_tokens * price.embedding_per_million / 1_000_000
        elif kind == "responses" and model == price.llm_model:
            long = input_tokens > 272_000
            tier_factor = (
                0.5
                if tier in {"flex", "batch"}
                else 2
                if tier in {"priority", "fast"}
                else 1
            )
            usd = (
                tier_factor
                * (
                    (2 if long else 1)
                    * (
                        max(0, input_tokens - cached - writes) * price.input_per_million
                        + cached * price.cached_input_per_million
                        + writes * price.cache_write_per_million
                    )
                    + (1.5 if long else 1) * output_tokens * price.output_per_million
                )
                / 1_000_000
            )
        meter.cost.calls.append(
            APICall(
                model=model,
                kind=kind,
                input_tokens=input_tokens,
                cached_input_tokens=cached,
                cache_write_tokens=writes,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning,
                service_tier=tier,
                estimated_usd=usd,
            )
        )
    except (ValueError, TypeError, AttributeError):
        meter.cost.missing_usage_calls += 1


class MeteredClient(DefaultHttpxClient):
    def send(self, request, **kwargs):
        response = super().send(request, **kwargs)
        if _meter.get() is not None:
            response.read()
            record_response(response)
        return response


class MeteredAsyncClient(DefaultAsyncHttpxClient):
    async def send(self, request, **kwargs):
        response = await super().send(request, **kwargs)
        if _meter.get() is not None:
            await response.aread()
            record_response(response)
        return response
