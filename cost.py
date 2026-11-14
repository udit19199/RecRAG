"""API usage per question and per index build.

Question code runs inside measure(); index builders run inside indexing(), which
routes their calls to the build meter of an enclosing metering_builds() (or drops
them when none is active). Every call records its wall time, and failed calls are
counted, because a timed-out request can still be billed and always costs time."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
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

    def list_usd(self, call: APICall) -> float | None:
        """Price at the uncached list rate, so cost does not depend on run order."""
        if call.kind == "embeddings" and call.model == self.embedding_model:
            return call.input_tokens * self.embedding_per_million / 1_000_000
        if call.kind == "responses" and call.model == self.llm_model:
            return (
                call.input_tokens * self.input_per_million
                + call.output_tokens * self.output_per_million
            ) / 1_000_000
        return None


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
    seconds: float = 0.0


class FailedCall(BaseModel):
    kind: str
    status: int | None
    error: str
    seconds: float


class QuestionCost(BaseModel):
    calls: list[APICall] = Field(default_factory=list)
    missing_usage_calls: int = 0
    failed_calls: list[FailedCall] = Field(default_factory=list)

    def list_usd(self, pricing: Pricing) -> float | None:
        prices = [pricing.list_usd(call) for call in self.calls]
        if self.missing_usage_calls or any(price is None for price in prices):
            return None
        return sum(price or 0 for price in prices)

    @property
    def model_seconds(self) -> float:
        return sum(call.seconds for call in self.calls) + sum(
            call.seconds for call in self.failed_calls
        )

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
_build_meter: ContextVar[Meter | None] = ContextVar("build_cost", default=None)


@contextmanager
def measure(pricing: Pricing):
    meter = Meter(pricing=pricing)
    token = _meter.set(meter)
    try:
        yield meter.cost
    finally:
        _meter.reset(token)


@contextmanager
def metering_builds(pricing: Pricing):
    """Collect every index-build call made inside indexing() blocks."""
    meter = Meter(pricing=pricing)
    token = _build_meter.set(meter)
    try:
        yield meter.cost
    finally:
        _build_meter.reset(token)


# The build meter of the index build running in this process, for calls made
# from worker threads, which do not inherit context variables (HippoRAG runs
# its OpenIE calls in a ThreadPoolExecutor, so its build calls went unmetered).
_active_build: Meter | None = None


def _current() -> Meter | None:
    return _meter.get() or _active_build


@contextmanager
def indexing():
    global _active_build
    meter = _build_meter.get()
    token = _meter.set(meter)
    outer = _active_build
    _active_build = meter
    try:
        yield
    finally:
        _active_build = outer
        _meter.reset(token)


@contextmanager
def unmetered():
    """Warm-up work that belongs to neither a question nor a build."""
    token = _meter.set(None)
    try:
        yield
    finally:
        _meter.reset(token)


def _kind(request) -> str:
    return request.url.path.rstrip("/").rsplit("/", 1)[-1]


def record_failure(request, status: int | None, error: str, seconds: float) -> None:
    meter = _current()
    if meter is None or _kind(request) not in {"responses", "embeddings"}:
        return
    meter.cost.failed_calls.append(
        FailedCall(
            kind=_kind(request), status=status, error=error[:300], seconds=seconds
        )
    )


def record_response(response, seconds: float = 0.0) -> None:
    meter = _current()
    if meter is None:
        return
    if not response.is_success:
        record_failure(response.request, response.status_code, response.text, seconds)
        return
    kind = _kind(response.request)
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
                seconds=seconds,
            )
        )
    except (ValueError, TypeError, AttributeError):
        meter.cost.missing_usage_calls += 1


# Index builds can ask OpenAI for a cheaper service tier (13 Nov 2026, final
# 500-question study). RECRAG_BUILD_SERVICE_TIER=flex adds service_tier to each
# build's Responses request body and nothing else: same prompt, model,
# reasoning effort and settings, so the build's caches and index keys are
# unchanged. Flex is billed at the Batch API rate but answers synchronously,
# so the build code runs as before. Question-time calls are never changed.
# A flex request the API cannot serve right now (429) was not processed; it is
# retried with backoff, then sent once at the default tier.
FLEX_ATTEMPTS = 5
# Flex requests can wait in OpenAI's queue far longer than the 120 s config
# timeout (13 Nov: builds stopped on timeouts), so a flex build request gets
# FLEX_TIMEOUT seconds to answer. A request at the default tier keeps its own.
FLEX_TIMEOUT = 900.0


def _build_tier(request) -> str | None:
    tier = os.environ.get("RECRAG_BUILD_SERVICE_TIER")
    meter = _current()
    if (
        not tier
        or meter is None
        or meter is not (_build_meter.get() or _active_build)
        or request.method != "POST"
        or _kind(request) != "responses"
    ):
        return None
    return tier


def _with_tier(request, tier: str | None):
    if tier is None:
        return request
    body = json.loads(request.content)
    body["service_tier"] = tier
    content = json.dumps(body).encode()
    headers = request.headers.copy()
    headers["content-length"] = str(len(content))
    extensions = dict(request.extensions)
    if tier != "default":
        extensions["timeout"] = {
            "connect": 30.0,
            "read": FLEX_TIMEOUT,
            "write": 60.0,
            "pool": FLEX_TIMEOUT,
        }
    # The same request class as the caller's: HippoRAG's environment sends
    # through a fork of httpx whose client rejects httpx.Request.
    return type(request)(
        request.method,
        request.url,
        headers=headers,
        content=content,
        extensions=extensions,
    )


def _tier_plan(request) -> list[str | None]:
    tier = _build_tier(request)
    if tier is None:
        return [None]
    plan: list[str | None] = [tier] * FLEX_ATTEMPTS
    return [*plan, "default"]


class MeteredClient(DefaultHttpxClient):
    def send(self, request, **kwargs):
        plan = _tier_plan(request)
        for attempt, tier in enumerate(plan):
            started = perf_counter()
            sent = _with_tier(request, tier)
            try:
                response = super().send(sent, **kwargs)
                if _current() is not None:
                    response.read()
            except Exception as exc:
                record_failure(sent, None, repr(exc), perf_counter() - started)
                raise
            if response.status_code == 429 and attempt + 1 < len(plan):
                record_failure(sent, 429, response.text, perf_counter() - started)
                response.close()
                time.sleep(2 ** (attempt + 1))
                continue
            record_response(response, perf_counter() - started)
            return response
        raise RuntimeError("unreachable")


class MeteredAsyncClient(DefaultAsyncHttpxClient):
    async def send(self, request, **kwargs):
        plan = _tier_plan(request)
        for attempt, tier in enumerate(plan):
            started = perf_counter()
            sent = _with_tier(request, tier)
            try:
                response = await super().send(sent, **kwargs)
                if _current() is not None:
                    await response.aread()
            except Exception as exc:
                record_failure(sent, None, repr(exc), perf_counter() - started)
                raise
            if response.status_code == 429 and attempt + 1 < len(plan):
                record_failure(sent, 429, response.text, perf_counter() - started)
                await response.aclose()
                await asyncio.sleep(2 ** (attempt + 1))
                continue
            record_response(response, perf_counter() - started)
            return response
        raise RuntimeError("unreachable")
