from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


@dataclass(slots=True, frozen=True)
class TokenCounts:
    input_tokens: int
    output_tokens: int


@dataclass(slots=True)
class TokenLedger(BaseCallbackHandler):
    input_price_per_million: float = 0.20
    output_price_per_million: float = 1.20
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = self._usage(response)
        if usage is None:
            return
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.calls += 1

    @staticmethod
    def _usage(response: LLMResult) -> TokenCounts | None:
        if not response.generations or not response.generations[0]:
            return None
        message = getattr(response.generations[0][0], "message", None)
        usage = getattr(message, "usage_metadata", None) or {}
        if not usage:
            output = response.llm_output or {}
            usage = output.get("token_usage", output.get("usage", {}))
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        if input_tokens is None or output_tokens is None:
            return None
        return TokenCounts(int(input_tokens), int(output_tokens))

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.input_tokens * self.input_price_per_million
            + self.output_tokens * self.output_price_per_million
        ) / 1_000_000
