"""Standalone router comparison: Jev versus GPT."""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient


class RouterChoice(BaseModel):
    candidate: str
    reason: str


@dataclass(slots=True)
class UseCaseProfile:
    id: str
    text: str


CANDIDATES = ("classical_rag", "graphrag_standard", "graphrag_ontology_guided")


def _gpt_choice(profile: UseCaseProfile, model: ChatOpenAI) -> RouterChoice:
    structured = model.with_structured_output(RouterChoice, method="function_calling")
    return structured.invoke(
        "Pick one candidate for this use-case profile and give a short reason. "
        f"Candidates: {', '.join(CANDIDATES)}.\n\n{profile.text}"
    )


def run_gpt_router(
    profiles: list[UseCaseProfile],
    model: ChatOpenAI,
) -> list[dict[str, object]]:
    results = []
    for profile in profiles:
        started = perf_counter()
        choice = _gpt_choice(profile, model)
        results.append(
            {
                "profile": profile.id,
                "router": "gpt",
                "candidate": choice.candidate,
                "reason": choice.reason,
                "latency_seconds": perf_counter() - started,
            }
        )
    return results


def run_jev_router(
    profiles: list[UseCaseProfile],
    client: TypeSafeClient | None = None,
) -> list[dict[str, object]]:
    client = client or TypeSafeClient(api_key=os.environ.get("TYPESAFE_API_KEY"))
    results = []
    for profile in profiles:
        started = perf_counter()
        response = client.system_one(
            state=profile.text,
            questions={
                "candidate": Choice(
                    instructions="Pick one candidate for this use-case profile",
                    criteria={
                        "classical_rag": "Passage retrieval without a graph",
                        "graphrag_standard": "Open-ended graph extraction",
                        "graphrag_ontology_guided": "Fixed type graph extraction",
                    },
                ),
                "multi_hop": Score(
                    instructions="How multi-hop is the question pattern",
                    criteria=[
                        "Single fact",
                        "Two linked facts",
                        "Many linked facts",
                    ],
                ),
                "citations_required": Noul(
                    instructions="The profile requires evidence or citations"
                ),
            },
        )
        answers = response.answers
        choice = answers["candidate"]
        results.append(
            {
                "profile": profile.id,
                "router": "jev",
                "candidate": choice.choice,
                "probabilities": dict(choice.probabilities),
                "confidence": choice.confidence,
                "multi_hop_score": answers["multi_hop"].score,
                "citations_required": answers["citations_required"].noul,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "latency_seconds": perf_counter() - started,
            }
        )
    return results


__all__ = ["CANDIDATES", "UseCaseProfile", "run_gpt_router", "run_jev_router"]
