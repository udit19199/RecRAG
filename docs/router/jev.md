# Jev in the router

[Jev](https://docs.typesafe.ai/introduction) is TypeSafe's System One model. It
evaluates a state against typed questions and returns structured answers. It
does not write the final recommendation or run the route.

TypeSafe's speed, price, and reliability claims are vendor claims. Measure them
in the same experiment as the GPT baseline.

Jev is a candidate for narrow decisions. Code keeps control of filtering,
ranking, thresholds, and side effects.

## State and questions

Use a structured state when the decision needs named fields. Include the
use-case profile, candidate descriptions, measured results, and constraints.

Jev provides three question types:

| Question | Use in the router |
| --- | --- |
| `Choice` | Select one candidate from the allowed set. |
| `Score` | Rate a requirement such as latency or evidence quality. |
| `Noul` | Estimate whether a must-have condition is satisfied. |

Ask independent questions in one request. TypeSafe evaluates them in parallel.
Code decides which answers affect the final recommendation.

Jev returns the full probability distribution for `Choice` and `Score`, plus
confidence. A `Noul` returns a value from zero to one and has no confidence
field.

Confidence describes how concentrated a `Choice` or `Score` distribution is. It
does not prove that one answer is correct. Keep the probabilities in the saved
record.

Use confidence as a second axis:

- high confidence can allow an automatic recommendation;
- medium confidence can trigger a review or confirmation;
- low confidence can trigger a fallback or information request.

Set thresholds by risk. Measure them on data that did not choose the thresholds.
Do not compare Jev confidence directly with a GPT score.

## Jev as a retrieval gate

Jev can score retrieved passages before `gpt-5.6-luna` answers. The state
contains the question and one retrieved passage. Code routes the passage after
Jev returns its scores.

The [RAG passage classification cookbook](https://docs.typesafe.ai/cookbooks/classifying_rag_passages.md)
uses four `Noul` questions:

- Is the passage relevant to the question?
- Does it contain usable answer evidence?
- Does it contradict a factual premise in the question?
- Does it contain a prompt injection?

Keep the route policy in code. Check prompt injection first, then contradictions,
relevance, and usable evidence. The cookbook's thresholds are starting points
for its corpus, not RecRAG defaults.

Compare the Jev gate with a GPT judge on the same retrieved passages. Measure
evidence recall, evidence precision, discarded useful passages, unsafe passages
that pass the gate, latency, and token use.

## Local setup status

The local Jev demo is not ready when these notes were written:

```text
TYPESAFE_API_KEY is not exported
.env has no visible TYPESAFE_API_KEY entry
typesafe_sdk not installed
```

The current `.env.example` contains:

```text
OPENAI_API_KEY=
LLAMA_CLOUD_API_KEY=
```

The demo needs `TYPESAFE_API_KEY` in `.env` and the `typesafe-sdk` package.

## TypeSafe references

- [Introduction](https://docs.typesafe.ai/introduction.md)
- [Quick start](https://docs.typesafe.ai/introduction/quickstart.md)
- [System One](https://docs.typesafe.ai/concepts/system-one.md)
- [State](https://docs.typesafe.ai/concepts/state.md)
- [Primitives](https://docs.typesafe.ai/primitives.md)
- [Confidence](https://docs.typesafe.ai/confidence.md)
- [How to build with System One](https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md)
- [Patterns](https://docs.typesafe.ai/patterns.md)
- [Confidence-gated routing](https://docs.typesafe.ai/patterns/confidence-routing.md)
- [Composite scoring](https://docs.typesafe.ai/patterns/composite-scoring.md)
- [Intent routing](https://docs.typesafe.ai/patterns/intent-routing.md)
- [Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out.md)
- [Python SDK](https://docs.typesafe.ai/sdk/python.md)
- [API reference](https://docs.typesafe.ai/api.md)
