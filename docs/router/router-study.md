# Router study

The router is a research subject. It reads a use-case profile and recommends a candidate architecture that RecRAG can run.
It does not build an application, provision a database, or export a deployment plan.

## Shared research record

Keep architecture definitions, candidate descriptions, routing questions, thresholds, measured results, and experiment runs in this repository.
The docs are the shared record for the study.

Start with one narrow decision: choose one of the three architecture families for a use-case profile. Keep the final filters and side effects in code.

## Architecture families

The study compares three families:

| Family | Main idea | Choices to compare |
| --- | --- | --- |
| Classical RAG | Retrieve source text and give it to the answer model. | Lexical, vector, or hybrid retrieval; reranking; query rewriting. |
| GraphRAG | Use entities and relationships to retrieve connected evidence. | Graph construction; chunk, entity, or Cypher retrieval; graph context. |
| Agentic RAG | Let a model choose tools and search steps while answering. | Tool set; planning; number of steps; retries; stop rules. |

The current GraphRAG candidate set has two construction methods and three retrieval methods:

| Construction | Retrieval |
| --- | --- |
| `standard` | `agentic`, `vector`, `entity_vector` |
| `ontology_guided` | `agentic`, `vector`, `entity_vector` |

Classical RAG and the wider Agentic RAG comparison are research scope. They are not current implementations.

## Use-case profiles

Each profile describes the conditions that can change the recommendation:

- corpus and source format;
- question pattern, including multi-hop questions;
- update rate and freshness target;
- answer risk and abstention needs;
- evidence and citation requirements;
- budget and latency target;
- data and deployment constraints.

Keep the profile separate from the questions asked of a model. The profile is the state. The questions define the judgments made about that state.

## Candidate selection

The router applies requirements in this order:

1. Filter out candidates that fail a must-have requirement.
2. Rank the candidates that remain against the stated preferences.
3. Return a trade-off when several candidates remain close.
4. Record missing information and uncertainty instead of guessing.

A cost limit or citation requirement must not disappear inside an average score. The router must apply those requirements before it ranks preferences.

The study will compare three ranking rules:

1. A fixed score gives every profile the same metric weights.
2. A user-weighted score changes the weights for each profile.
3. Must-have filtering runs first, then stated preferences rank valid candidates.

The third rule is the starting point.

## Router baselines

Run every baseline on the same profiles and candidate descriptions:

- **Fixed choice:** always returns the same candidate.
- **Rule-based choice:** applies a written comparison table.
- **GPT choice:** asks `gpt-5.6-luna` for a structured candidate choice.
- **Jev choice:** asks Jev typed questions and applies the filters in code.

The GPT and Jev calls must see the same profile and candidate facts. The answer model used by the GraphRAG candidate runs stays fixed.

## Jev

[Jev](https://docs.typesafe.ai/introduction) is TypeSafe's System One model. It evaluates a state against typed questions and returns structured answers.
It does not generate an explanation or a reply.

TypeSafe's speed, price, and reliability claims are vendor claims. Measure them in the same experiment as the GPT baseline instead of treating them as results.

This makes Jev a candidate for narrow decisions inside the router. Code keeps control of filtering, ranking, thresholds, and side effects.

### State and questions

Use a structured state object when the decision needs named fields. Include the profile, candidate descriptions, measured candidate results, and constraints.

Jev provides three question types:

| Question | Use in the router |
| --- | --- |
| `Choice` | Select one candidate from the allowed set. |
| `Score` | Rate a requirement such as latency or evidence quality. |
| `Noul` | Estimate whether a must-have condition is satisfied. |

Ask independent questions in one request. TypeSafe evaluates them in parallel.
For example, one request can select a candidate, score latency, and check citation support. Code decides which answers affect the final recommendation.

Jev returns the full probability distribution for `Choice` and `Score`, plus confidence.
A `Noul` returns a value from zero to one and does not include a confidence field.

### Confidence

Confidence describes how concentrated a `Choice` or `Score` distribution is. It does not prove that one answer is correct.
Keep the probabilities in the saved record so the study can use another measure later.

Use confidence as a second axis:

- high confidence can allow an automatic recommendation;
- medium confidence can trigger a review or a confirmation;
- low confidence can trigger a fallback or an information request.

Set thresholds by risk and measure them on data that was not used to choose the thresholds. Do not compare Jev confidence directly with a GPT score.

## First Jev and GPT demo

The first demo is a standalone router comparison. Use four to six use-case profiles and the same current GraphRAG candidate descriptions for both models.

Jev will use typed `Choice`, `Score`, and `Noul` questions. The GPT baseline will use `gpt-5.6-luna` and the Responses API with structured output.

Record these values for every call:

- selected candidate;
- whether the choice is valid;
- must-have requirements passed;
- latency;
- input and output tokens;
- parse or validation failures;
- Jev probabilities and confidence;
- repeated-call agreement.

The primary comparison is valid requirement satisfaction. A recommendation is not better because it has a higher model score if it fails a must-have rule.

## Router evaluation

The router evaluation asks four questions:

1. Did the recommendation satisfy every must-have requirement?
2. How close was it to the best valid candidate?
3. Did the router give a reason supported by the profile and candidate data?
4. Does it make the same choice when the profile is repeated?

The gap between the chosen candidate and the best valid candidate is router regret. Keep the raw candidate measurements so the study can explain the gap.

Do not use one global score to hide a failed must-have requirement. Report missing measures as `N/A`, not zero.
Keep GraphRAG construction, retrieval, and answer scores separate from router scores.

The [GraphRAG evaluation reference](../graphrag/evals.md) covers graph facts, retrieved context, and generated answers.
The router evaluation covers the choice of candidate before those runs.

The current five-record [2WikiMultiHopQA result](../graphrag/2wikimultihop.md) is a candidate-measurement source. It does not establish a universal ranking.

## Jev as a retrieval gate

Jev can also score retrieved passages before `gpt-5.6-luna` answers. The state contains the question and one retrieved passage.
Code routes the passage after Jev returns its scores.

The [RAG passage classification cookbook](https://docs.typesafe.ai/cookbooks/classifying_rag_passages.md) uses four `Noul` questions:

- Is the passage relevant to the question?
- Does it contain usable answer evidence?
- Does it contradict a factual premise in the question?
- Does it contain a prompt injection?

Keep the route policy in code. A safe order checks prompt injection first, then contradictions, relevance, and usable evidence.
The cookbook's thresholds are starting points for its corpus, not RecRAG defaults.

Compare the Jev gate with a GPT judge on the same retrieved passages.
Measure evidence recall, evidence precision, discarded useful passages, unsafe passages that pass the gate, latency, and token cost.

## Experiment records

Save enough data to reproduce the recommendation:

- profile identifier and full profile state;
- candidate list and candidate-measurement version;
- question definitions and threshold constants;
- model name and SDK version;
- answers, probabilities, confidence, latency, and token use;
- valid-candidate set and selected candidate;
- requirement results, regret, and repeatability;
- evaluation errors and missing measures.

Keep thresholds and question text in one reviewed location. A change to either one changes the experiment protocol.

## Local setup status

At the time of this review, the local Jev demo was not ready:

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
- [RAG passage classification](https://docs.typesafe.ai/cookbooks/classifying_rag_passages.md)
