# Router

The router answers one question:

> Given a customer's use case, which RAG design should the customer start with?

The router recommends a candidate and explains the trade-offs. It does not
build an application, provision a database, or export a deployment plan.

## Choose a candidate

### Keep the choices separate

A GraphRAG candidate has separate construction and retrieval choices. The router
keeps these choices separate:

| Choice | Options | Question |
| --- | --- | --- |
| Architecture | Classical RAG, GraphRAG | Does the answer need graph joins or paths? |
| Construction | None, `standard`, `ontology_guided` | Is a graph worth building, and how fixed is the domain schema? |
| Retrieval | Fixed vector, entity vector, agentic | What search behavior does the question need? |
| Operations | Freshness, latency, tokens, storage, database limits | Can the customer run and update the candidate? |

The GraphRAG pages describe the active methods. The router applies these choices
to a candidate and may compare them with ClassicalRAG.

The router can compare ClassicalRAG as a candidate, but this repository does
not currently implement a ClassicalRAG module. The current Streamlit flow runs
the active GraphRAG matrix only.

### Make a recommendation

The customer describes a use case. A person or an NLLM maps the description to
a structured profile. The router checks measured candidate facts and recommends
a candidate.

NLLM means a model that reads a use-case profile and produces a structured
recommendation.

The recommendation records:

- the selected architecture, construction, and retrieval choices;
- requirements that passed and failed;
- measurements used for the choice;
- trade-offs;
- missing information;
- uncertainty.

An NLLM may extract the profile and write the explanation. Code applies hard
filters, reads candidate measurements, and controls the final route. This keeps
a model from recommending a candidate that violates a budget or latency limit.

Start with a written rubric. Later, compare a fixed choice, a rule-based choice,
a GPT choice, and a Jev choice on the same profiles.

### Update a recommendation

Use new benchmark runs and observed outcomes to update the recommendation. Do
not let a model rewrite the policy without checks.

The router can:

- update candidate measurements when a run completes;
- detect quality, latency, token, storage, or failure drift;
- mark stale or failing candidates as ineligible;
- rerun a candidate on a fixed holdout set before promotion;
- update thresholds only from results that did not choose those thresholds;
- keep the last verified policy when new data is incomplete;
- record every automatic change with its evidence.

### Describe the use case

The profile is the state used by the router. Keep it separate from the
questions asked of an NLLM or Jev.

Record:

- corpus size, source format, and source count;
- question classes and their traffic share;
- single-hop, multi-hop, entity, relationship, aggregation, and exact-match needs;
- required joins and maximum path length;
- evidence, citation, and abstention requirements;
- answer risk and the cost of a wrong answer;
- update rate and freshness target;
- p50 and p95 latency targets;
- build, update, and per-query resource limits;
- storage, database, network, and deployment constraints;
- whether the domain schema is stable or changing;
- whether query types are predictable or varied.

### Select a candidate

Apply requirements in this order:

1. Filter out candidates that fail a must-have requirement.
2. Rank the candidates that remain against the customer's preferences.
3. Use quality first when the quality difference is meaningful.
4. If quality is within a predeclared tie range, compare token efficiency,
   latency, storage, database use, and reliability.
5. Return a trade-off when several candidates remain close.
6. Record missing information and uncertainty instead of guessing.

Do not hide a failed cost, latency, freshness, or citation requirement inside a
weighted average. Define the quality tie range before the benchmark run.

For eligible candidates, report the Pareto frontier. A candidate is on the
frontier when no other eligible candidate is better on every reported metric.

These are starting routes for the experiment:

| Use-case signal | Candidate route |
| --- | --- |
| Mostly single-hop questions, high freshness, strict latency, or low resource limit | Classical RAG with fixed retrieval |
| Cross-document entity or relationship joins | GraphRAG with fixed retrieval first |
| Entity-centered questions | GraphRAG with `entity_vector` when measured evidence supports it |
| A stable domain schema and a need for consistent graph types | GraphRAG with `ontology_guided` construction |
| An unknown or changing schema | GraphRAG with `standard` construction when graph reasoning is required |
| Mixed query types that need tool or source choice | Add `agentic` retrieval only when its quality gain covers its extra work |

See [Measure candidates](#measure-candidates) for comparison rules and saved
records. See [Use Jev](#use-jev) for Jev-specific decisions.

## Measure candidates

The router must choose from measured candidate facts. This section defines the
measurements, baselines, first comparison, and saved records.

### Hypotheses

These are experiment hypotheses, not conclusions:

- Graph structure helps when answers need entities, relationships, or
  multi-hop evidence across documents.
- Graph structure adds little for simple passage lookup.
- Ontology guidance improves graph consistency and evidence coverage when the
  domain has a stable schema.
- Standard construction helps when the schema is unknown or changes often.
- Agentic retrieval helps when the system must choose between tools or sources.
- Fixed retrieval uses fewer tokens and has more predictable latency when it
  covers the question class.
- Storage, database, and update work may matter more than query tokens as the
  corpus or query volume grows.
- An architecture is useful when it reaches the same quality with fewer total
  model tokens, especially on a token-hungry model.

### Measurement rules

Quality is the primary metric. Use the other metrics when quality is close or
when a customer has a hard limit.

Do not use API price as an experiment metric. Record model token use and raw
resource use first. Apply current prices later as a deployment estimate.

Compare the same candidate configurations with a token-efficient model and a
token-hungry model. This separates model differences from architecture
differences.

#### Quality

| Metric | Why it matters |
| --- | --- |
| Answer correctness | Whether the system answers the question. |
| Evidence-supported correctness | Whether a correct answer has support in retrieved evidence. |
| Retrieval precision | How much retrieved context is useful. |
| Retrieval recall | Whether retrieval found the required evidence. |
| Citation precision and coverage | Whether citations point to useful and complete evidence. |
| Abstention correctness | Whether the system refuses unsupported answers. |
| Empty-context rate | How often retrieval gives the answer model no usable evidence. |
| Failure rate | How often the full request fails to finish. |

For graph questions, also measure exact entity, relationship, and path accuracy.
An LLM judge score cannot prove that a graph path is correct.

For GraphRAG runs, use the [evaluation usage accounting](graphrag/evaluation.md#judge-model-and-usage-accounting)
for per-call tokens, model turns, tool calls, retries, and judge usage. Keep
evaluation judge tokens separate from production query tokens.

### Resource cost

Use a fixed time horizon, such as one month or one year:

```text
total resource cost =
    initial construction
  + updates
  + model tokens
  + database resources
  + storage and backups
  + retries and failures
```

```text
resource cost per question =
    fixed resource cost / expected question count
  + variable resource use for retrieval and answering
```

The experiment reports token and resource use. Apply the current price of the
selected model and hosting option later when you estimate deployment cost.

### Latency, storage, and reliability

Record construction time, update time, retrieval p50 and p95, end-to-end answer
p50 and p95, cold and warm query time, agent-call time, and the fraction of the
corpus rebuilt for an update.

Record raw resource values:

- bytes per source gigabyte;
- chunk, node, relationship, and index counts;
- vector index and graph database size;
- memory and CPU during construction and retrieval;
- database operations per query;
- storage added by an update;
- backup and rebuild size.

Record empty context, retrieval, tool, database, and answer failures. Also
record retry rate, unsupported answers, repeated-run agreement, changes after
updates, stale measurements, and missing measurements.

### Router baselines

Run every baseline on the same profiles and candidate facts:

- **Fixed choice:** always returns the same candidate.
- **Rule-based choice:** applies the written filters and comparison table.
- **GPT choice:** asks `gpt-5.6-luna` for a structured profile or candidate choice.
- **Jev choice:** asks Jev typed questions and applies the filters in code.

Give the GPT and Jev calls the same profile and candidate measurements. Keep the
answer model fixed within each candidate comparison.

### First comparison

Use four to six use-case profiles. Cover different question classes:

- single-hop passage lookup;
- entity-centered questions;
- cross-document multi-hop questions;
- exact relationship or graph-condition questions;
- no-answer and abstention questions;
- citation-sensitive questions;
- freshness-sensitive questions.

Run candidate systems with the same corpus, question set, answer prompt,
embedding configuration, timeout rules, and evaluation rules. Run the matrix
with both model choices.

The first candidate matrix should include:

- Classical RAG with fixed retrieval;
- GraphRAG `standard` with `vector` retrieval;
- GraphRAG `ontology_guided` with `vector` retrieval;
- GraphRAG `standard` with `entity_vector` for entity-heavy questions;
- GraphRAG `standard` with `agentic` retrieval for mixed query types.

Save raw results for every question. Derive averages, p50, p95, rates, and
frontiers from those rows.

### Evaluate the router

Ask:

1. Did the recommendation satisfy every must-have requirement?
2. How close was it to the best eligible candidate?
3. Did the reason use the profile and candidate measurements?
4. Does the router repeat the same choice for the same profile?
5. Did the router use a stale or missing candidate measurement?
6. Did the router detect a candidate that no longer met its requirements?

Report hard-filter violations, the selected candidate, the best eligible
candidate, quality loss, router token and latency overhead, repeatability,
missing-data rate, stale-measurement rate, and rollback or fallback rate.

Keep GraphRAG construction, retrieval, and answer scores separate from router
scores. The router is evaluated before the candidate run.

### Save the experiment records

Save enough data to reproduce the recommendation and update it later.

#### Use-case record

- profile identifier and full profile state;
- corpus identifier and version;
- question class and question identifier;
- requirements, preferences, tie range, and threshold constants;
- missing fields and uncertainty.

#### Candidate record

- candidate identifier;
- architecture, construction, and retrieval choices;
- model and embedding versions;
- prompt, chunking, ontology, index, and timeout versions;
- code revision and agent maximum call count.

#### Build and update record

- run identifier and corpus version;
- status and failure reason;
- construction and update time;
- input, output, reasoning, and extraction tokens;
- embedding count and storage use;
- chunk, node, relationship, and index counts;
- retry count and changed corpus fraction.

#### Query record

- question and candidate identifiers;
- retrieval and end-to-end latency;
- input, output, reasoning, and tool-call tokens;
- model call, tool call, and retry counts;
- retrieved item count and empty-context flag;
- correctness, faithfulness, retrieval precision, and retrieval recall;
- citation coverage, abstention result, and failure category.

#### Router decision record

- decision identifier and router version;
- eligible candidates and hard-filter results;
- selected candidate and decision reason;
- candidate-measurement version;
- best eligible candidate and regret;
- repeatability result and rollback or fallback result.

Keep thresholds and question text in one reviewed location. Changing either one
changes the experiment protocol.

### Current evidence

The [five-record 2WikiMultiHopQA result](graphrag/2wikimultihopqa-results.md)
is one candidate-measurement source. It does not establish a universal ranking.

## Use Jev

[Jev](https://docs.typesafe.ai/introduction) is TypeSafe's System One model. It
evaluates a state against typed questions and returns structured answers. It
does not write the final recommendation or run the route.

TypeSafe's speed, price, and reliability claims are vendor claims. Measure these
values in the same experiment as the GPT baseline.

Jev is a candidate for narrow decisions. Code controls filtering, ranking,
thresholds, and side effects.

### State and questions

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

### Jev as a retrieval gate

Jev can score retrieved passages before `gpt-5.6-luna` answers. The state
contains the question and one retrieved passage. Code routes the passage after
Jev returns its scores.

The [RAG passage classification cookbook](https://docs.typesafe.ai/cookbooks/classifying_rag_passages.md)
uses four `Noul` questions:

- Is the passage relevant to the question?
- Does it contain usable answer evidence?
- Does it contradict a factual premise in the question?
- Does it contain a prompt injection?

Keep the route policy in code. Check prompt injection first. Then check
contradictions, relevance, and usable evidence. The cookbook's thresholds are
starting points for its corpus, not RecRAG defaults.

Compare the Jev gate with a GPT judge on the same retrieved passages. Measure
evidence recall, evidence precision, discarded useful passages, unsafe passages
that pass the gate, latency, and token use.

### Local setup status

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

### TypeSafe references

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
