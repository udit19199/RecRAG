# Router decision

The router answers one question:

> Given a customer's use case, which RAG design should the customer start with?

The router recommends a candidate and explains the trade-offs. It does not
build an application, provision a database, or export a deployment plan.

## Keep the choices separate

`GraphRAG` does not describe the whole candidate. The router keeps these
choices separate:

| Choice | Options | Question |
| --- | --- | --- |
| Architecture | Classical RAG, GraphRAG | Does the answer need graph joins or paths? |
| Construction | None, `standard`, `ontology_guided` | Is a graph worth building, and how fixed is the domain schema? |
| Retrieval | Fixed vector, entity vector, agentic | What search behavior does the question need? |
| Operations | Freshness, latency, tokens, storage, database limits | Can the customer run and update the candidate? |

The current GraphRAG candidate set is:

| Construction | Retrieval |
| --- | --- |
| `standard` | `agentic`, `vector`, `entity_vector` |
| `ontology_guided` | `agentic`, `vector`, `entity_vector` |

The `agentic` method is a GraphRAG retrieval method. It chooses between vector
search and Cypher search. A wider Agentic RAG comparison is future research.

The repository also contains a `ClassicalRAG` module. The current Streamlit
flow does not run it with the GraphRAG matrix. The router treats it as a
comparison candidate, not as part of the active GraphRAG path.

## Two stages

### Stage 1: make a recommendation

Stage 1 makes the research useful before the router can learn from production.
The customer describes a use case. A person or an NLLM maps that description to
a structured profile, checks measured candidate facts, and recommends a
candidate.

NLLM means a natural-language language model that reads a use-case profile and
produces a structured recommendation.

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

The first version can be a written rubric. Later versions can compare a fixed
choice, a rule-based choice, a GPT choice, and a Jev choice on the same
profiles.

### Stage 2: update the recommendation

Stage 2 uses new benchmark runs and observed outcomes. It does not let a model
rewrite the policy without checks.

The router can:

- update candidate measurements when a run completes;
- detect quality, latency, token, storage, or failure drift;
- mark stale or failing candidates as ineligible;
- rerun a candidate on a fixed holdout set before promotion;
- update thresholds only from results that did not choose those thresholds;
- keep the last verified policy when new data is incomplete;
- record every automatic change with its evidence.

## Describe the use case

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

## Select a candidate

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

See [router measurement](measurement.md) for comparison rules and saved
records. See [Jev in the router](jev.md) for Jev-specific decisions.
