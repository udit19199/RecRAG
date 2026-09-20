# Router measurement

The router must choose from measured candidate facts. This page defines the
measurements, baselines, first comparison, and saved records.

## Hypotheses

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
- A useful architecture reaches the same quality with fewer total model tokens,
  especially on a token-hungry model.

## Measurement rules

Quality is the primary metric. Use the other metrics when quality is close or a
customer has a hard limit.

Do not use API price as an experiment metric. Record model token use and raw
resource use first. Apply current prices later as a deployment estimate.

Compare the same candidate configurations with both a token-efficient model and
a token-hungry model. This separates a model difference from an architecture
difference.

### Quality

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

### Token and agent work

Record input, output, reasoning, and tool-call tokens separately for
construction, updates, retrieval, agent calls, and answer generation. Also
record:

- total model tokens per question;
- total tokens per correct, evidence-supported answer;
- model calls per question;
- retrieval tool calls per question;
- calls before the first useful evidence;
- retries, failed tool calls, and repeated searches;
- questions solved without another tool call;
- the agent stop reason.

The key comparison is:

```text
tokens per supported answer = total model tokens / supported answers
```

Keep evaluation judge tokens separate from production query tokens.

## Resource cost

Use a fixed time horizon such as one month or one year:

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

The experiment reports token and resource use. A deployment estimate can apply
the current price of the selected model and hosting option afterward.

### Latency, storage, and reliability

Record construction time, update time, retrieval p50 and p95, end-to-end
answer p50 and p95, cold and warm query time, agent-call time, and the fraction
of the corpus rebuilt for an update.

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

## Router baselines

Run every baseline on the same profiles and candidate facts:

- **Fixed choice:** always returns the same candidate.
- **Rule-based choice:** applies the written filters and comparison table.
- **GPT choice:** asks `gpt-5.6-luna` for a structured profile or candidate choice.
- **Jev choice:** asks Jev typed questions and applies the filters in code.

The GPT and Jev calls must see the same profile and candidate measurements. The
answer model stays fixed within each candidate comparison.

## First comparison

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

## Evaluate the router

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

## Save the experiment records

Save enough data to reproduce the recommendation and update it later.

### Use-case record

- profile identifier and full profile state;
- corpus identifier and version;
- question class and question identifier;
- requirements, preferences, tie range, and threshold constants;
- missing fields and uncertainty.

### Candidate record

- candidate identifier;
- architecture, construction, and retrieval choices;
- model and embedding versions;
- prompt, chunking, ontology, index, and timeout versions;
- code revision and agent maximum call count.

### Build and update record

- run identifier and corpus version;
- status and failure reason;
- construction and update time;
- input, output, reasoning, and extraction tokens;
- embedding count and storage use;
- chunk, node, relationship, and index counts;
- retry count and changed corpus fraction.

### Query record

- question and candidate identifiers;
- retrieval and end-to-end latency;
- input, output, reasoning, and tool-call tokens;
- model call, tool call, and retry counts;
- retrieved item count and empty-context flag;
- correctness, faithfulness, retrieval precision, and retrieval recall;
- citation coverage, abstention result, and failure category.

### Router decision record

- decision identifier and router version;
- eligible candidates and hard-filter results;
- selected candidate and decision reason;
- candidate-measurement version;
- best eligible candidate and regret;
- repeatability result and rollback or fallback result.

Keep thresholds and question text in one reviewed location. Changing either one
changes the experiment protocol.

## Current evidence and limits

The [five-record 2WikiMultiHopQA result](../graphrag/2wikimultihopqa-results.md)
is a candidate-measurement source. It does not establish a universal ranking.
Ontology-guided construction scored higher on graph-quality measures, while
standard construction with vector retrieval had the strongest recorded answer
measures.

The current evaluations do not measure exact graph triple recall, exact
multi-hop path accuracy, latency, token use, or agent tool choice.
