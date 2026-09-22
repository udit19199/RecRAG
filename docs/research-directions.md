# Research directions

This note describes two paper ideas for RecRAG. It defines the main terms so
that the note can stand on its own.

The current five-record experiment is too small to rank the methods. It gives
us questions to test, not general conclusions about graph quality or answer
quality.

## Background terms

**Retrieval-augmented generation (RAG)** retrieves text before a language
model writes an answer. The retrieved text gives the model facts that may not
be in its parameters.

**GraphRAG** adds a graph to this process. A graph has nodes, such as people or
events, and edges, such as `works_for` or `caused`. Graph retrieval selects
nodes, edges, paths, or nearby source passages before the language model writes
the answer.

**Graph construction** turns source text into nodes and edges. RecRAG has two
construction choices:

- Standard construction lets the model extract the types and relations it
  finds in the text.
- Ontology-guided construction gives the model a set of expected node and
  relation types. The model can still add types when the schema allows it.

**Retrieval** selects evidence for a question. Vector retrieval selects text
or graph items by embedding similarity. Entity-vector retrieval finds chunks
linked to matching entities without adding graph context. Path retrieval selects a chain
of connected edges. Hybrid retrieval combines more than one of these methods.

In this note, evidence is **safe to use** when it supports the question, does
not contradict the source, and does not contain instructions that try to change
the model's task.

## RLCD-based graph-evidence decisions

### Research question

Can an RLCD-based typed decision model improve evidence control in GraphRAG?

### What RLCD and typed decisions mean

TypeSafe calls its training method **Reinforcement Learning for Calibrated
Decisions (RLCD)**. TypeSafe presents RLCD as a way to train models for
decisions with useful probabilities, rather than for free-form chat responses.
The TypeSafe description contrasts this with reinforcement learning from human
feedback, which mainly rewards answers that people prefer.

Jev is TypeSafe's first public model in this model class. It accepts a state,
such as a question and retrieved evidence, and returns answers in a fixed
schema. The fixed schema makes the result easier for code to use.

Jev supports three decision types:

- `Choice` selects one item from a fixed list, such as `supported`,
  `unsupported`, or `contradictory`.
- `Score` returns a number on a defined scale, such as a path-completeness
  score from 0 to 1.
- `Noul` returns a probability for a yes-or-no question, such as whether the
  evidence supports the answer.

The word **calibration** means that a model's confidence matches its actual
accuracy. For example, among decisions with 80% confidence, about 80% should
be correct. The experiment must measure this property. It must not assume the
vendor's claim that Jev is calibrated.

Jev does not write the final answer in this study. It judges or filters the
evidence. `gpt-5.6-luna` remains the final answer model.

See the [TypeSafe announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
and the [Jev developer guide](https://www.jevtypesafe.org/docs/jev-sdk/).
Jev is currently a hosted API with early-access requirements, so access and
network cost are practical limits for the experiment.

### Hypothesis

An RLCD-based typed decision model can judge graph evidence for relevance,
sufficiency, contradiction, and answer support with better calibration, lower
cost, and more consistent behavior than a generative LLM judge.

A **generative LLM judge** is a chat-style language model prompted to read the
question and evidence, then write an assessment. Its output can be structured,
but the model still generates the assessment as text. This study compares that
approach with Jev's fixed decision schema.

### Test

- Give every judge the same question, graph path, and source passages.
- Ask whether the evidence is relevant, sufficient, contradictory, and safe to
  use.
- Use Jev `Noul` questions for yes-or-no checks and `Score` questions for
  evidence sufficiency or path completeness.
- Compare Jev with `gpt-5.6-luna` as a judge and with fixed retrieval rules.
- Keep the final answer model and prompt fixed across all judges.
- Measure evidence precision and recall, answer correctness, unsupported
  answers, calibration, repeated-run agreement, latency, and token cost.

Evidence precision is the share of selected evidence that supports the answer.
Evidence recall is the share of the required evidence that the system finds.
An unsupported answer makes a claim that the selected evidence does not support.

The paper should not claim that Jev is the first small model used in RAG.
[Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/) and
[ReflectiveRAG](https://aclanthology.org/2026.eacl-industry.27/) already study
smaller models as retrieval controllers. A narrower claim is an evaluation of
an RLCD-based typed decision model for graph-evidence control in GraphRAG.

## Retrieval-dependent value of ontology-guided construction

### Research question

When does ontology-guided graph construction help GraphRAG retrieval?

### Hypothesis

Ontology-guided construction is not always better or worse. Its value depends
on the retrieval method. It may help retrieval methods that use entity types
and relations. It may give less benefit, or hurt, when retrieval relies mainly
on vector similarity.

The key idea is an **interaction**. An interaction means that the effect of
one choice changes when another choice changes. Here, the effect of graph
construction may change when the retrieval method changes.

### Test

Run every construction choice with every retrieval choice:

- construction: standard and ontology-guided;
- retrieval: vector, entity-vector, and path or hybrid;
- answer model and context-token budget: fixed.

A **context-token budget** is the maximum number of evidence tokens given to
the final answer model. Keeping this budget fixed makes the comparison fair.

Test the construction-by-retrieval interaction instead of selecting one
overall winner. Measure answer correctness, evidence coverage, relation and
path completeness, latency, and token use.

Split and bootstrap by record. A **record-level split** keeps questions from
one source record in the same split. **Bootstrapping** resamples whole records
to estimate uncertainty. Both steps prevent several questions from one record
from making the sample look larger than it is.

The result can show when ontology guidance helps, when it does not, and which
retrieval methods can use the extra graph structure.

## Literature boundary

The paper should not be a plain RAG-versus-GraphRAG comparison. Recent work
already covers broad comparisons in [RAG vs. GraphRAG](https://arxiv.org/abs/2502.11371)
and [GraphRAG-Bench](https://arxiv.org/abs/2506.05690). [Dissecting
GraphRAG](https://aclanthology.org/2026.tacl-1.29/) studies construction,
clustering, and reporting choices. [SetR](https://aclanthology.org/2025.acl-long.861/)
studies selecting a set of passages instead of ranking passages one at a time.

These papers mean that the contribution should be specific. The two directions
here focus on typed evidence decisions and the interaction between graph
construction and graph retrieval.
