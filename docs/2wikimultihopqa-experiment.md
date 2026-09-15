# 2WikiMultiHopQA experiment

This note records the five-record study. See [Findings](findings.md) for the
current interpretation and [Experimental methodology](methodology.md) for the
comparison rules.

2WikiMultiHopQA tests questions that need more than one step of reasoning. The
answer may require a fact about one page and a second fact about a related page.
The dataset module reads source pages, supporting facts, evidence triples, and
the expected answer from `dev.json`. It also resolves answer aliases from
`id_aliases.json`.

The scores in this document came from the saved five-record report at
[`results/recrag/2026090905_report.pdf`](../results/recrag/2026090905_report.pdf).
The repository also contains the loader and evaluation code used by the active
GraphRAG path.

## Example question

Record `8813f87c0bdd11eba7f7acde48001122` asks:

> Who is the mother of the director of film *Polish-Russian War (Film)*?

The expected answer is Małgorzata Braunek. The supporting facts provide the two
steps needed to reach it:

1. *Polish-Russian War* was directed by Xawery Żuławski.
2. Xawery Żuławski was the son of Małgorzata Braunek.

![2WikiMultiHopQA example record](2wikimultihopqa-example-record.png)

## Methods

### Construction

```text
source pages -> [standard | ontology_guided] -> Neo4j graph
```

- **Standard** uses the default `neo4j-graphrag` extraction schema and prompt.
- **Ontology-guided** gives the extractor fixed top-level kinds such as Person,
  Organization, Place, CreativeWork, Event, Concept, and Thing. It still allows
  additional kinds and relationships.

The comparison asks whether the guided rules improve the graph.

### Retrieval

```text
question -> [text2cypher | agentic | vector | hybrid] -> context -> answer
```

| Method | What the code does |
| --- | --- |
| `text2cypher` | Gives the schema to an LLM. The LLM writes a read-only Cypher query. |
| `agentic` | Lets an agent choose vector search or Cypher search and refine the query. |
| `vector` | Searches `chunk_embeddings` and adds entities and graph paths. |
| `hybrid` | Searches `chunk_embeddings` and `chunk_fulltext`, then adds graph context. |

![2WikiMultiHopQA retrieval methods](2wikimultihopqa-retrieval-methods.png)

## Construction scores

The scores are averages over five records. Each score is between 0 and 1.

| Construction | Groundedness | Completeness | Supporting evidence coverage |
| --- | ---: | ---: | ---: |
| Standard | 0.56 | 0.44 | 0.34 |
| Ontology-guided | 0.78 | 0.58 | 0.64 |

Groundedness asks whether graph facts appear in the source text. Completeness
asks how much source information the graph captures. Supporting evidence
coverage asks how much information from the dataset's supporting passages the
graph captures.

Ontology-guided construction scored higher on all three measures in this run.

![2WikiMultiHopQA construction scores](2wikimultihopqa-construction-scores.png)

## Retrieval and answer scores

These scores are also averages over five records. Retrieval evaluation checks
the first five returned items, using `top_k=5`.

| Construction | Retrieval | Precision | Recall | Context relevance | Answer relevance | Correctness |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Standard | `text2cypher` | 0.60 | 0.30 | 0.60 | 1.00 | 0.60 |
| Standard | `agentic` | 0.87 | 1.00 | 0.13 | 1.00 | 1.00 |
| Standard | `vector` | 0.94 | 1.00 | 0.15 | 1.00 | 1.00 |
| Standard | `hybrid` | 0.92 | 1.00 | 0.13 | 1.00 | 1.00 |
| Ontology-guided | `text2cypher` | 0.20 | 0.00 | 0.20 | 0.20 | 0.20 |
| Ontology-guided | `agentic` | 0.91 | 0.80 | 0.21 | 1.00 | 0.98 |
| Ontology-guided | `vector` | 0.86 | 1.00 | 0.19 | 1.00 | 0.98 |
| Ontology-guided | `hybrid` | 0.94 | 1.00 | 0.12 | 1.00 | 0.98 |

Precision measures whether useful results appear near the top. Recall measures
how much required supporting information was returned. Context relevance
measures whether the returned context matches the question. Answer relevance
measures whether the answer addresses the question. Correctness measures
whether the answer matches the expected answer or an alias.

![2WikiMultiHopQA retrieval and answer scores](2wikimultihopqa-retrieval-answer-scores.png)

## Faithfulness and exact matching

The answer evaluator gives DeepEval the dataset's supporting sentences as
`context` and the returned items as `retrieval_context`. DeepEval's faithfulness
metric checks the answer against `retrieval_context`, so it uses the passages
returned by retrieval. The experiment did not score faithfulness when retrieval
returned no context.

The recorded faithfulness scores were:

- Standard `text2cypher`: 0.67 on three cases with context.
- Standard `agentic`, `vector`, and `hybrid`: 1.00.
- Ontology-guided `text2cypher`: 1.00 on one case with context.
- Ontology-guided `agentic`: 0.80.
- Ontology-guided `vector` and `hybrid`: 1.00.

![2WikiMultiHopQA faithfulness scores](2wikimultihopqa-faithfulness-scores.png)

The exact answer check is strict. It normalizes the generated answer and checks
whether the full result equals the expected answer or one of its aliases. A
correct answer with an explanation can fail that check. The answer judge also
scores relevance and correctness separately.

## What this result tells us

The five-record run favors ontology-guided construction for graph quality.
Standard construction paired with vector, hybrid, or agentic retrieval produced
perfect answer relevance and correctness in the recorded table. Ontology-guided
retrieval kept answer relevance at 1.00 and correctness at 0.98 for those three
methods. `text2cypher` scored lower in both construction settings in this run.

These results are observations from five records. They are not a claim that one
method wins on every dataset.
