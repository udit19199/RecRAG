# 2WikiMultiHopQA results from five records

> Ontology-guided graphs held more facts. Standard graphs gave better answers.
> Five records is too few to pick a winner.

This run asks which candidate answers multi-hop questions best. A candidate is
one construction choice plus one retrieval choice, such as standard
construction with vector retrieval.

```mermaid
flowchart LR
    A[Five 2Wiki records] --> B[Two construction choices]
    B --> C[Ten Neo4j graphs]
    C --> D[Three retrieval choices per graph]
    D --> E[Thirty retrieval and answer cases]
    E --> F[Saved scores]
```

## Method in brief

Each record asks a question that needs facts from more than one passage. Each
candidate builds its own graph, retrieves context for the question, and asks
the answer model once.

- [Graph construction](construction.md) builds one database per record and
  choice.
- [Graph retrieval](retrieval.md) picks context for one question.
- [GraphRAG evaluation](evaluation.md) scores the graph, context, and answer
  separately. The scores are recorded and never combined into one number.

## Run setup

- Records: 5 from `dev.json`, with source pages, supporting passages, and
  expected answers.
- Answer and judge model: `gpt-5.6-luna`.
- Retrieval depth: top 5 items per question.
- Saved report: [`2026090905_report.pdf`](../../results/recrag/2026090905_report.pdf).
- Tables below are transcribed from that report.

The report uses old retrieval names. `vector_cypher` in the tables is today's
`vector`. Standalone `text2cypher` no longer exists in the current retrieval
method list. `entity_vector` was not in this run.

## Example record

- Record ID: `8813f87c0bdd11eba7f7acde48001122`
- Question: Who is the mother of the director of film *Polish-Russian War (Film)*?
- Expected answer: Małgorzata Braunek

Supporting facts:

1. Page: `Polish-Russian War (film)`
   Passage 1:

   > Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by Xawery Żuławski based on the novel Polish-Russian War under the white-red flag by Dorota Masłowska.

2. Page: `Xawery Żuławski`
   Passage 2:

   > He is the son of actress Małgorzata Braunek and director Andrzej Żuławski.

## What the run found

Scores are averages over 5 records. Each judge score runs from 0 to 1. G, C,
and S mean groundedness, completeness, and supporting evidence coverage.

| Construction | Groundedness | Completeness | Supporting evidence coverage |
| --- | ---: | ---: | ---: |
| Standard | 0.56 | 0.44 | 0.34 |
| Ontology-guided | 0.78 | 0.58 | 0.64 |

Ontology-guided scored higher on all three checks. It also built bigger graphs:
70.80 triples on average versus 46.40 for standard, at 41.45 build seconds
versus 30.74.

| Record | Standard G / C / S | Ontology-guided G / C / S |
| ---: | --- | --- |
| 1 | 0.60 / 0.30 / 0.70 | 0.90 / 0.70 / 0.70 |
| 2 | 0.30 / 0.30 / 0.20 | 0.50 / 0.60 / 0.20 |
| 3 | 0.90 / 0.90 / 0.70 | 0.90 / 0.70 / 0.90 |
| 4 | 0.30 / 0.30 / 0.10 | 0.90 / 0.80 / 0.70 |
| 5 | 0.70 / 0.40 / 0.00 | 0.70 / 0.10 / 0.70 |

### Answer results

All scores are averages over 5 records.

| Construction | Retrieval | Contextual precision | Contextual recall | Contextual relevancy | Answer relevancy | Correctness | Faithfulness | Alias match |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | `text2cypher` | 0.60 | 0.30 | 0.60 | 1.00 | 0.60 | 0.67 | 1/5 |
| Standard | `agentic` | 0.87 | 1.00 | 0.13 | 1.00 | 1.00 | 1.00 | 0/5 |
| Standard | `vector_cypher` | 0.94 | 1.00 | 0.15 | 1.00 | 1.00 | 1.00 | 1/5 |
| Ontology-guided | `text2cypher` | 0.20 | 0.00 | 0.20 | 0.20 | 0.20 | 1.00 | 0/5 |
| Ontology-guided | `agentic` | 0.91 | 0.80 | 0.21 | 1.00 | 0.98 | 0.80 | 0/5 |
| Ontology-guided | `vector_cypher` | 0.86 | 1.00 | 0.19 | 1.00 | 0.98 | 1.00 | 0/5 |

Standard with vector had the best mix: 0.94 precision, 1.00 recall, and 1.00
correctness. Standalone `text2cypher` was weakest on both graph choices.

Faithfulness averages hide missing cases. Standard `text2cypher` scored 0.67
on only 3 cases with context. Ontology-guided `text2cypher` scored 1.00 on
only 1 case. Faithfulness is skipped when retrieval returns no context.

## Limits and next run

- Five records cannot rank methods. Record 5 shows why: ontology-guided
  completeness fell to 0.10 while standard held 0.40.
- Better graphs did not mean better answers here. That gap is still open.
- The alias check is strict. A longer correct answer can fail it.
- This run did not test `entity_vector`, latency, or token cost.

Repeat the full 2 × 3 matrix on more records. Record the sample size, split,
and selection rule. Compare evidence-backed correct answers first.
