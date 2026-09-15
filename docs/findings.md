# Findings

This page records the current interpretation of the saved evidence. It
separates observations from decisions and keeps small or incomplete runs from
looking more certain than they are.

## Current decision

The five-record 2WikiMultiHopQA run gives `standard + vector` the provisional
lead for final-answer quality. It is not locked. The next study must run the
same eight combinations on a larger sample.

`ontology_guided` is the stronger construction method in the small run. It
scored higher on groundedness, completeness, and supporting-evidence coverage.
That graph advantage did not translate into the best final-answer scores in
this sample.

## Five-record 2WikiMultiHopQA run

The saved report is [2026090905_report.pdf](../results/recrag/2026090905_report.pdf).
The scores below are averages over five records.

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

The answer scores favour standard construction. Among its strong retrieval
methods, vector retrieval has the highest precision in this table and matches
agentic and hybrid retrieval on recall, answer relevance, correctness, and
recorded faithfulness.

The result also shows why the project needs more than one metric. The
ontology-guided graph is better represented in the graph metrics, but the
standard graph paired with vector retrieval gives the strongest final-answer
profile in this small sample. `text2cypher` is the clear weak point in this
run.

## Early 39-record Hannon run

The [final 39-record artifact](../results/graphrag/20260827_132202_1451/report.json)
contains 39 Hannon records and 7 synthesis questions. It compares an LLM
construction path with an NLP construction path. It is historical and
incomplete as a configuration study.

The construction observations are clear:

| Construction path | Input records | Model calls | Extracted links | Reported abstentions | Construction judge accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLM | 39 | 39 | 600 | 0 | 0.9815 |
| NLP | 39 | 0 | 43 | 32 | 0.9051 |

The LLM path produced a much richer graph and received the higher construction
judge score. The NLP path was cheaper because it made no model calls, but it
abstained on most records in this run. The judge supports also differ, so these
values describe the run rather than establish a final cost-quality tradeoff.

The retrieval evidence is partial. In one earlier LLM comparison, hybrid
retrieval had higher entity recall and judge accuracy than node retrieval:

| Retrieval | Entity recall | Judge accuracy | Relation recall |
| --- | ---: | ---: | ---: |
| Hybrid | 0.583 | 0.486 | 0.000 |
| Node | 0.292 | 0.104 | 0.000 |

Both methods had zero relation recall in that comparison. The final 39-record
artifact tested node retrieval for both construction paths only. Its retrieval
evaluation covered 7 questions, not all 39 records. It therefore cannot decide
the best GraphRAG configuration.

## What the project has learned

- Construction quality and answer quality are related, but they are not the
  same measurement.
- Retrieval method matters. Hybrid and vector retrieval can expose useful
  context without requiring the answer model to generate a Cypher query.
- Direct `text2cypher` retrieval is promising for relationship-heavy questions
  in principle, but it performed poorly in the current small run.
- Partial runs are useful for finding failure modes, not for locking a winner.
- The next comparison must keep the matrix and evaluation protocol fixed while
  increasing the 2WikiMultiHopQA sample.

## Open interpretation

The larger 2WikiMultiHopQA run will answer whether the provisional
`standard + vector` result holds beyond five records. It may also show whether
the graph-quality advantage of `ontology_guided` becomes useful at a larger
sample or remains separate from final-answer quality.
