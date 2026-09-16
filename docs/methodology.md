# Experimental methodology

## Experimental unit

The basic unit is one dataset record. For 2WikiMultiHopQA, a record contains a
question, source pages, an expected answer, and passages that support the
answer.

Each record is run through one configuration. The construction method builds a
graph from the source pages. The retrieval method searches that graph. The
answering step then receives the returned context.

The comparison is a 2 × 4 matrix:

| | `text2cypher` | `agentic` | `vector` | `hybrid` |
| --- | --- | --- | --- | --- |
| `standard` | run | run | run | run |
| `ontology_guided` | run | run | run | run |

## Controlled conditions

The active comparison keeps these conditions fixed:

- the dataset split and selected records;
- the `gpt-5.6-luna` model and configured reasoning effort;
- the `text-embedding-3-small` embedding model;
- Neo4j as the graph, chunk, and index store;
- the answer-generation path;
- the evaluation inputs and metric definitions.

The model stays fixed so a model change cannot look like a construction or
retrieval improvement. A model comparison is a separate study. It changes only
the model and keeps the record sample, prompts, methods, and eval rules fixed.

Each record and construction method uses a separate Neo4j database. This keeps
the graph versions separate while retrieval methods query the same constructed
graph. The construction flow creates both vector and full-text indexes before
retrieval starts.

## Evaluation hierarchy

The primary outcome is the quality of the final answer:

- is the answer correct;
- does it answer the question;
- is it grounded in the context returned by retrieval?

The project uses graph and retrieval metrics to explain that outcome:

- graph groundedness, completeness, and supporting-evidence coverage;
- retrieval precision, recall, and context relevance;
- answer faithfulness, relevance, correctness, and strict alias matching.

These metrics are not one universal score. A graph can score well on source
coverage while a retriever misses the path needed by the question. A retriever
can return useful context while the answer model still makes a mistake.

The detailed evaluator contract is in [Evaluation reference](graphrag/evals.md).

## Why the full run costs more

Each record runs two construction methods and four retrieval methods. With all
evals enabled, one record produces:

- two graph builds;
- eight retrieval and answer runs;
- two graph evaluations;
- eight retrieval evaluations;
- eight answer evaluations.

The full matrix therefore runs 28 work units per record. Some evaluation units
make more than one model call. Agentic retrieval can make up to five model calls
for one search. See [Cost and model estimates](api-cost-estimate.md).

## Winner rule

The project will choose the configuration with the strongest correct,
evidence-grounded final answers on the larger 2WikiMultiHopQA run. Graph and
retrieval metrics explain the result and break ties. A small run can produce a
provisional winner, but it cannot lock the project-wide configuration.

## Completion rules

A comparison is complete only when:

- every matrix cell has run on the same selected sample;
- construction and retrieval results exist for every intended combination;
- the evaluator ran with the same definitions across combinations;
- the run records its dataset, split, sample, model, configuration, and errors;
- raw results are preserved under `results/`;
- incomplete combinations are marked instead of being averaged with complete ones.

Past runs used different datasets, methods, and evaluation paths. Their scores
are evidence for specific questions. They must not be pooled into one winner.
