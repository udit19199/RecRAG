# How RecRAG works

RecRAG tests whether a graph helps a language model answer questions whose
answers are spread across several pieces of text.

The current code uses 2WikiMultiHopQA. A record contains a question, source
pages, a known answer, and the passages that support that answer. The app then
does the following:

1. It turns the source pages into a graph in Neo4j.
2. It searches that graph with one or more retrieval methods.
3. It asks a language model to answer from the retrieved context.
4. It scores the graph, the retrieved context, and the answer.

The research question is whether the construction method and retrieval method
change the quality of the final answer.

## Read these notes in this order

- [System overview](architecture.md) explains each step and why the code does it.
- [2WikiMultiHopQA experiment](2wikimultihopqa-experiment.md) records the current dataset experiment and its scores.
- [Dataset focus](dataset-focus.md) explains the planned datasets and the support that exists today.
- [REFinD experiment](refind-experiment.md) records an older experiment that is not part of the current code path.

## Terms used in these notes

- A **record** is one dataset example. It contains a question and its source material.
- A **node** is an entity in the graph, such as a person, company, or film.
- A **relationship** is a named link between two nodes, such as `DIRECTED_BY`.
- A **chunk** is a smaller piece of source text used for search.
- An **embedding** is a number-based representation of text meaning. The system compares embeddings to find text with similar meaning.
- **Retrieval** is the step that chooses context for a question.
- **Construction** is the step that creates the graph from source text.
- An **LLM** is a language model that reads text and generates text.

These docs describe the experiment. They do not change the repository operating
guide in `AGENTS.md`.
