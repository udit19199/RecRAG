# RecRAG research context

This glossary defines the terms for comparing complete RAG architectures and
studying how a model can route a use case to one of them.

## Language

**Architecture**:
A complete RAG system, from source ingestion through answer generation. It
includes the data representation, indexes, retrieval, reasoning, and answer
steps.
_Avoid_: setup, pipeline

**Architecture family**:
A broad design category such as Classic RAG, GraphRAG, or Agentic RAG.
_Avoid_: mode, type

**Candidate**:
One concrete architecture configuration that can run in an experiment.
_Avoid_: option, setup

**Use-case profile**:
A description of the data, questions, and requirements that a candidate must
meet.
_Avoid_: prompt, user query

**Hybrid architecture**:
A candidate that combines methods from different architecture families. Hybrid
architectures are out of scope for the current study.
_Avoid_: mixed RAG
