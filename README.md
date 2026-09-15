# RecRAG

RecRAG is a research project by HPE and JECRC University. It compares ways to
build and search a knowledge graph for multi-hop question answering.

## Current study

The active study uses the 2WikiMultiHopQA dataset. Each record has a question,
source pages, supporting passages, and a known answer.

We compare:

- Graph construction: `standard` and `ontology_guided`
- Retrieval: `text2cypher`, `agentic`, `vector`, and `hybrid`

The run keeps the dataset, answer model, prompts, and evaluation rules fixed
while it changes the construction and retrieval methods.

The main flow is:

```text
2Wiki record -> build graph -> retrieve context -> generate answer -> evaluate
```

We measure three things:

- Graph quality: did construction preserve supported facts?
- Retrieval quality: did retrieval return the facts needed for the answer?
- Answer quality: is the final answer correct and supported by the context?

We also record model calls, token use, and cost. The [cost and model estimates](docs/api-cost-estimate.md)
document explains why the full comparison becomes expensive.

## Run the app

1. Install the dependencies:

   ```sh
   uv sync --extra experiments
   ```

2. Create `.env` from `.env.example` and set the required API keys.

3. Start Neo4j Desktop Enterprise with APOC enabled.

4. Run the Streamlit app:

   ```sh
   uv run streamlit run streamlit_app.py
   ```

The app builds graphs in Neo4j and runs the selected construction, retrieval,
answering, and evaluation methods.

## Read the docs

- [Research overview](docs/README.md) explains the project with a 2Wiki example.
- [System architecture](docs/architecture.md) shows construction and retrieval.
- [Evaluation reference](docs/evaluation.md) defines each score and its limits.
- [2WikiMultiHopQA experiment](docs/2wikimultihopqa-experiment.md) records the dataset run.
- [Experimental methodology](docs/methodology.md) defines controls and run rules.
- [Findings](docs/findings.md) records saved results.
- [Cost and model estimates](docs/api-cost-estimate.md) breaks down model use and cost.

## Repository layout

```text
graphrag/       Graph construction, retrieval, answering, and evaluation
docs/           Experiment explanations, diagrams, results, and plans
streamlit_app.py  Research interface
config.toml     Model, embedding, and adapter settings
```
