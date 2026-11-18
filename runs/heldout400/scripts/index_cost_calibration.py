"""Metered sample of graph-extraction calls, scaled to the full MultiHop-RAG build.
Index building was never metered (cost.indexing() suspends the meter) and the
construction cache keeps no usage, so this measures real per-chunk usage."""
import json, os, random, statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from neo4j_graphrag.experimental.components.types import Neo4jGraph
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from benchmark import split_pages
from cost import Pricing, measure
from dataset import load_multihop_rag_record
from graphrag_construction import NEWS_ONTOLOGY_SCHEMA, ONTOLOGY_EXTRACTION_PROMPT
from graphrag_retrieval import GraphRAG

rag = GraphRAG.from_config(Path(os.environ["RECRAG_CONFIG"]), dataset="multihop_rag")
llm = rag._construction_llm
llm._model.cache = None
chunks = split_pages(load_multihop_rag_record(0).pages)
sample = random.Random(11).sample(chunks, 100)
template = ERExtractionTemplate(template=ONTOLOGY_EXTRACTION_PROMPT)
schema = NEWS_ONTOLOGY_SCHEMA.model_dump(exclude_none=True)

def one(text):
    with measure(Pricing()) as cost:
        llm.invoke([{"role": "user", "content": template.format(text=text, schema=schema, examples="")}],
                   response_format=Neo4jGraph)
    return cost

with ThreadPoolExecutor(8) as pool:
    costs = list(pool.map(one, sample))
calls = [c for q in costs for c in q.calls if c.kind == "responses"]
out = {"chunks_total": len(chunks), "sampled": len(sample), "calls_per_chunk": len(calls) / len(sample),
       "input_per_chunk": st.mean(sum(c.input_tokens for c in q.calls) for q in costs),
       "output_per_chunk": st.mean(sum(c.output_tokens for c in q.calls) for q in costs),
       "reasoning_per_chunk": st.mean(sum(c.reasoning_tokens for c in q.calls) for q in costs),
       "list_usd_per_chunk": st.mean(sum(c.input_tokens * 0.10 + c.output_tokens * 0.50 for c in q.calls) / 1e6 for q in costs)}
out["list_usd_full_build"] = out["list_usd_per_chunk"] * len(chunks)
json.dump(out, open("runs/heldout400/build_calibration/graph_build_calibration.json", "w"), indent=1)
print(out)
