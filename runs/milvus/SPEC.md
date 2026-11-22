# Milvus as the vector database, no graph (21 Nov 2026)

Udit's words (21 Nov 2026):
> Add mivus as a db, test on that as well. Self host milvus.
> I want both agent and non agent. Basically seeing without graph what happens in a normal vector db
> We need same tools as the current setup without graph search or cypher search. Vector search is the one that remains. All these results should be in the excalidraw scene. Instabox is fine. Go ahead.
> We can do the work in bits and chunks.

Question: with no graph, does swapping Neo4j's vector index (agent) or the
llama-index store (Normal RAG) for a dedicated vector database change the results?

## Host

milvus: milvusdb/milvus:v2.5.10 standalone (embedded etcd, local storage), Docker on the Instabox VM omarchy-mpv0 (4 vCPU, 8 GB), port 19530 bound to 127.0.0.1; data under /data/home/milvus
runs: on the VM only (never the Mac), one job at a time; results copied back to runs/milvus/runs/ after each chunk
python env (VM): Mac .venv package versions (uv pip freeze, installed --no-deps), plus torch 2.8.0+cpu, transformers 5.23.0, pymilvus 2.5.10

## Data (no re-embedding)

chunks: the 7,728 MultiHop-RAG chunks (benchmark.split_pages per page). The graph's chunk set and Normal RAG's llama-index nodes are the same 7,728 texts (checked 21 Nov).
agent_chunks: title, text, vector from the graph build's embedding cache (every chunk found by its text hash) = the vectors in Neo4j that A1 searched
rag_chunks: text, vector from Normal RAG's saved llama-index store = the vectors Normal RAG searched
index: HNSW, COSINE, M=16, efConstruction=200; search ef=128
export: runs/milvus/scripts/export_chunks.py -> runs/milvus/export/*.npz; load: runs/milvus/scripts/load.py

## Variants (250 MultiHop-RAG comparison + temporal questions of runs/nograph/questions.json, 3 repeats)

| variant | compared with | the one difference |
|---|---|---|
| Agent on Milvus (scripts/agent.py) | A1 no-graph agent (runs/nograph) | vector_search queries Milvus agent_chunks, top 10; same tool description, same NO_GRAPH_AGENT_INSTRUCTION, same picking, same answer; scores (1 + cos) / 2; Neo4j never contacted |
| B0 on Milvus (selection run.py B0, RECRAG_VECTOR_DB=milvus) | Normal RAG (runs/final500) | top 10 from Milvus rag_chunks |
| B2 on Milvus (selection run.py B2, RECRAG_VECTOR_DB=milvus) | B2 (runs/selection) | top 50 from Milvus rag_chunks, same selection prompt |

answer_model: gpt-6-luna, reasoning effort medium; embedding_model: text-embedding-3-small (query embeddings only)

## Scoring (same as runs/final500)

answered_right: exact match, then the Claude Haiku answer judge, blind to method.
gold_facts: Claude Haiku fact judge on the frozen runs/final500 claims.
Per question type, each repeat. All results go into the Excalidraw Experiments scene.

## Cost estimate

Agent: about $0.004 x 750 = $3. B0: about $0.0003 x 750 = $0.25. B2: about $0.0015 x 750 = $1.1. Total about $4.4.
