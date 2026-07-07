# GraphRAG Retrieval Patterns: Hybrid, Community, PathRAG vs Neighborhood

Research finding for **action item #1** from the GraphRAG analysis. This document
explains how graph-based retrieval differs from flat vector RAG, with emphasis on
**hybrid retrieval** and **community-graph retrieval**, and the core distinction
between **PathRAG** and **neighborhood retrieval** — including worked examples in
the financial-filing domain (aligned with FiNER-139).

**Interactive UI:** `/graphrag` in the frontend (nav: GraphRAG).

> **Scope:** Design reference for future `RagArchitecture.GRAPH` work (D15). RecRAG
> production retrieval today is dense vector search only (`src/pipelines/retrieval.py`).

---

## 1. Why graph retrieval exists

Flat RAG retrieves isolated text **chunks** by embedding similarity. GraphRAG first
builds a **knowledge graph** (entities + relations), then retrieves **structured
context** — subgraphs, paths, or community summaries — before prompting the LLM.

| Approach | What gets retrieved | Best for |
|----------|---------------------|----------|
| **Vector RAG** | Top-k similar chunks | Single-fact lookup, definitions |
| **Neighborhood retrieval** | Ego-network around seed entities (k-hop neighbors) | Local multi-entity context |
| **Community-graph retrieval** | Pre-summarized clusters (local or global) | Thematic / corpus-wide questions |
| **PathRAG** | Pruned relational **paths** between entities | Multi-hop reasoning with less noise |
| **Hybrid retrieval** | Vector + graph combined (selection or integration) | Mixed query types in one product |

---

## 2. Community-graph retrieval (Microsoft GraphRAG style)

### How it works

1. **Build** a KG from documents (LLM extraction or rules — see FiNER-139 construction methods).
2. **Detect communities** (e.g. Leiden algorithm) at multiple levels.
3. **Summarize** each community into a report (bottom-up: detail → themes).
4. **Retrieve** using one of two modes:

| Mode | Also called | What is retrieved | Query fit |
|------|-------------|---------------------|-----------|
| **Local search** | Community local / neighborhood-style | Matched entities, incident edges, **lower-level community reports** | “What is X’s relationship to Y in this filing?” |
| **Global search** | Community global | **High-level community summaries** only (semantic match to query) | “What are the main risk themes across all 10-Ks?” |

### Example — financial filing corpus

**Corpus:** 50 SEC 10-K filings (technology sector).

**Graph construction:** Entities = companies, products, revenue lines, risk factors;
edges = `REPORTS_REVENUE`, `ACQUIRED`, `COMPETES_WITH`, `DISCLOSES_RISK`.

**Communities detected (simplified):**

```
Community C1 (Cloud revenue)     Community C2 (Supply chain)
  ├─ AWS, Azure, GCP summaries      ├─ semiconductor shortage
  └─ segment revenue tables           └─ vendor concentration

Community C3 (Regulatory) — parent summary: "Cross-filing compliance themes"
```

**Query A (local / neighborhood-like):**  
*“What revenue did Company X report for cloud services in FY2023?”*

- Extract seed entity: `Company X`
- **Local search** pulls: `Company X` node, `REPORTS_REVENUE → Cloud segment` edges,
  and the **C1 community report** snippet that already aggregates cloud metrics
- LLM answers from focused local context

**Query B (global):**  
*“What are the dominant risk themes across all filings this year?”*

- **Global search** retrieves top-level summaries from C1, C2, C3 (not every chunk)
- LLM synthesizes themes without reading thousands of pages

### Trade-offs

| Pros | Cons |
|------|------|
| Handles **global** questions vector RAG misses | Upfront cost: community detection + summarization |
| Hierarchical summaries = scalable context | Community boundaries may split related facts |
| Strong for thematic synthesis | Local mode still risks **redundant** neighbor context |

---

## 3. Neighborhood retrieval

### How it works

Given a query:

1. **Seed** — map query to one or more graph nodes (entity linking, NER, or embedding match).
2. **Expand** — collect all nodes within **k hops** (or top-N neighbors by edge weight).
3. **Package** — serialize nodes, edges, and attached text spans into the LLM prompt.

This is the “ego network” pattern used (implicitly) by many GraphRAG **local** variants
and by systems like **LightRAG** (local keyword → neighbor subgraph).

### Example — same filing graph

**Query:** *“Who acquired whom, and for how much, involving Company X?”*

**Seeds:** `Company X` (matched from query).

**1-hop neighborhood:**

```
Company X --ACQUIRED--> Subsidiary Y ($2.1B, 2022)
Company X --REPORTS--> DebtInstrumentFaceAmount $500M
Competitor Z --COMPETES_WITH--> Company X
```

**2-hop neighborhood** adds neighbors of `Subsidiary Y`, `Competitor Z`, etc.

**Retrieved context (flat list):** all nodes + edges in the subgraph, often as
triples or JSON — **everything reachable within k hops**.

### Trade-offs

| Pros | Cons |
|------|------|
| Simple to implement | **Redundancy** — many edges/paths irrelevant to the question |
| Good local coverage | Context grows quickly with k (token cost, LLM confusion) |
| Natural for “everything near X” | No ranking of *which* paths matter for reasoning |

---

## 4. PathRAG vs neighborhood retrieval — the main difference

Both start from seed entities. The difference is **what you keep** after expansion.

| Dimension | Neighborhood retrieval | PathRAG |
|-----------|------------------------|---------|
| **Unit retrieved** | Entire k-hop subgraph (nodes + all edges) | **Key relational paths** only (pruned sequences) |
| **Redundancy** | High — duplicate facts via multiple edges | Low — flow-based pruning drops weak paths |
| **Prompt structure** | Flat triple list or node descriptions | **Ordered paths** (A → rel → B → rel → C) |
| **Reasoning cue** | Implicit in bag of triples | Explicit chain in prompt |
| **Token efficiency** | Poor at larger k | ~40% fewer tokens vs full subgraph (per PathRAG paper) |
| **Best queries** | “What is near X?” | “How does A connect to B?” / multi-hop causal chains |

### Intuition

- **Neighborhood:** “Give me everything within distance k of these seeds.”
- **PathRAG:** “Give me the **few most reliable paths** that explain how seeds connect to answers.”

PathRAG adds a **pruning/scoring** step (e.g. resource propagation with decay) so only
high-confidence relational chains reach the LLM.

### Side-by-side example

**Query:** *“What is the link between stock-based compensation expense and unrecognized compensation cost for Company X?”*

**Graph facts (simplified):**

```
(Company X) --RECORDED--> (stock-based compensation expense: $45M)
(Company X) --HAS--> (unrecognized compensation cost: $12M)
(Company X) --OPERATES_IN--> (North America)
(Company X) --FILED--> (10-K 2023)
(North America) --HAS_REGULATION--> (SEC Rule 402)
... 40 more incidental edges within 2 hops ...
```

#### Neighborhood retrieval (k=2)

Returns **all** of the above — ~45 triples. The LLM must find the two compensation
facts among noise (regulations, filing metadata, geography).

#### PathRAG

1. Vector-match seeds: `stock-based compensation`, `unrecognized compensation cost`
2. Score paths by flow from seeds
3. Return **top paths only**, e.g.:

```
Path 1 (score 0.91):
  Company X --RECORDED--> stock-based compensation expense ($45M)
  Company X --HAS--> unrecognized compensation cost ($12M)

Path 2 (score 0.12, dropped):
  Company X --OPERATES_IN--> North America --HAS_REGULATION--> SEC Rule 402
```

Prompt contains **2 hops on-topic**, not 45 triples.

### When to prefer which

| Choose **neighborhood** when | Choose **PathRAG** when |
|------------------------------|-------------------------|
| Exploration / broad context around an entity | Precise multi-hop questions |
| Small graphs, low k | Large graphs, token budget matters |
| Prototyping local GraphRAG | Production paths with measurable noise |

---

## 5. Hybrid retrieval

**Hybrid** here means combining **vector (semantic) retrieval** with **graph
(structural) retrieval** — not to be confused with FiNER-139’s **Hybrid Construction**
graph-construction method (schema-guided LLM extraction). See `/graphrag` UI and
[finer139-graph-construction.md](./finer139-graph-construction.md).

### Two practical strategies (RAG vs GraphRAG survey, 2025)

| Strategy | Mechanism | Example |
|----------|-----------|---------|
| **Selection** | Router classifies query → run **either** vector RAG **or** graph retrieval | Simple fact → Milvus; “compare themes across filings” → community global |
| **Integration** | Run **both**, merge/dedupe contexts, single LLM call | Query entities via graph local search **and** fetch similar chunks via embeddings |

### Example — hybrid **selection** router

```
Query: "What is the ticker symbol of Company X?"
  → intent: single-fact lookup
  → route: vector RAG only (chunk mentions ticker in header)

Query: "How did supply chain issues propagate from vendors to margin guidance?"
  → intent: multi-hop thematic
  → route: PathRAG (paths: vendor → disruption → cost → margin guidance)
```

### Example — hybrid **integration**

```
Query: "Summarize Company X compensation disclosures and similar peer practices"

1. Graph local: seeds Company X → neighborhood/path around compensation entities
2. Vector: top-5 chunks semantically similar to "peer compensation disclosure"
3. Merge contexts (dedupe by source paragraph)
4. Single LLM answer with citations from both sources
```

### Relation to community + PathRAG

A production **hybrid** stack might route like:

```
                    ┌─ vector RAG (naive)
                    │
Query ── router ────┼─ community local (neighborhood + reports)
                    │
                    ├─ community global (thematic)
                    │
                    └─ PathRAG (pruned paths for multi-hop)
```

RecRAG’s recommendation target (`RagArchitecture.GRAPH`) would eventually pick among
these patterns per use case — today only naive vector is implemented.

---

## 6. Mapping to RecRAG roadmap

| Pattern | RecRAG status | FiNER-139 connection |
|---------|---------------|----------------------|
| Vector RAG | **Shipped** (`RetrievalPipeline.retrieve`) | N/A |
| Neighborhood / community local | Planned (D15 `GRAPH`) | Needs accurate **entity recognition** at graph build time |
| Community global | Planned | Needs good **community summaries** |
| PathRAG | Planned | Needs reliable **relation extraction** between recognized entities |
| Hybrid selection/integration | Planned | Orchestrator could route by query type |

**FiNER-139 action item #2** benchmarks which **graph construction** method best
recognizes entities — the prerequisite for any retrieval pattern above. See
[finer139-method-ranking.md](./finer139-method-ranking.md).

---

## References

- Edge et al. — Microsoft GraphRAG (community detection + local/global search)
- PathRAG — flow-based path pruning ([arXiv:2502.14902](https://arxiv.org/html/2502.14902))
- RAG vs GraphRAG systematic evaluation — hybrid Selection/Integration ([arXiv:2502.11371](https://arxiv.org/html/2502.11371v2))
- LightRAG — dual-level (local + global) retrieval
- RecRAG FiNER-139 experiment — `src/experiments/finer139/`
