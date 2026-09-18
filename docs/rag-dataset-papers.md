# RAG papers for RecRAG datasets

Status: 2026-09-17

RAG means retrieving evidence and then using a language model to answer. This note records how published papers use the datasets in or near this repo.
It also records when a paper changes the dataset format, samples a subset, or uses only the retrieval task.

## Scope

- The active RecRAG dataset is **2WikiMultiHopQA**.
- HotpotQA, FinQA, TAT-QA, LegalBench-RAG, and BioASQ remain candidates.
  No order has been chosen. They are not loaded by the current code.
  Dataset choice is a research decision, not a fixed implementation backlog.

## Dataset focus

This section records the dataset comparison and the choice of the next dataset.
It is not a list of loaders that already exist in the code.

The six datasets cover different document types and different kinds of answers.
This matters because a method that works on short Wikipedia passages may not work on tables or legal text.

### Current code status

The Streamlit app currently loads 2WikiMultiHopQA through `graphrag/dataset_records/two_wiki_multihopqa.py`.
Loaders were not found in the current code for HotpotQA, FinQA, TAT-QA, LegalBench-RAG, or BioASQ.

| Dataset | Current status |
| --- | --- |
| 2WikiMultiHopQA | Loaded by the current app and used by the GraphRAG evaluation code. |
| HotpotQA | Focus target. A loader was not found in the current code. |
| FinQA | Focus target. A loader was not found in the current code. |
| TAT-QA | Focus target. A loader was not found in the current code. |
| LegalBench-RAG | Focus target. A loader was not found in the current code. |
| BioASQ | Focus target. A loader was not found in the current code. |

### Dataset comparison

| Dataset | Domain | Documents or corpus | Question | Gold answer | Gold evidence | Extra supervision |
| --- | --- | --- | --- | --- | --- | --- |
| HotpotQA | General, Wikipedia | Wikipedia paragraphs or the full processed Wikipedia corpus | Yes | Yes | Supporting sentences | Question type, difficulty |
| 2WikiMultiHopQA | General, Wikipedia | Wikipedia paragraphs plus hyperlink metadata | Yes | Yes | Supporting sentences | Gold relation triples, reasoning chain |
| FinQA | Finance | Financial report text plus tables | Yes | Yes | Gold supporting facts | Gold numerical reasoning program |
| TAT-QA | Finance | Financial report paragraphs plus tables | Yes | Yes | Relevant text or table locations | Derivation, answer type, source, scale |
| LegalBench-RAG | Legal | Raw legal and contract text files | Yes | Not primarily generation QA | Exact relevant character spans | Retrieval-focused |
| BioASQ | Biomedical | PubMed titles plus abstracts | Yes | Exact answer plus long-form ideal answer | Gold PubMed documents plus snippets | Concepts or RDF triples in some releases |

- **Question** means the dataset provides a question for each example.
- **Gold answer** means the dataset provides the expected answer.
- **Gold evidence** means the dataset identifies text that supports the answer.
- **Extra supervision** means the dataset provides more labels, such as a reasoning chain, table location, or answer type.

### Example records

One real row per set. Cut for length.

- HotpotQA. Q: Which magazine was started first Arthur's Magazine or First for Women? A: Arthur's Magazine. Support: `[Arthur's Magazine, sent 0]`, `[First for Women, sent 0]`. Type: comparison. Source: `hotpotqa/hotpot_qa` row `5a7a0693`.
- 2WikiMultiHopQA. Q: Who is the founder of the company that distributed La La Land film? A: Bernd Eichinger. Triples: `[La La Land, distributor, Summit Entertainment]`, `[Summit Entertainment, founded by, Bernd Eichinger]`. Type: compositional. Source: repo `Alab-NII/2wikimultihop` plus paper Table 3.
- FinQA. Q: What percent of total cash and investments as of Dec 29 2012 was available-for-sale investments? A: 53%. Facts: available-for-sale `$14001`, total `$26302`. Source: `FinQA` row `INTC/2013/page_71.pdf-4`.
- TAT-QA. Q: What is the change in Other in 2019 from 2018? A: `-12.6`, scale million. Math: `44.1 - 56.7`. Type: arithmetic, from table-text. Source: `TAT-QA` site.
- LegalBench-RAG. Q: Are the licenses in the Cardlytics-Bank of America agreement non-transferable? A: character span in `CardlyticsInc_20180112_S-1_EX-10.16_Maintenance Agreement1.txt`, range `[44579, 45211]`. Source: paper `2408.10343` section 3.1.4.
- BioASQ. Q: Do CpG islands colocalise with transcription start sites? Exact: yes. Ideal: yes, CpG island near TSS links to gene expression. Proof: snippet on G+C rise near TSS plus doc `CpG Islands: Starting Blocks for Replication and Transcription`. Source: BioASQ-QA paper.

### Choosing the next dataset

They test whether a configuration selected on multi-hop Wikipedia questions transfers to different document structures and answer types.

Choose the next dataset after the larger 2WikiMultiHopQA run, based on what the result leaves unclear.

REFinD is documented in [the historical experiment note](refind/refind-experiment.md).
The repository guide also lists Hannon and FiNER-139 as historical datasets.

## Dataset reference

[Ho et al., “Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps” (COLING 2020)](https://aclanthology.org/2020.coling-main.580/) introduces 2WikiMultiHopQA.
It combines structured and unstructured evidence and includes reasoning-path information.
Later papers usually turn its question, context, and evidence records into an open-domain retrieval task.

## Papers that benchmark 2WikiMultiHopQA

### IRCoT (ACL 2023)

[IRCoT: “Iterative Retrieval Augmented Generation for Knowledge-Intensive Multi-Hop Question Answering”](https://aclanthology.org/2023.acl-long.557/)

- **Dataset slice:** HotpotQA, 2WikiMultiHopQA, MuSiQue, and IIRC. For each
  dataset, the authors use 100 random development questions for tuning and 500 other random development questions for testing.
- **Corpus:** the associated contexts supplied with each dataset, including
  the 2Wiki contexts. This is not a full external Wikipedia crawl.
- **Retriever:** BM25 through Elasticsearch. A first query retrieves from
  the question. Later queries use the latest generated chain-of-thought sentence.
- **RAG flow:** alternate between generating the next reasoning sentence and
  retrieving paragraphs.
  Stop when the answer is generated or the step limit is reached, then pass the gathered paragraphs to a direct-answer or chain-of-thought reader.
- **Baselines and scores:** compare one-shot question retrieval with IRCoT.
  They report paragraph recall and answer exact match/F1, with a fixed total paragraph budget and a maximum of 15 paragraphs.
- **What RecRAG can compare:** iterative retrieval and graph retrieval can be
  compared under the same question/context records, but the paper's chain-of-thought loop is a different retrieval policy from RecRAG's four retrieval methods.

### FLARE (EMNLP 2023)

[FLARE: “Forward-Looking Active Retrieval Augmented Generation”](https://aclanthology.org/2023.emnlp-main.495/)

- **Dataset slice:** includes a 500-example 2WikiMultiHopQA evaluation set.
- **Corpus and retriever:** Wikipedia with BM25; the main setup retrieves the
  top two passages. The paper also reports no-retrieval and other retrieval baselines.
- **RAG flow:** generate the next sentence while watching token confidence.
  If confidence is low, use the predicted sentence as a search query, fetch evidence, and regenerate the sentence with that evidence.
- **Baselines and scores:** compare no retrieval, one-time retrieval,
  previous-window or previous-sentence retrieval, question decomposition, and FLARE variants. They report exact match, F1, precision, and recall.
- **What RecRAG can compare:** adaptive retrieval timing is the main idea to
  compare. Its Wikipedia/BM25 corpus setup is not the same as Neo4j-backed GraphRAG.

### Adaptive-RAG (NAACL 2024)

[Adaptive-RAG: “Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity”](https://aclanthology.org/2024.naacl-long.389/)

- **Dataset slice:** SQuAD, NQ, TriviaQA, MuSiQue, HotpotQA, and
  2WikiMultiHopQA. The evaluation uses about 500 samples per dataset.
- **Corpus and retriever:** BM25 for the retrieval methods. For the multi-hop
  tasks, the paper uses the preprocessed corpus from the prior multi-hop QA work rather than the repo's Neo4j graph.
- **RAG flow:** a classifier sends each question to one of three paths: no
  retrieval, one retrieval step, or multi-step retrieval. The multi-step path uses an iterative method similar to IRCoT.
- **Baselines and scores:** compare no retrieval, single-step retrieval,
  Adaptive-RAG, Self-RAG, and a complex multi-step IRCoT baseline. Metrics include exact match, F1, accuracy, retrieval-step count, and relative time.
- **What RecRAG can compare:** method selection by question complexity. The
  paper is also useful for measuring retrieval cost, not only answer quality.

### End-to-End Beam Retrieval (NAACL 2024)

[“End-to-End Beam Retrieval for Multi-Hop Question Answering”](https://aclanthology.org/2024.naacl-long.96/)

- **Dataset slice:** MuSiQue-Ans, HotpotQA in the distractor setting, and
  2WikiMultiHopQA. The authors sample 500 questions per dataset.
- **Corpus:** each question has a fixed candidate set: 20 passages for
  MuSiQue-Ans and 10 passages for both HotpotQA and 2WikiMultiHopQA.
  The authors do not use 2Wiki's entity-relation tuple annotations in the training or evaluation setup.
- **Retriever and reader:** Beam Retrieval is trained to select a sequence
  of passages. For 2Wiki and HotpotQA, a multitask reader extracts both the answer and supporting facts.
  The paper also tests beam retrieval with a few-shot language-model reader.
- **Metrics:** retrieval precision/recall-style measures, answer exact
  match/F1, and supporting-fact exact match/F1.
- **What RecRAG can compare:** passage-selection quality and answer quality
  under a fixed candidate pool. This is closer to a retriever/reader benchmark than to a full GraphRAG construction benchmark.

### LongRAG (EMNLP 2024)

[LongRAG: “Enhancing Retrieval-Augmented Generation with Long-context LLMs”](https://aclanthology.org/2024.emnlp-main.1259/)

- **Dataset slice:** HotpotQA, 2WikiMultiHopQA, and MuSiQue from LongBench.
  This is a standardized LongBench version of the tasks, not a direct run on the repo's local files.
- **Retrieval settings:** four chunk-size/top-k settings: 200x7, 200x12,
  500x3, and 500x5. The first number is chunk size and the second is the number of retrieved chunks.
- **RAG flow:** a hybrid retriever gets long chunks. A chain-of-thought-guided
  filter keeps factual details, and an information extractor maps evidence back to the source paragraph and extracts global information.
  The generator receives both kinds of information.
- **Baselines and scores:** compare vanilla RAG, CFIC, CRAG, Self-RAG, and
  LongRAG variants that use retrieval, filtering, extraction, or both. The main answer metric is F1.
- **What RecRAG can compare:** long-context chunking and evidence filtering.
  It does not isolate graph construction in the way RecRAG does.

### HippoRAG (NeurIPS 2024)

[HippoRAG: “Neurobiologically Inspired Long-Term Memory for Large Language Models”](https://arxiv.org/abs/2405.14831)

- **Dataset slice:** MuSiQue and 2WikiMultiHopQA are the main retrieval
  evaluation tasks, with HotpotQA also included. The paper samples 1,000 questions from each validation set.
- **Corpus:** for each selected question, the candidate passages include the
  question's supporting and distractor passages.
  The 2Wiki graph is built from those passages; the paper reports 6,119 passages, 42,694 unique nodes, and 7,867 unique edges in that corpus.
- **GraphRAG flow:** an LLM extracts open-domain entities and relations from
  passages. Dense-encoder links add synonym edges.
  At query time, an LLM extracts query entities, links them to the graph, and Personalized PageRank selects relevant passages.
- **Baselines and scores:** compare BM25, Contriever, GTR, ColBERTv2,
  Propositionizer, RAPTOR, and iterative IRCoT. They report retrieval recall at several cutoffs plus answer exact match/F1.
- **What RecRAG can compare:** this is the closest published graph-retrieval
  comparison for RecRAG.
  It gives a graph construction, entity-linking, and graph traversal baseline, but its graph is built from the sampled QA passages rather than a filing corpus.

### ChainRAG (ACL 2025)

[ChainRAG: “Integrating Knowledge Retrieval and Answer Generation through Query-Oriented Semantic Structuring”](https://aclanthology.org/2025.acl-long.1089/)

- **Dataset slice:** MuSiQue, 2WikiMultiHopQA, and HotpotQA, using the
  LongBench-style data setting. It also includes a 300-example diagnostic study per dataset.
- **Retriever and generator:** standardizes OpenAI text-embedding-small-v3
  and BGE-Reranker across methods. Generators include GPT-4o-mini, Qwen2.5-72B, and GLM-4-Plus.
- **RAG flow:** rewrite a question into progressive sub-questions, complete
  missing entities, retrieve sentences, and build a sentence graph.
  The system integrates retrieved context and intermediate sub-answers through separate answer-integration and context-integration variants.
- **Baselines and scores:** compare NaiveRAG, question-decomposition RAG,
  Iter-RetGen, LongRAG, and HippoRAG with IRCoT. Metrics include answer F1, exact match, sub-question Recall@2, and model-call count.
- **What RecRAG can compare:** sentence-level multi-hop retrieval and
  answer/context integration. Its controlled embedding and reranker setup is useful when comparing retrieval behavior across RecRAG methods.

### L-RAG (Findings of ACL 2025)

[L-RAG: “Multi-hop Retrieval-Augmented Generation with Latent Representations”](https://aclanthology.org/2025.findings-acl.816/)

- **Dataset slice:** MuSiQue, HotpotQA, and 2WikiMultiHopQA.
- **RAG flow:** use intermediate representations from the language model's
  middle layers to retrieve the next-hop evidence. The method avoids generating a separate natural-language query for every hop.
- **Extra annotation:** the authors manually annotate intermediate answers
  for a 2Wiki-derived set to establish multi-hop ground truth. This is an added annotation layer, not part of the original repo records.
- **Baselines and scores:** compare against standard and multi-hop RAG
  methods and report answer and retrieval performance.
  The paper's headline result is that latent intermediate representations improve multi-hop RAG while keeping retrieval overhead close to standard RAG.
- **What RecRAG can compare:** intermediate-hop retrieval behavior. Do not
  claim an exact apples-to-apples split or sample count unless the added annotation subset is reproduced.

### S2G-RAG (ACL 2026)

[S2G-RAG: “Structured-to-Gap Retrieval-Augmented Generation for Multi-Hop Question Answering”](https://aclanthology.org/2026.acl-long.1185/)

- **Dataset slice:** TriviaQA, HotpotQA, and 2WikiMultiHopQA.
- **Retriever:** tests both BM25 sparse retrieval and E5 dense retrieval.
- **RAG flow:** a structured sufficiency/gap judge checks whether current
  evidence is enough. If not, it creates structured gap items that become the next retrieval query. Evidence is kept at sentence level.
  The retriever and generator are not retrained.
- **Baselines and scores:** compare with SIM-RAG and RAG-Critic and report
  exact match and F1. Under the matched BM25 setting on 2Wiki, the paper reports 41.7 EM / 48.6 F1 for S2G-RAG versus 34.1 EM / 40.2 F1 for SIM-RAG.
- **What RecRAG can compare:** explicit evidence-gap detection and repeated
  retrieval. This is directly relevant to testing whether graph context is sufficient before answering.

## Other dataset references

[Yang et al., “HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering” (EMNLP 2018)](https://arxiv.org/abs/1809.09600) introduces HotpotQA.
It has 113k Wikipedia question-answer pairs that need two supporting documents.
The distractor setting gives 2 gold plus 8 distractor documents per question, with sentence-level supporting facts and comparison questions.

[Trivedi et al., “MuSiQue: Multihop Questions via Single-hop Question Composition” (TACL 2022)](https://arxiv.org/abs/2108.00573) introduces MuSiQue.
It has 25k 2-4 hop questions built bottom-up from composable single-hop pairs, with filters for connected reasoning. MuSiQue-Ans is the answerable set;
MuSiQue-Full adds unanswerable contrast questions.
The authors report a 3x human-machine gap and a 30-point F1 drop for a single-hop model, which is the source of the “harder than HotpotQA/2Wiki” claim.

## GraphRAG architecture papers and which benchmarks they use

These papers define the RAG architecture. Only HippoRAG uses 2WikiMultiHopQA. The rest use their own corpora or graph question sets.

### Microsoft GraphRAG (2024)

[Edge et al., “From Local to Global: A Graph RAG Approach to Query-Focused Summarization”](https://arxiv.org/abs/2404.16130)

- **What the paper is about:** RAG fails on global questions over a whole
  corpus (for example “what are the main themes”).
  The fix builds an entity graph with an LLM, groups entities into communities, and pre-generates a summary per community.
  At query time each community summary gives a partial answer, then a final summary merges them.
- **Benchmarks used:** no public QA set. Two private corpora in the 1
  million token range: podcast transcripts (1,669 chunks of 600 tokens) and news articles 2013–2023 (3,197 chunks).
- **How they use them:** index the full corpus in each case, then ask 125
  LLM-generated global sensemaking questions per corpus.
- **What they measure:** LLM-as-judge win rates on Comprehensiveness,
  Diversity, Empowerment, plus claim counts. No EM/F1, no Recall@k.

### LightRAG (2024)

[Guo et al., “LightRAG: Simple and Fast Retrieval-Augmented Generation”](https://arxiv.org/abs/2410.05779)

- **What the paper is about:** flat chunk retrieval gives fragmented
  answers.
  The fix adds graph structure to indexing plus a dual-level retrieval path (low-level entities and high-level relations), with an incremental update path for new data.
- **Benchmarks used:** UltraDomain textbook subset (428 textbooks, 18
  domains such as Agriculture, CS, Legal, Mix). Each corpus is about 600k–5M tokens.
- **How they use them:** index each full textbook corpus, then ask the
  same style of 125 LLM-generated global questions per corpus as GraphRAG.
- **What they measure:** GPT-4o-mini judge win rates on
  Comprehensiveness, Diversity, Empowerment, Overall against NaiveRAG/RQ-RAG/HyDE/GraphRAG. No retrieval recall, no EM/F1.

### RAPTOR (2024)

[Sarthi et al., “RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval”](https://arxiv.org/abs/2401.18059)

- **What the paper is about:** short chunks lose whole-document context.
  The fix embeds, clusters, and summarizes chunks bottom-up into a tree, then retrieves from mixed tree levels at query time.
- **Benchmarks used:** NarrativeQA (books/movies), QASPER (5,049
  questions over 1,585 NLP papers), QuALITY (long-passage multiple choice plus HARD subset).
- **How they use them:** one tree per document, 100-token leaf chunks,
  collapsed-tree query of about 2,000 tokens. No graph triples.
- **What they measure:** NarrativeQA ROUGE-L/BLEU/METEOR, QASPER answer
  F1 (55.7 with GPT-4), QuALITY accuracy (82.6%, +20 points best prior on that split).

### G-Retriever (2024)

[He et al., “G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering”](https://arxiv.org/abs/2402.07630)

- **What the paper is about:** chat with a textual graph. The fix
  retrieves top-k nodes/edges with SentenceBERT, builds a connected subgraph via Prize-Collecting Steiner Tree, and answers with a GNN plus Llama2-7b soft prompt to cut hallucination.
- **Benchmarks used:** own GraphQA collection: ExplaGraphs (2,766 small
  commonsense graphs), SceneGraphs (100k GQA graphs), WebQSP (4,737 questions over Freebase, 2-hop).
- **How they use them:** one graph per question, no passage corpus.
  Typical setting retrieves k=3 nodes/edges before the Steiner step.
- **What they measure:** Accuracy on ExplaGraphs/SceneGraphs, Hit@1 on
  WebQSP, plus hallucination rate (valid nodes/edges) and token/node savings.

### Think-on-Graph / ToG (ICLR 2024)

[Sun et al., “Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph”](https://arxiv.org/abs/2307.07697)

- **What the paper is about:** LLMs hallucinate on deep reasoning. The
  fix treats the LLM as an agent that beam-searches a knowledge graph for reasoning paths, with traceable and correctable steps and no extra training.
- **Benchmarks used:** CWQ, WebQSP, GrailQA, QALD10-en, Simple
  Questions, WebQuestions, T-REx, Zero-Shot RE, Creak. GrailQA and Simple Questions are subsampled to 1,000.
- **How they use them:** live Freebase/Wikidata graphs, no indexed
  passages. Beam width/depth 3 agent walk per question.
- **What they measure:** Exact Match / Hits@1 only. Reports SOTA on 6
  of 9 sets (for example WebQSP 82.6, GrailQA 81.4 with GPT-4).

### HippoRAG (NeurIPS 2024) — the 2Wiki graph baseline

[Gutiérrez et al., “HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models”](https://arxiv.org/abs/2405.14831)

- **What the paper is about:** integrate lots of new passages without
  forgetting. The fix mimics hippocampal indexing: LLM OpenIE triples plus dense synonym edges, then Personalized PageRank from query-linked entities to passages.
  Single-step HippoRAG matches iterative IRCoT at 10-30x lower cost.
- **Benchmarks used:** MuSiQue, 2WikiMultiHopQA, HotpotQA.
- **How they use them:** 1,000 validation questions per set. Corpus per
  set is the union of supporting plus distractor passages for those questions only (2Wiki: 6,119 passages, 42,694 nodes, 7,867 edges).
- **What they measure:** retrieval Recall@2/@5 and All-Recall plus
  answer EM/F1 with a shared reader. 2Wiki gain is the headline: R@2 59.2 to 70.7, R@5 68.2 to 89.1, QA F1 43.3 to 59.5.

## Standard RAG benchmarks and which architecture papers use them

### Original RAG (NeurIPS 2020)

[Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks”](https://arxiv.org/abs/2005.11401)

- **What the paper is about:** combine a seq2seq model with a dense
  Wikipedia index so answers use fresh text, not only weights.
- **Benchmarks used:** three open-domain QA tasks (Natural Questions,
  TriviaQA, WebQuestions lineage) plus generation tasks.
- **How they use them:** full December-2018 Wikipedia dump, 21M
  100-word passages, DPR index. Same passages condition the whole output (RAG-Sequence) or per-token (RAG-Token).
- **What they measure:** answer EM plus generation
  specificity/diversity/factuality. Sets SOTA on the three open QA tasks at the time.

### KILT (NAACL 2021)

[Petroni et al., “KILT: a Benchmark for Knowledge Intensive Language Tasks”](https://arxiv.org/abs/2009.02252)

- **What the paper is about:** one shared Wikipedia snapshot for all
  knowledge tasks so components can be reused.
- **Benchmarks used:** open QA, fact checking, slot filling, entity
  linking, dialogue — all grounded in the same snapshot.
- **How they use them:** single dense index plus seq2seq baseline
  across every task.
- **What they measure:** downstream EM/accuracy plus provenance
  R-Precision/Recall@k. This is grounding quality, not hop reasoning.

### Self-RAG (2023)

[Asai et al., “Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection”](https://arxiv.org/abs/2310.11511)

- **What the paper is about:** fixed top-k retrieval hurts when
  retrieval is unneeded or passages are bad. The fix trains reflection tokens so the model decides when to retrieve and which passages to trust.
- **Benchmarks used:** Natural Questions, TriviaQA, PopQA (long-tail
  Wikidata entities), FactScore for long-form factuality.
- **How they use them:** full Wikipedia retrieval with on-demand calls,
  7B and 13B models versus ChatGPT and retrieval-augmented Llama2.
- **What they measure:** open-QA EM/accuracy by popularity bucket,
  reasoning and fact-verification scores, citation accuracy on long-form answers.

### MultiHop-RAG (2024) — RAG-native multi-hop over news

[Tang and Yang, “MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries”](https://arxiv.org/abs/2401.15391)

- **What the paper is about:** existing RAG fails on queries needing
  2-4 news evidences. New set plus tests of embeddings and LLM readers.
- **Benchmarks used:** its own English news knowledge base with
  multi-hop queries, gold answers, and supporting evidence.
- **How they use them:** test one embedding model at a time for
  evidence retrieval, then test GPT-4/PaLM/Llama2-70B as readers over gold evidence.
- **What they measure:** retrieval Hit@k plus answer EM/F1. Headline
  is that all three LLMs do poorly, so the gap is retrieval plus cross-evidence reasoning. Closest production-style foil to 2Wiki.

### CRUD-RAG (2024) — Chinese RAG beyond QA

[Lyu et al., “CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation”](https://arxiv.org/abs/2401.17043)

- **What the paper is about:** QA-only sets miss real RAG uses. Splits
  work into Create (write new text), Read (knowledge QA), Update (fix errors), Delete (summarize).
- **Benchmarks used:** large Chinese sets per CRUD type.
- **How they use them:** vary retriever, knowledge-base build, context
  length, and LLM jointly per task type.
- **What they measure:** BLEU/ROUGE/EM per CRUD type. Tests the whole
  pipeline, not hop count.

### RAGBench (2024) — industry faithfulness

[Friel et al., “RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems”](https://arxiv.org/abs/2407.11005)

- **What the paper is about:** production RAG needs explainable
  quality labels. Releases 100k examples over 5 industry domains (user manuals and similar).
- **Benchmarks used:** its own industry corpora with provided passages
  per query.
- **How they use them:** score any retriever plus LLM stack with the
  TRACe label set.
- **What they measure:** TRACe: context relevance, utilization,
  completeness, adherence. Finds a fine-tuned RoBERTa judge beats LLM judges. Tests citation faithfulness, not multi-hop chaining.

## What this means for RecRAG

- Use 2WikiMultiHopQA papers for retrieval and answer-quality baselines.
- Treat HippoRAG as the closest published graph-retrieval reference.
- Only HippoRAG among GraphRAG architectures uses 2Wiki. GraphRAG,
  LightRAG, RAPTOR, G-Retriever, and ToG each bring their own corpus and judge or accuracy metrics, so do not compare their headline numbers with 2Wiki EM/F1.
- Keep corpus construction separate in reports: public papers use sampled QA
  passages, Wikipedia, or LongBench conversions; RecRAG stores graph data in Neo4j and also tests filing-style corpora.
- Report both answer metrics and retrieval metrics when possible. The papers
  commonly use exact match/F1 for answers and recall or Recall@k for evidence.
