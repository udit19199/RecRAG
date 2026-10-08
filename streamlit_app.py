from __future__ import annotations

import json
import logging
import tomllib
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import streamlit as st
from langchain_core.callbacks import get_usage_metadata_callback
from neo4j_graphrag.schema import get_schema

from benchmark import NEWS_QA_INSTRUCTION, NOVELQA_QA_INSTRUCTION, QA_INSTRUCTION
from cost import Pricing, measure
from dataset import DATASET_SOURCES, get_source, load_hotpotqa_corpus
from evals.construction import read_graph_statistics
from evals.qa_metrics import (
    GraphRAGRecordResult,
    benchmark_prompt,
    hotpotqa_f1,
    score_novelqa_answer,
    score_qa_answer,
    summarize_qa_results,
)
from graphrag_construction import CONSTRUCTION_METHOD, database_name
from graphrag_retrieval import (
    RETRIEVAL_METHOD,
    GraphRAG,
    build_agentic_retriever,
)

st.set_page_config(page_title="GraphRAG Demo", page_icon=":material/account_tree:")
st.title("GraphRAG Demo")

CONFIG = tomllib.loads((Path(__file__).resolve().parent / "config.toml").read_text())


def load_dataset_record(dataset_name: str, index: int):
    return get_source(dataset_name).load_record(index)


load_record_cached = st.cache_data(load_dataset_record)


def render_token_usage(title, usage):
    if usage is None:
        return
    st.caption(title)
    st.write(usage.usage_metadata)


def store_token_usage(path: Path, *, record_id, stage, usage, **context):
    event = {"record_id": record_id, "stage": stage, **context}
    event["usage"] = usage.usage_metadata
    with path.open("a", encoding="utf-8") as file:
        json.dump(event, file)
        file.write("\n")


def source_key(pages):
    content = "\n".join(
        f"{page.title}\n{chr(10).join(page.passages)}" for page in pages
    )
    return sha256(content.encode()).hexdigest()


def render_graph_statistics(statistics, construction_seconds):
    with st.container(horizontal=True):
        st.metric("Build time", f"{construction_seconds:.1f}s", border=True)
        if statistics:
            st.metric(
                "Graph size",
                f"{statistics.get('entity_count', 0)} entities, "
                f"{statistics.get('relationship_count', 0)} relationships",
                border=True,
            )
    if statistics:
        with st.expander("Graph details"):
            st.write(
                {
                    "Relation types": statistics.get("relation_type_count", 0),
                    "Duplicate entity name groups": statistics.get(
                        "duplicate_entity_name_groups", 0
                    ),
                    "Duplicate entity nodes": statistics.get(
                        "duplicate_entity_nodes", 0
                    ),
                    "Isolated entities": statistics.get("isolated_entity_count", 0),
                    "Self-loops": statistics.get("self_loop_count", 0),
                }
            )


selected_dataset = st.selectbox(
    "Dataset",
    DATASET_SOURCES,
    format_func=lambda source: source.display_name,
)
sample_count = st.number_input(
    "Records to compare", min_value=1, max_value=20, value=10, step=1
)
run_retrieval = st.checkbox("Run retrieval and answer", value=True)
try:
    loaded_records = [
        load_record_cached(selected_dataset.name, index)
        for index in range(sample_count)
    ]
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

run_clicked = st.button("Run", type="primary", icon=":material/account_tree:")

if run_clicked:
    try:
        project_root = Path(__file__).resolve().parent
        run_id = f"{datetime.now(UTC):%Y%m%d%H%M%S%f}"
        usage_path = project_root / "runs" / f"usage-{run_id}Z.jsonl"
        results_path = project_root / "runs" / f"graphrag-{run_id}Z.jsonl"
        summary_path = project_root / "runs" / f"graphrag-summary-{run_id}Z.json"
        usage_path.parent.mkdir(exist_ok=True)
        st.caption(f"Token usage saved to `{usage_path.relative_to(project_root)}`")
        st.caption(
            f"Baseline scores saved to `{results_path.relative_to(project_root)}`"
        )
        shared_pages = (
            load_hotpotqa_corpus() if selected_dataset.name == "hotpotqa" else None
        )
        shared_key = source_key(shared_pages) if shared_pages is not None else None
        if shared_pages is not None:
            st.info(
                "HotpotQA uses one shared graph over "
                f"{len(shared_pages):,} unique pages from the loaded validation rows."
            )
        rag = GraphRAG.from_config()
        constructed_databases = set()
        baseline_results: list[GraphRAGRecordResult] = []
        try:
            for record_number, record in enumerate(loaded_records, start=1):
                graph_pages = shared_pages if shared_pages is not None else record.pages
                graph_key = (
                    shared_key if shared_key is not None else source_key(record.pages)
                )
                graph_scope = (
                    "HotpotQA validation corpus"
                    if shared_pages is not None
                    else f"record {record_number}"
                )
                method = CONSTRUCTION_METHOD
                database = database_name(graph_key, run_id)
                construction_status = st.status(
                    f"{graph_scope}: constructing graph", expanded=False
                )
                if database in constructed_databases:
                    construction_usage = None
                    construction_seconds = 0.0
                    construction_status.write(
                        f"Reusing {method} graph for {graph_scope}"
                    )
                else:
                    construction_status.write(
                        f"Building {method} graph for {graph_scope}"
                    )
                    construction_started = perf_counter()
                    with get_usage_metadata_callback() as construction_usage:
                        rag.construct(graph_pages, database=database)
                    construction_seconds = perf_counter() - construction_started
                    constructed_databases.add(database)
                    store_token_usage(
                        usage_path,
                        record_id=(
                            "hotpotqa-validation-corpus"
                            if shared_pages is not None
                            else record.id
                        ),
                        stage="graphrag_construction",
                        usage=construction_usage,
                        dataset=selected_dataset.name,
                        question_type=(
                            None
                            if shared_pages is not None
                            else getattr(record, "question_type", None)
                        ),
                        construction_method=method,
                        elapsed_seconds=construction_seconds,
                    )
                statistics = read_graph_statistics(rag._driver, database=database)
                construction_status.write(f"Graph complete: {method}, {graph_scope}")
                construction_status.update(
                    label="Construction complete", state="complete"
                )
                st.subheader(f"Record {record_number}")
                st.write(record.question)
                st.markdown("### Construction")
                with st.container(border=True):
                    st.markdown(f"#### {method}")
                    render_graph_statistics(statistics, construction_seconds)
                    render_token_usage("Construction tokens", construction_usage)

                if not run_retrieval:
                    continue
                st.markdown("### Retrieval and answer")
                retrieval_status = st.status(
                    f"Record {record_number}: retrieving answers", expanded=False
                )
                retrieval_status.write(
                    f"Answering record {record_number} with {method} graph"
                )
                st.markdown(f"#### {method}")
                retrieval_started = perf_counter()
                with (
                    measure(Pricing.model_validate(CONFIG["pricing"])) as question_cost,
                    get_usage_metadata_callback() as retrieval_usage,
                ):
                    search_llm = rag._answer_llm
                    retriever = build_agentic_retriever(
                        rag._driver,
                        rag._llm,
                        rag._embedder,
                        database,
                        search_llm,
                        neo4j_schema=get_schema(rag._driver, database=database),
                    )
                    retrieval_result = retriever.search(record.question, top_k=5)
                    items = retrieval_result.items
                    retrieved_context = [
                        str(item.content) for item in retrieval_result.items
                    ]
                    answer_prompt = benchmark_prompt(
                        "\n\n".join(retrieved_context), record.question
                    )
                    answer = search_llm.invoke(
                        [
                            {
                                "role": "system",
                                "content": (
                                    NEWS_QA_INSTRUCTION
                                    if selected_dataset.name == "multihop_rag"
                                    else NOVELQA_QA_INSTRUCTION
                                    if selected_dataset.name == "novelqa"
                                    else QA_INSTRUCTION
                                ),
                            },
                            {"role": "user", "content": answer_prompt},
                        ]
                    ).text
                retrieval_and_answer_seconds = perf_counter() - retrieval_started
                prompt = benchmark_prompt(
                    "\n\n".join(retrieved_context), record.question
                )
                expected_answers = list(record.answer_aliases())
                canonical_answer = record.answer
                answer_metrics = (
                    score_novelqa_answer(answer, canonical_answer, prompt)
                    if selected_dataset.name == "novelqa"
                    and canonical_answer is not None
                    else score_qa_answer(
                        answer,
                        expected_answers,
                        canonical_answer,
                        prompt,
                        multihop_rag=selected_dataset.name == "multihop_rag",
                    )
                    if canonical_answer is not None and expected_answers
                    else None
                )
                if selected_dataset.name == "hotpotqa" and answer_metrics is not None:
                    precision, recall, f1 = hotpotqa_f1(answer, expected_answers)
                    answer_metrics = answer_metrics.model_copy(
                        update={"precision": precision, "recall": recall, "f1": f1}
                    )
                baseline_result = GraphRAGRecordResult(
                    run_id=run_id,
                    dataset=selected_dataset.name,
                    record_id=record.id,
                    question_type=getattr(record, "question_type", None),
                    construction_method=method,
                    retrieval_method=RETRIEVAL_METHOD,
                    question=record.question,
                    expected_answers=expected_answers,
                    answer=answer,
                    retrieved_context=retrieved_context,
                    metrics=answer_metrics,
                    construction_seconds=construction_seconds,
                    retrieval_and_answer_seconds=retrieval_and_answer_seconds,
                    question_cost=question_cost,
                    estimated_question_cost_usd=question_cost.estimated_usd,
                    complexity=getattr(record, "complexity", None),
                )
                baseline_results.append(baseline_result)
                with results_path.open("a", encoding="utf-8") as file:
                    file.write(baseline_result.model_dump_json())
                    file.write("\n")
                store_token_usage(
                    usage_path,
                    record_id=record.id,
                    stage="retrieval_and_answer",
                    usage=retrieval_usage,
                    dataset=selected_dataset.name,
                    question_type=getattr(record, "question_type", None),
                    construction_method=method,
                    retrieval_method=RETRIEVAL_METHOD,
                    elapsed_seconds=retrieval_and_answer_seconds,
                    estimated_question_cost_usd=question_cost.estimated_usd,
                )
                with st.container(border=True):
                    st.markdown(f"**{RETRIEVAL_METHOD}**")
                    st.markdown("**Answer**")
                    st.write(answer)
                    st.markdown("**Baseline QA metrics**")
                    metrics = baseline_result.metrics
                    if metrics is None:
                        st.caption(
                            "Not scored. The NovelQA input files used here do not "
                            "include gold answers."
                        )
                    else:
                        with st.container(horizontal=True):
                            st.metric("Exact match", f"{metrics.exact_match:.0%}")
                            st.metric("Precision", f"{metrics.precision:.0%}")
                            st.metric("Recall", f"{metrics.recall:.0%}")
                            st.metric("F1", f"{metrics.f1:.0%}")
                            st.metric(
                                "Paper overlap accuracy",
                                f"{metrics.overlap_accuracy:.0%}",
                            )
                            st.metric(
                                "Gold answer in prompt",
                                f"{metrics.retrieval_accuracy:.0%}",
                            )
                    st.metric(
                        "Retrieval and answer time",
                        f"{retrieval_and_answer_seconds:.2f}s",
                    )
                    render_token_usage("Retrieval and answer tokens", retrieval_usage)
                    st.metric(
                        "Question API cost",
                        f"${question_cost.estimated_usd:.6f}"
                        if question_cost.estimated_usd is not None
                        else "Unavailable",
                    )
                    st.caption(
                        "Includes retrieval and answer API calls. Excludes indexing and machine cost."
                    )
                    with st.expander(f"Retrieved context ({len(items)} items)"):
                        for rank, item in enumerate(items, start=1):
                            st.markdown(f"**Rank {rank}**")
                            st.write(item.content)
                retrieval_status.update(label="Retrieval complete", state="complete")
        finally:
            rag.close()
        summary = summarize_qa_results(run_id, selected_dataset.name, baseline_results)
        summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        st.caption(
            f"Baseline summary saved to `{summary_path.relative_to(project_root)}`"
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("GraphRAG run failed")
        st.error(f"{type(exc).__name__}: {exc}")
