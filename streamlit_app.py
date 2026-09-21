from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import streamlit as st
from langchain_core.callbacks import get_usage_metadata_callback

from graphrag.construction import ConstructionMethod
from graphrag.dataset_adapters.registry import DATASET_SOURCES, get_source
from graphrag.graph_rag import GraphRAG
from graphrag.retrieval.answering import RETRIEVAL_METHODS

st.set_page_config(page_title="GraphRAG Demo", page_icon=":material/account_tree:")
st.title("GraphRAG Demo")


def load_dataset_record(dataset_name: str, index: int):
    return get_source(dataset_name).load_record(index)


load_record_cached = st.cache_data(load_dataset_record)


def render_metric_group(title, metrics, names):
    if title:
        st.markdown(f"**{title}**")
    if not any(name in metrics for name in names):
        if metrics.get("reason"):
            st.caption(str(metrics["reason"]))
        return
    with st.container(horizontal=True):
        for name in names:
            metric = metrics.get(name)
            if not isinstance(metric, dict):
                continue
            score = metric.get("score")
            value = "Not scored" if score is None else f"{float(score):.0%}"
            with st.container(border=True):
                st.metric(name, value)
                if metric.get("reason"):
                    st.caption(str(metric["reason"]))


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


def render_construction_evaluation(evaluation):
    if not evaluation:
        st.caption("Evaluation not run.")
        return
    metrics = evaluation.get("deepeval", {})
    metric_labels = {
        "groundedness": "Source support",
        "completeness": "Source coverage",
        "supporting_evidence_coverage": "Supporting passages",
    }
    with st.container(horizontal=True):
        for name, label in metric_labels.items():
            metric = metrics.get(name, {})
            score = metric.get("score")
            value = "Not scored" if score is None else f"{float(score):.0%}"
            with st.container(border=True):
                st.metric(label, value)
                st.caption(str(metric.get("description") or ""))
                if metric.get("reason"):
                    with st.expander("Why this score?"):
                        st.write(str(metric["reason"]))

    render_token_usage("Construction judge tokens", evaluation.get("token_usage"))

    statistics = evaluation.get("graph_statistics", {})
    construction_seconds = evaluation.get("construction_seconds")
    with st.container(horizontal=True):
        if construction_seconds is not None:
            st.metric("Build time", f"{float(construction_seconds):.1f}s", border=True)
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
    if evaluation.get("error"):
        st.warning(str(evaluation["error"]))


selected_dataset = st.selectbox(
    "Dataset",
    DATASET_SOURCES,
    format_func=lambda source: source.display_name,
)
record_count = st.slider("Records to load", 1, 20, 2)
selected_construction_methods = (
    st.pills(
        "Construction methods",
        list(ConstructionMethod),
        default=list(ConstructionMethod),
        selection_mode="multi",
    )
    or []
)
selected_retrieval_methods = (
    st.pills(
        "Retrieval methods",
        RETRIEVAL_METHODS,
        default=RETRIEVAL_METHODS,
        selection_mode="multi",
    )
    or []
)
score_with_deepeval = st.checkbox(
    "Score with DeepEval",
    value=True,
    help="Scores graph construction, retrieval, and answers with LLM judges.",
)
try:
    loaded_records = [
        load_record_cached(selected_dataset.name, index)
        for index in range(record_count)
    ]
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

run_clicked = st.button("Run", type="primary", icon=":material/account_tree:")

if run_clicked:
    if not selected_construction_methods:
        st.warning("Select at least one construction method.")
    elif not selected_retrieval_methods:
        st.warning("Select at least one retrieval method.")
    else:
        if score_with_deepeval:
            try:
                from graphrag.evals.construction import (
                    evaluate_answer,
                    evaluate_construction,
                    read_entity_triples,
                    read_graph_statistics,
                )
                from graphrag.evals.retrieval import evaluate_retrieval
            except ImportError:
                st.error("DeepEval scoring requires `uv sync --extra experiments`.")
                st.stop()

        try:
            project_root = Path(__file__).resolve().parent
            usage_path = (
                project_root
                / "runs"
                / (f"usage-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.jsonl")
            )
            usage_path.parent.mkdir(exist_ok=True)
            st.caption(f"Token usage saved to `{usage_path.relative_to(project_root)}`")
            rag = GraphRAG.from_config()
            try:
                for record_number, record in enumerate(loaded_records, start=1):
                    construction_evaluations = []
                    construction_usages = []
                    construction_status = st.status(
                        f"Record {record_number}: constructing graphs", expanded=False
                    )
                    for method in selected_construction_methods:
                        database = method.database_name(record.id)
                        construction_status.write(
                            f"Building {method} graph for record {record_number}"
                        )
                        construction_started = perf_counter()
                        with get_usage_metadata_callback() as construction_usage:
                            rag.construct(
                                record.pages,
                                method=method,
                                database=database,
                            )
                        store_token_usage(
                            usage_path,
                            record_id=record.id,
                            stage="construction",
                            usage=construction_usage,
                            construction_method=method.value,
                        )
                        construction_seconds = perf_counter() - construction_started
                        evaluation = None
                        if score_with_deepeval:
                            try:
                                with get_usage_metadata_callback() as evaluation_usage:
                                    evaluation = evaluate_construction(
                                        record,
                                        read_entity_triples(
                                            rag._driver, database=database
                                        ),
                                        graph_statistics=read_graph_statistics(
                                            rag._driver, database=database
                                        ),
                                        construction_seconds=construction_seconds,
                                    )
                            except Exception as exc:
                                evaluation = {
                                    "construction_seconds": construction_seconds,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            else:
                                evaluation["token_usage"] = evaluation_usage
                            store_token_usage(
                                usage_path,
                                record_id=record.id,
                                stage="construction_evaluation",
                                usage=evaluation_usage,
                                construction_method=method.value,
                            )
                        construction_evaluations.append(evaluation)
                        construction_usages.append(construction_usage)
                        construction_status.write(
                            f"Graph complete: {method}, record {record_number}"
                        )
                    construction_status.update(
                        label="Construction complete", state="complete"
                    )
                    st.subheader(f"Record {record_number}")
                    st.write(record.question)
                    st.markdown("### Construction")
                    for method, evaluation, construction_usage in zip(
                        selected_construction_methods,
                        construction_evaluations,
                        construction_usages,
                    ):
                        with st.container(border=True):
                            st.markdown(f"#### {method}")
                            render_construction_evaluation(evaluation)
                            render_token_usage(
                                "Construction tokens", construction_usage
                            )

                    st.markdown("### Retrieval and answer")
                    retrieval_status = st.status(
                        f"Record {record_number}: retrieving answers", expanded=False
                    )
                    for method in selected_construction_methods:
                        retrieval_status.write(
                            f"Answering record {record_number} with {method} graph"
                        )
                        st.markdown(f"#### {method}")
                        for retrieval_method in selected_retrieval_methods:
                            with get_usage_metadata_callback() as retrieval_usage:
                                result = rag.answer(
                                    record.question,
                                    database=method.database_name(record.id),
                                    retrieval_methods=[retrieval_method],
                                )[0]
                            store_token_usage(
                                usage_path,
                                record_id=record.id,
                                stage="retrieval_and_answer",
                                usage=retrieval_usage,
                                construction_method=method.value,
                                retrieval_method=retrieval_method,
                            )
                            retrieval_evaluation = None
                            if score_with_deepeval:
                                with get_usage_metadata_callback() as evaluation_usage:
                                    retrieval_evaluation = evaluate_retrieval(
                                        record, result
                                    )
                                retrieval_evaluation["token_usage"] = evaluation_usage
                                store_token_usage(
                                    usage_path,
                                    record_id=record.id,
                                    stage="retrieval_evaluation",
                                    usage=evaluation_usage,
                                    construction_method=method.value,
                                    retrieval_method=retrieval_method,
                                )
                            answer_evaluation = None
                            if score_with_deepeval:
                                with get_usage_metadata_callback() as evaluation_usage:
                                    answer_evaluation = evaluate_answer(record, result)
                                answer_evaluation["token_usage"] = evaluation_usage
                                store_token_usage(
                                    usage_path,
                                    record_id=record.id,
                                    stage="answer_evaluation",
                                    usage=evaluation_usage,
                                    construction_method=method.value,
                                    retrieval_method=retrieval_method,
                                )
                            with st.container(border=True):
                                st.markdown(f"**{retrieval_method}**")
                                if retrieval_evaluation is not None:
                                    retrieval = retrieval_evaluation
                                    st.markdown("**Retrieval evaluation**")
                                    st.caption(
                                        f"DeepEval on top {retrieval['top_k']} results"
                                    )
                                    render_metric_group(
                                        "Retrieval judge",
                                        retrieval["deepeval"],
                                        [
                                            "contextual_precision",
                                            "contextual_recall",
                                            "contextual_relevancy",
                                        ],
                                    )
                                    render_token_usage(
                                        "Retrieval judge tokens",
                                        retrieval["token_usage"],
                                    )
                                st.markdown("**Answer**")
                                st.write(result.answer)
                                render_token_usage(
                                    "Retrieval and answer tokens", retrieval_usage
                                )
                                if answer_evaluation is not None:
                                    answer = answer_evaluation["answer"]
                                    st.markdown("**Answer evaluation**")
                                    with st.container(border=True):
                                        st.markdown("**Expected answer**")
                                        st.write(answer["expected"])
                                        st.metric(
                                            "Exact answer or alias match",
                                            "Yes" if answer["alias_match"] else "No",
                                        )
                                    render_metric_group(
                                        "Answer judge", answer, ["faithfulness"]
                                    )
                                    render_metric_group(
                                        None,
                                        answer["deepeval"],
                                        ["relevancy", "correctness"],
                                    )
                                    render_token_usage(
                                        "Answer judge tokens",
                                        answer_evaluation["token_usage"],
                                    )
                                items = (
                                    result.retriever_result.items
                                    if result.retriever_result is not None
                                    else []
                                )
                                with st.expander(
                                    f"Retrieved context ({len(items)} items)"
                                ):
                                    for rank, item in enumerate(items, start=1):
                                        st.markdown(f"**Rank {rank}**")
                                        st.write(item.content)
                    retrieval_status.update(
                        label="Retrieval complete", state="complete"
                    )
            finally:
                rag.close()
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
