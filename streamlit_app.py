from __future__ import annotations

from time import perf_counter

import streamlit as st

from hybridrag.construction.construction import ConstructionMethod
from hybridrag.dataset_records.two_wiki_multihopqa import load_records
from hybridrag.hybrid_rag import HybridRAG
from hybridrag.retrieval.answering import RETRIEVAL_METHODS

st.set_page_config(page_title="GraphRAG Demo", page_icon=":material/account_tree:")
st.title("GraphRAG Demo")


@st.cache_data
def load_records_for_ui(limit: int):
    return load_records(limit)


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
                st.metric(name.replace("_", " ").capitalize(), value)
                if metric.get("reason"):
                    st.caption(str(metric["reason"]))


def display_method(method):
    return str(method).replace("_", " ").title()


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


record_count = st.slider("Records to load", 1, 20, 2)
selected_construction_methods = st.pills(
    "Construction methods",
    list(ConstructionMethod),
    default=list(ConstructionMethod),
    selection_mode="multi",
    format_func=display_method,
) or []
selected_retrieval_methods = st.pills(
    "Retrieval methods",
    RETRIEVAL_METHODS,
    default=RETRIEVAL_METHODS,
    selection_mode="multi",
    format_func=display_method,
) or []
score_with_deepeval = st.checkbox(
    "Score with DeepEval",
    value=True,
    help="Scores graph construction, retrieval, and answers with LLM judges.",
)
loaded_records = load_records_for_ui(record_count)

run_clicked = st.button("Run", type="primary", icon=":material/account_tree:")

if run_clicked:
    if not selected_construction_methods:
        st.warning("Select at least one construction method.")
    elif not selected_retrieval_methods:
        st.warning("Select at least one retrieval method.")
    else:
        if score_with_deepeval:
            try:
                from hybridrag.evals.construction import (
                    evaluate_answer,
                    evaluate_construction,
                    read_graph_statistics,
                    read_entity_triples,
                )
                from hybridrag.evals.retrieval import evaluate_retrieval
            except ImportError:
                st.error("DeepEval scoring requires `uv sync --extra experiments`.")
                st.stop()

        try:
            rag = HybridRAG.from_config()
            try:
                for record_number, record in enumerate(loaded_records, start=1):
                    construction_evaluations = []
                    construction_status = st.status(
                        f"Record {record_number}: constructing graphs", expanded=False
                    )
                    for method in selected_construction_methods:
                        database = method.database_name(record.id)
                        construction_status.write(
                            f"Building {display_method(method)} graph for record {record_number}"
                        )
                        construction_started = perf_counter()
                        rag.construct(
                            record.pages,
                            method=method,
                            database=database,
                        )
                        construction_seconds = perf_counter() - construction_started
                        evaluation = None
                        if score_with_deepeval:
                            try:
                                evaluation = evaluate_construction(
                                    record,
                                    read_entity_triples(rag._driver, database=database),
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
                        construction_evaluations.append(evaluation)
                        construction_status.write(
                            f"Graph complete: {display_method(method)}, record {record_number}"
                        )
                    construction_status.update(
                        label="Construction complete", state="complete"
                    )
                    st.subheader(f"Record {record_number}")
                    st.write(record.question)
                    st.markdown("### Construction")
                    for method, evaluation in zip(
                        selected_construction_methods, construction_evaluations
                    ):
                        with st.container(border=True):
                            st.markdown(f"#### {display_method(method)}")
                            render_construction_evaluation(evaluation)

                    st.markdown("### Retrieval and answer")
                    retrieval_status = st.status(
                        f"Record {record_number}: retrieving answers", expanded=False
                    )
                    for method in selected_construction_methods:
                        retrieval_status.write(
                            f"Answering record {record_number} with {display_method(method)} graph"
                        )
                        results = rag.answer(
                            record.question,
                            database=method.database_name(record.id),
                            retrieval_methods=selected_retrieval_methods,
                        )
                        st.markdown(f"#### {display_method(method)}")
                        for retrieval_method, result in zip(
                            selected_retrieval_methods, results
                        ):
                            retrieval_evaluation = (
                                evaluate_retrieval(record, result)
                                if score_with_deepeval
                                else None
                            )
                            answer_evaluation = (
                                evaluate_answer(record, result)
                                if score_with_deepeval
                                else None
                            )
                            with st.container(border=True):
                                st.markdown(f"**{display_method(retrieval_method)}**")
                                if retrieval_evaluation is not None:
                                    retrieval = retrieval_evaluation
                                    st.markdown("**Retrieval evaluation**")
                                    st.caption(f"DeepEval on top {retrieval['top_k']} results")
                                    render_metric_group(
                                        "Retrieval judge",
                                        retrieval["deepeval"],
                                        [
                                            "contextual_precision",
                                            "contextual_recall",
                                            "contextual_relevancy",
                                        ],
                                    )
                                st.markdown("**Answer**")
                                st.write(result.answer)
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
