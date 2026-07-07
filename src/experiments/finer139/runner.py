"""Orchestrate a FiNER-139 benchmark run across the selected methods.

Pure, importable entry point (`run_benchmark`) used by the orchestrator's
experiments router. Emits progress via a callback and returns a JSON-serializable
results dict. All heavy imports (datasets/spaCy) happen lazily inside the engine.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from experiments.finer139.methods import (
    ALL_METHODS,
    METHOD_DISPLAY_NAMES,
    BaseExtractor,
    DynamicExtractor,
    HybridExtractor,
    LLMExtractor,
    OntologyExtractor,
    SpacyExtractor,
    dedup_spans,
)
from experiments.finer139.analysis import (
    aggregate_multi_seed,
    analyze,
    compare_methods_head_to_head,
    compare_methods_paired,
    extended_to_dict,
)
from experiments.finer139.scoring import (
    metrics_to_dict,
    numeric_filter,
)
from experiments.finer139.types import Sentence, Span

logger = logging.getLogger(__name__)

ProgressCb = Callable[[str, int, int, str], None] | None
LLM_METHODS = {"llm", "hybrid", "dynamic"}
DEFAULT_SPACY_MODEL = "en_core_web_sm"
# FiNER-139 research default: OpenAI for all LLM-based extractors (D-agnostic).
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL = "gpt-4o-mini"


@dataclass
class RunParams:
    sample_size: int = 100
    seed: int = 42
    seeds: list[int] | None = None
    split: str = "validation"
    stratified: bool = False
    suite_path: str | None = None
    methods: list[str] = field(default_factory=lambda: list(ALL_METHODS))
    provider: str | None = DEFAULT_LLM_PROVIDER
    model: str | None = DEFAULT_LLM_MODEL
    max_examples: int = 15
    concurrency: int = 6
    spacy_model: str = DEFAULT_SPACY_MODEL


def _resolve_seeds(params: RunParams) -> list[int]:
    if params.seeds:
        return list(params.seeds)
    return [params.seed]


def _load_sentences(
    params: RunParams,
    ds_mod: Any,
    seed: int,
) -> tuple[list[Sentence], dict[str, Any]]:
    if params.suite_path:
        sentences, spec = ds_mod.load_frozen_suite(params.suite_path)
        dataset_meta = {
            "id": ds_mod.DATASET_ID,
            "split": spec.get("split", params.split),
            "suite_path": params.suite_path,
            "suite_version": spec.get("version"),
            "num_sentences": len(sentences),
            "num_gold_entities": sum(len(s.gold_spans) for s in sentences),
            "frozen": True,
        }
        return sentences, dataset_meta

    sentences = ds_mod.load_sample(
        params.sample_size,
        seed,
        split=params.split,
        stratified=params.stratified,
    )
    dataset_meta = {
        "id": ds_mod.DATASET_ID,
        "split": params.split,
        "num_sentences": len(sentences),
        "num_gold_entities": sum(len(s.gold_spans) for s in sentences),
        "stratified": params.stratified,
        "frozen": False,
    }
    return sentences, dataset_meta


def _noop(stage: str, current: int, total: int, message: str) -> None:
    return None


def _build_llm(provider: str | None, model: str | None) -> Any:
    from adapters import create_llm_from_config
    from config import find_config_path, load_config

    config = load_config(find_config_path())
    return create_llm_from_config(
        config,
        provider=provider or DEFAULT_LLM_PROVIDER,
        model=model or DEFAULT_LLM_MODEL,
        temperature=0,
    )


def _run_stateless(
    extractor: BaseExtractor,
    sentences: list[Sentence],
    report: Callable[[int, int], None],
    concurrency: int,
) -> list[list[Span]]:
    """Run a stateless extractor over all sentences.

    The first prediction runs synchronously so configuration errors (missing
    key, missing spaCy model) surface immediately and fail the whole method.
    Remaining LLM predictions may run concurrently; per-call errors degrade to
    an empty prediction rather than aborting the method.
    """
    n = len(sentences)
    preds: list[list[Span]] = [[] for _ in range(n)]
    if n == 0:
        return preds

    preds[0] = extractor.predict(sentences[0])
    report(1, n)

    if extractor.uses_llm and concurrency > 1 and n > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(extractor.predict, sentences[i]): i for i in range(1, n)
            }
            done = 1
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    preds[i] = fut.result()
                except Exception:  # noqa: BLE001 - degrade single call
                    logger.warning("prediction failed for sentence %d", i, exc_info=True)
                    preds[i] = []
                done += 1
                report(done, n)
    else:
        for i in range(1, n):
            try:
                preds[i] = extractor.predict(sentences[i])
            except Exception:  # noqa: BLE001 - degrade single call
                logger.warning("prediction failed for sentence %d", i, exc_info=True)
                preds[i] = []
            report(i + 1, n)
    return preds


def run_benchmark(params: RunParams, progress_cb: ProgressCb = None) -> dict[str, Any]:
    seeds = _resolve_seeds(params)
    if len(seeds) > 1:
        return _run_multi_seed_benchmark(params, seeds, progress_cb)
    return _run_single_benchmark(params, seeds[0], progress_cb)


def _run_multi_seed_benchmark(
    params: RunParams,
    seeds: list[int],
    progress_cb: ProgressCb,
) -> dict[str, Any]:
    report = progress_cb or _noop
    per_seed_runs: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        report(
            "running",
            i,
            len(seeds),
            f"Multi-seed run {i + 1}/{len(seeds)} (seed={seed})",
        )
        per_seed_runs.append(_run_single_benchmark(params, seed, None))

    aggregate = aggregate_multi_seed(per_seed_runs)
    primary = per_seed_runs[0]
    return {
        "params": {
            **primary["params"],
            "seeds": seeds,
            "multi_seed": True,
        },
        "dataset": primary["dataset"],
        "llm": primary["llm"],
        "multi_seed_aggregate": aggregate,
        "per_seed_runs": per_seed_runs,
        "evaluation": {
            **primary["evaluation"],
            "multi_seed": True,
            "n_seeds": len(seeds),
        },
    }


def _run_single_benchmark(
    params: RunParams,
    seed: int,
    progress_cb: ProgressCb,
) -> dict[str, Any]:
    report = progress_cb or _noop

    report("loading", 0, 0, "Loading FiNER-139 sample...")
    from experiments.finer139 import dataset as ds_mod
    from experiments.finer139.schema import build_gazetteer, build_gazetteer_regex

    sentences, dataset_meta = _load_sentences(params, ds_mod, seed)
    label_names = ds_mod.get_label_names()
    concept_names = ds_mod.concept_names_from_labels(label_names)
    gazetteer_regex = build_gazetteer_regex(build_gazetteer(concept_names))

    selected = [m for m in ALL_METHODS if m in set(params.methods)]
    if not selected:
        selected = list(ALL_METHODS)

    needs_llm = any(m in LLM_METHODS for m in selected)
    llm: Any = None
    llm_error: str | None = None
    if needs_llm:
        try:
            llm = _build_llm(params.provider, params.model)
        except Exception as exc:  # noqa: BLE001
            llm_error = str(exc)
            logger.warning("LLM unavailable: %s", exc)

    all_preds: dict[str, list[list[Span]]] = {}
    hybrid_preds: list[list[Span]] | None = None
    method_results: list[dict[str, Any]] = []

    total_methods = len(selected)
    for m_idx, name in enumerate(selected):
        display = METHOD_DISPLAY_NAMES.get(name, name)

        def make_report(method_name: str, method_display: str) -> Callable[[int, int], None]:
            def _r(cur: int, tot: int) -> None:
                report(
                    "running",
                    cur,
                    tot,
                    f"[{m_idx + 1}/{total_methods}] {method_display}: {cur}/{tot}",
                )

            return _r

        step_report = make_report(name, display)
        report("running", 0, len(sentences), f"[{m_idx + 1}/{total_methods}] {display}")

        error: str | None = None
        llm_calls = 0
        preds: list[list[Span]] = [[] for _ in sentences]
        t0 = time.perf_counter()
        try:
            if name == "llm":
                if llm is None:
                    raise RuntimeError(llm_error or "LLM unavailable")
                preds = _run_stateless(
                    LLMExtractor(llm), sentences, step_report, params.concurrency
                )
                llm_calls = len(sentences)
            elif name == "nlp":
                preds = _run_stateless(
                    SpacyExtractor(params.spacy_model), sentences, step_report, 1
                )
            elif name == "ontology":
                preds = _run_stateless(
                    OntologyExtractor(gazetteer_regex), sentences, step_report, 1
                )
            elif name == "hybrid":
                if llm is None:
                    raise RuntimeError(llm_error or "LLM unavailable")
                preds = _run_stateless(
                    HybridExtractor(llm, concept_names),
                    sentences,
                    step_report,
                    params.concurrency,
                )
                hybrid_preds = preds
                llm_calls = len(sentences)
            elif name == "dynamic":
                if llm is None:
                    raise RuntimeError(llm_error or "LLM unavailable")
                base = hybrid_preds
                if base is None:
                    base = _run_stateless(
                        HybridExtractor(llm, concept_names),
                        sentences,
                        step_report,
                        params.concurrency,
                    )
                    llm_calls = len(sentences)
                dyn = DynamicExtractor()
                preds = [
                    dyn.predict_incremental(s, base[i]) for i, s in enumerate(sentences)
                ]
        except Exception as exc:  # noqa: BLE001 - record and continue
            error = str(exc)
            logger.warning("method '%s' failed: %s", name, exc, exc_info=True)

        latency = time.perf_counter() - t0

        entry: dict[str, Any] = {
            "name": name,
            "display_name": display,
            "uses_llm": name in LLM_METHODS,
            "latency_s": round(latency, 3),
            "llm_calls": llm_calls,
            "error": error,
        }
        if error is None:
            all_preds[name] = preds
            ext = analyze(sentences, preds, seed=seed)
            entry.update(
                {
                    "strict": metrics_to_dict(ext.strict),
                    "relaxed": metrics_to_dict(ext.relaxed),
                    "partial": metrics_to_dict(ext.partial),
                    "macro_strict": metrics_to_dict(ext.macro_strict),
                    "macro_partial": metrics_to_dict(ext.macro_partial),
                    "num_pred": ext.strict.tp + ext.strict.fp,
                    "num_gold": ext.strict.tp + ext.strict.fn,
                    "diagnostics": extended_to_dict(ext),
                }
            )
        method_results.append(entry)

    report("scoring", total_methods, total_methods, "Building examples...")
    examples = _build_examples(sentences, all_preds, params.max_examples)

    report("complete", total_methods, total_methods, "Done")

    comparison = compare_methods_head_to_head(sentences, all_preds) if all_preds else {}
    paired = compare_methods_paired(sentences, all_preds, seed=seed) if len(all_preds) >= 2 else []

    return {
        "params": {
            "sample_size": params.sample_size,
            "seed": seed,
            "split": params.split,
            "stratified": params.stratified,
            "suite_path": params.suite_path,
            "methods": selected,
            "provider": params.provider,
            "model": params.model,
            "spacy_model": params.spacy_model,
        },
        "dataset": dataset_meta,
        "llm": {
            "provider": params.provider,
            "model": params.model,
            "error": llm_error,
        },
        "methods": method_results,
        "comparison": comparison,
        "paired_comparisons": paired,
        "evaluation": {
            "protocol_version": 3,
            "primary_metric": "strict_micro_f1",
            "secondary_metrics": [
                "partial_micro_f1",
                "macro_strict_f1",
                "bootstrap_strict_f1_ci",
                "paired_bootstrap_delta",
                "mcnemar_sentence_hits",
                "sentence_hit_rate",
                "error_taxonomy",
                "concept_recall",
            ],
            "partial_match_iou_threshold": 0.5,
            "split": params.split,
            "stratified_sampling": params.stratified,
        },
        "examples": examples,
    }


def _build_examples(
    sentences: list[Sentence],
    all_preds: dict[str, list[list[Span]]],
    max_examples: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for i in range(min(max_examples, len(sentences))):
        s = sentences[i]
        preds_by_method: dict[str, list[list[int]]] = {}
        for name, preds in all_preds.items():
            filtered = dedup_spans(numeric_filter(preds[i], s))
            preds_by_method[name] = [[p.start, p.end] for p in filtered]
        examples.append(
            {
                "index": s.index,
                "tokens": s.tokens,
                "text": s.text,
                "gold": [[g.start, g.end] for g in s.gold_spans],
                "predictions": preds_by_method,
            }
        )
    return examples
