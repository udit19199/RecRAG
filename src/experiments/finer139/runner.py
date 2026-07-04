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
from experiments.finer139.scoring import (
    metrics_to_dict,
    numeric_filter,
    score,
)
from experiments.finer139.types import Sentence, Span

logger = logging.getLogger(__name__)

ProgressCb = Callable[[str, int, int, str], None] | None
LLM_METHODS = {"llm", "hybrid", "dynamic"}
DEFAULT_SPACY_MODEL = "en_core_web_sm"


@dataclass
class RunParams:
    sample_size: int = 100
    seed: int = 42
    methods: list[str] = field(default_factory=lambda: list(ALL_METHODS))
    provider: str | None = None
    model: str | None = None
    max_examples: int = 15
    concurrency: int = 6
    spacy_model: str = DEFAULT_SPACY_MODEL


def _noop(stage: str, current: int, total: int, message: str) -> None:
    return None


def _build_llm(provider: str | None, model: str | None) -> Any:
    from adapters import create_llm_from_config
    from config import find_config_path, load_config

    config = load_config(find_config_path())
    return create_llm_from_config(config, provider=provider or None, model=model or None)


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
    report = progress_cb or _noop

    report("loading", 0, 0, "Loading FiNER-139 validation sample...")
    from experiments.finer139 import dataset as ds_mod
    from experiments.finer139.schema import build_gazetteer, build_gazetteer_regex

    sentences = ds_mod.load_sample(params.sample_size, params.seed)
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
            sc = score(sentences, preds)
            entry.update(
                {
                    "strict": metrics_to_dict(sc.strict),
                    "relaxed": metrics_to_dict(sc.relaxed),
                    "num_pred": sc.num_pred,
                    "num_gold": sc.num_gold,
                }
            )
        method_results.append(entry)

    report("scoring", total_methods, total_methods, "Building examples...")
    examples = _build_examples(sentences, all_preds, params.max_examples)

    total_gold = sum(len(s.gold_spans) for s in sentences)
    report("complete", total_methods, total_methods, "Done")

    return {
        "params": {
            "sample_size": params.sample_size,
            "seed": params.seed,
            "methods": selected,
            "provider": params.provider,
            "model": params.model,
            "spacy_model": params.spacy_model,
        },
        "dataset": {
            "id": ds_mod.DATASET_ID,
            "split": ds_mod.SPLIT,
            "num_sentences": len(sentences),
            "num_gold_entities": total_gold,
        },
        "llm": {
            "provider": params.provider,
            "model": params.model,
            "error": llm_error,
        },
        "methods": method_results,
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
