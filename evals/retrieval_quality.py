"""Score saved retrieval runs against gold supporting sentences by corpus offsets.

Pure local computation over ``retrieval.jsonl`` files: no model calls and no
network. Every gold sentence and every retrieved text is located in one
normalized copy of the dataset corpus, so methods with different chunking
(GraphRAG's own ~1000-char windows) are compared on the same character axis
instead of by string similarity thresholds.

Run with ``uv run python -m evals.retrieval_quality runs/<method>/<run> ...``.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import string
import unicodedata
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from benchmark import ROOT, Method, SavedRetrieval, split_pages
from dataset import SourcePage, load_hotpotqa_corpus, load_multihop_rag_record

METHOD_ORDER: list[Method] = ["rag", "hipporag", "raptor", "graphrag"]
RANKED_METHODS: set[Method] = {"rag", "hipporag", "raptor"}
BOOTSTRAP_METRICS = [
    "gold_recall",
    "full_span_recall",
    "all_gold_hit",
    "page_recall",
    "budget_recall",
]
DIFFERENCE_METRICS = ["gold_recall", "all_gold_hit", "budget_recall"]

# Curly quotes, primes, dash variants and odd spaces fold to ASCII; invisible
# characters are dropped. maketrans fails loudly if the two strings drift apart.
_FOLD = str.maketrans(
    "‘’‚‛′“”„‟″"
    "‐‑‒–—―−   ",
    "'''''\"\"\"\"\"-------   ",
    "​‌‍﻿­",
)
_WHITESPACE = re.compile(r"\s+")
# Agentic GraphRAG concatenated pages as "Page: <title>\n<body>" before re-chunking.
_PAGE_HEADER = re.compile(r"(?:^|\n\n)Page: ([^\n]*)\n")
# A window can end inside the next page header ("...\n\nPage: Some Ti").
_TRAILING_HEADER = re.compile(r"\n\n(?:P(?:a(?:g(?:e(?::(?: [^\n]*)?)?)?)?)?)?$")

ItemKind = Literal["chunk", "located_text", "summary", "window_segment", "unlocated"]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(text)).translate(_FOLD)
    return _WHITESPACE.sub(" ", text.lower()).strip()


def gold_key(sentence: str) -> str:
    return normalize(sentence).rstrip(string.punctuation + " ")


def _tokenizer():
    # tiktoken otherwise downloads cl100k_base; LlamaIndex ships a local copy.
    import llama_index.core
    import tiktoken

    os.environ.setdefault(
        "TIKTOKEN_CACHE_DIR",
        str(Path(llama_index.core.__file__).parent / "_static" / "tiktoken_cache"),
    )
    return tiktoken.get_encoding("cl100k_base")


@dataclass(slots=True, frozen=True)
class Span:
    start: int
    end: int

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(slots=True)
class Corpus:
    """Normalized page bodies joined by NUL so no match can cross pages."""

    dataset: str
    titles: list[str]
    text: str
    starts: list[int]
    ends: list[int]
    title_order: list[int]
    chunks: set[str]
    chunks_sha256: str

    @classmethod
    def build(cls, dataset: str, pages: Sequence[SourcePage]) -> Corpus:
        if any("\n" in page.title for page in pages):
            raise ValueError("Chunk titles are split on the first newline.")
        bodies = [normalize("\n\n".join(page.passages)) for page in pages]
        starts: list[int] = []
        ends: list[int] = []
        offset = 0
        for body in bodies:
            starts.append(offset)
            ends.append(offset + len(body))
            offset += len(body) + 1
        titles = [normalize(page.title) for page in pages]
        chunks = split_pages(pages)
        return cls(
            dataset=dataset,
            titles=titles,
            text="\x00".join(bodies),
            starts=starts,
            ends=ends,
            title_order=sorted(range(len(pages)), key=titles.__getitem__),
            chunks=set(chunks),
            chunks_sha256=sha256("\n\n".join(chunks).encode()).hexdigest(),
        )

    def page_of(self, offset: int) -> int:
        return bisect_right(self.starts, offset) - 1

    def page_text(self, page: int) -> str:
        return self.text[self.starts[page] : self.ends[page]]

    def pages_titled(self, title: str) -> list[int]:
        key = normalize(title)
        low = bisect_left(self.title_order, key, key=self.titles.__getitem__)
        high = bisect_right(self.title_order, key, key=self.titles.__getitem__)
        return self.title_order[low:high]

    def find_all(self, needle: str, pages: Sequence[int] | None = None) -> list[Span]:
        if not needle:
            return []
        ranges = (
            [Span(self.starts[page], self.ends[page]) for page in pages]
            if pages is not None
            else [Span(0, len(self.text))]
        )
        found: list[Span] = []
        for scope in ranges:
            position = self.text.find(needle, scope.start, scope.end)
            while position != -1:
                found.append(Span(position, position + len(needle)))
                position = self.text.find(needle, position + len(needle), scope.end)
        return found

    def locate(self, needle: str, title: str | None = None) -> list[Span]:
        if title is not None:
            spans = self.find_all(needle, self.pages_titled(title))
            if spans:
                return spans
        return self.find_all(needle)


def load_corpus(dataset: str) -> Corpus:
    if dataset == "hotpotqa":
        return Corpus.build(dataset, load_hotpotqa_corpus())
    if dataset == "multihop_rag":
        # Every MultiHop-RAG record shares the same full news corpus.
        return Corpus.build(dataset, load_multihop_rag_record(0).pages)
    raise ValueError(f"No retrieval-quality corpus adapter for {dataset!r}.")


@dataclass(slots=True)
class Item:
    rank: int
    kind: ItemKind
    text: str
    spans: list[Span]


class _GraphRAGEvidence(BaseModel):
    # entities and graph_facts are derived from the text; only text is scored.
    model_config = ConfigDict(extra="ignore")
    text: str


def chunk_item(rank: int, passage: str, corpus: Corpus) -> Item:
    newline = passage.find("\n")
    text = normalize(passage[newline + 1 :])
    return Item(rank, "chunk", text, corpus.locate(text, passage[:newline]))


def ranked_items(saved: SavedRetrieval, corpus: Corpus) -> list[Item]:
    items: list[Item] = []
    for rank, passage in enumerate(saved.retrieval.passages):
        if passage in corpus.chunks:
            items.append(chunk_item(rank, passage, corpus))
        elif saved.method == "raptor":
            # Generated summary: keeps its rank slot and token cost, earns no span.
            items.append(Item(rank, "summary", normalize(passage), []))
        else:
            text = normalize(passage[passage.find("\n") + 1 :])
            spans = corpus.find_all(text)
            items.append(
                Item(rank, "located_text" if spans else "unlocated", text, spans)
            )
    return items


def _suffix_spans(text: str, pages: Sequence[int], corpus: Corpus) -> list[Span]:
    return [
        Span(corpus.ends[page] - len(text), corpus.ends[page])
        for page in pages
        if text and corpus.page_text(page).endswith(text)
    ]


def _prefix_spans(text: str, pages: Sequence[int], corpus: Corpus) -> list[Span]:
    return [
        Span(corpus.starts[page], corpus.starts[page] + len(text))
        for page in pages
        if text and corpus.page_text(page).startswith(text)
    ]


def window_items(rank: int, window: str, corpus: Corpus) -> list[Item]:
    """Split one Agentic GraphRAG window at page headers and anchor each segment.

    The text before the first header is the tail of the page that precedes
    the next titled page; each headed body is the start of its titled page.
    Anchoring stops short tails like "d." from matching across the corpus.
    """
    parts = _PAGE_HEADER.split(window)
    titles = parts[1::2]
    bodies = [parts[0], *parts[2::2]]
    bodies[-1] = _TRAILING_HEADER.sub("", bodies[-1])
    items: list[Item] = []
    for index, raw in enumerate(bodies):
        text = normalize(raw)
        if not text:
            continue
        if index == 0:
            spans: list[Span] = []
            candidates = [raw]
            if "\n" in raw:
                # The window may open inside the previous page's title line.
                candidates.append(raw[raw.find("\n") + 1 :])
            for candidate in candidates:
                text = normalize(candidate)
                if titles:
                    before = [p - 1 for p in corpus.pages_titled(titles[0]) if p > 0]
                    spans = _suffix_spans(text, before, corpus)
                spans = spans or corpus.find_all(text)
                if spans:
                    break
            if not spans:
                text = normalize(raw)
        else:
            title = titles[index - 1]
            spans = _prefix_spans(text, corpus.pages_titled(title), corpus)
            spans = spans or corpus.locate(text, title)
        items.append(
            Item(rank, "window_segment" if spans else "unlocated", text, spans)
        )
    return items


def graphrag_items(saved: SavedRetrieval, corpus: Corpus) -> list[Item]:
    seen: set[str] = set()
    items: list[Item] = []
    for rank, passage in enumerate(saved.retrieval.passages):
        window = _GraphRAGEvidence.model_validate_json(passage).text
        if window in seen:
            continue
        seen.add(window)
        items.extend(window_items(rank, window, corpus))
    return items


@dataclass(slots=True)
class Gold:
    sentence: str
    key: str
    spans: list[Span]
    pages: frozenset[int]


def locate_gold(sentences: Sequence[str], corpus: Corpus) -> list[Gold]:
    golds: list[Gold] = []
    for sentence in sentences:
        key = gold_key(sentence)
        # Accept every page that contains the sentence, not just one.
        spans = corpus.find_all(key)
        golds.append(
            Gold(
                sentence,
                key,
                spans,
                frozenset(corpus.page_of(span.start) for span in spans),
            )
        )
    return golds


def merge(spans: Sequence[Span]) -> list[Span]:
    merged: list[Span] = []
    for span in sorted(spans, key=lambda span: span.start):
        if merged and span.start <= merged[-1].end:
            last = merged[-1]
            merged[-1] = Span(last.start, max(last.end, span.end))
        else:
            merged.append(span)
    return merged


def any_overlap(gold: Gold, spans: Sequence[Span]) -> bool:
    return any(g.overlaps(span) for g in gold.spans for span in spans)


def union_covers(gold: Gold, merged: Sequence[Span]) -> bool:
    """True if one gold occurrence lies inside the union of retrieved spans.

    Spans are merged first, so a sentence split across two touching or
    overlapping windows counts as fully covered.
    """
    return any(
        span.start <= g.start and g.end <= span.end for g in gold.spans for span in merged
    )


def budget_spans(items: Sequence[Item], budget: int, encoder) -> list[Span]:
    """Take ranked items until the token budget runs out, cutting the last."""
    taken: list[Span] = []
    used = 0
    for item in items:
        tokens = encoder.encode(item.text)
        if used + len(tokens) <= budget:
            taken.extend(item.spans)
            used += len(tokens)
            continue
        prefix = encoder.decode(tokens[: budget - used]) if budget > used else ""
        prefix_chars = len(prefix.rstrip("�"))
        taken.extend(
            Span(span.start, span.start + prefix_chars)
            for span in item.spans
            if prefix_chars
        )
        break
    return taken


def graphrag_tokens(items: Sequence[Item], corpus: Corpus, encoder) -> int:
    # Overlapping windows count once: tokens of the merged spans, plus any
    # segment that could not be located.
    located = sum(
        len(encoder.encode(corpus.text[span.start : span.end]))
        for span in merge([span for item in items for span in item.spans])
    )
    return located + sum(
        len(encoder.encode(item.text)) for item in items if item.kind == "unlocated"
    )


class GoldDetail(BaseModel):
    sentence: str
    occurrences: int
    pages: list[int]
    covered: bool
    fully_covered: bool
    budget_covered: bool
    hit_ranks: list[int]


class QuestionScore(BaseModel):
    dataset: str
    method: Method
    run: str
    record_id: str
    question_type: str | None
    question: str
    scored: bool
    gold_total: int
    gold_unlocated: int
    gold_covered: int
    gold_fully_covered: int
    gold_recall: float | None
    full_span_recall: float | None
    all_gold_hit: float | None
    page_groups: int
    page_hits: int
    page_recall: float | None
    budget_tokens: int
    budget_recall: float | None
    retrieved_tokens: int
    item_kinds: list[ItemKind]
    gold: list[GoldDetail]


def score_question(
    saved: SavedRetrieval,
    run: str,
    items: list[Item],
    golds: list[Gold],
    budget: int,
    corpus: Corpus,
    encoder,
) -> QuestionScore:
    spans = [span for item in items for span in item.spans]
    merged = merge(spans)
    ranked = saved.method in RANKED_METHODS
    cut = budget_spans(items, budget, encoder) if ranked else spans
    retrieved_pages = {corpus.page_of(span.start) for span in spans}
    groups = {gold.pages for gold in golds if gold.pages}
    page_hits = sum(1 for group in groups if group & retrieved_pages)
    details = [
        GoldDetail(
            sentence=gold.sentence,
            occurrences=len(gold.spans),
            pages=sorted(gold.pages),
            covered=any_overlap(gold, spans),
            fully_covered=union_covers(gold, merged),
            budget_covered=any_overlap(gold, cut),
            hit_ranks=sorted(
                {item.rank for item in items if any_overlap(gold, item.spans)}
            ),
        )
        for gold in golds
    ]
    total = len(golds)
    covered = sum(detail.covered for detail in details)

    def share(count: int) -> float | None:
        return count / total if total else None

    return QuestionScore(
        dataset=saved.dataset,
        method=saved.method,
        run=run,
        record_id=saved.record_id,
        question_type=saved.question_type,
        question=saved.question,
        scored=total > 0,
        gold_total=total,
        gold_unlocated=sum(1 for gold in golds if not gold.spans),
        gold_covered=covered,
        gold_fully_covered=sum(detail.fully_covered for detail in details),
        gold_recall=share(covered),
        full_span_recall=share(sum(detail.fully_covered for detail in details)),
        all_gold_hit=float(covered == total) if total else None,
        page_groups=len(groups),
        page_hits=page_hits,
        page_recall=page_hits / len(groups) if groups else None,
        budget_tokens=budget,
        budget_recall=share(sum(detail.budget_covered for detail in details)),
        retrieved_tokens=(
            sum(len(encoder.encode(item.text)) for item in items) if ranked else budget
        ),
        item_kinds=[item.kind for item in items],
        gold=details,
    )


class Estimate(BaseModel):
    mean: float
    low: float
    high: float


class KindCount(BaseModel):
    kind: ItemKind
    count: int


class MethodTokens(BaseModel):
    method: Method
    mean_retrieved_tokens: float


class MethodSummary(BaseModel):
    method: Method
    run: str
    bound: Literal["exact", "lower_bound"]
    note: str
    corpus_sha256_matches: bool
    scored_questions: int
    gold_sentences: int
    gold_unlocated: int
    gold_recall: Estimate
    full_span_recall: Estimate
    all_gold_hit: Estimate
    page_recall: Estimate
    budget_recall: Estimate
    mean_retrieved_tokens: float
    mean_budget_tokens: float
    item_kind_counts: list[KindCount]


class Difference(BaseModel):
    metric: str
    method: Method
    baseline: Method
    estimate: Estimate


class DatasetReport(BaseModel):
    dataset: str
    questions: int
    scored_questions: int
    excluded_record_ids: list[str]
    excluded_question_types: list[str]
    excluded_mean_retrieved_tokens: list[MethodTokens]
    methods: list[MethodSummary]
    differences: list[Difference]


class Report(BaseModel):
    created_at: str
    bootstrap_samples: int
    seed: int
    runs: list[str]
    definitions: list[str]
    datasets: list[DatasetReport]


DEFINITIONS = [
    "Text is normalized (HTML unescape, NFKC, quote/dash/space folding, "
    "lowercase, whitespace collapse); gold also drops trailing punctuation.",
    "Offsets live in one normalized corpus string; every occurrence of a gold "
    "sentence in any page is a valid target.",
    "covered: some retrieved span overlaps a gold occurrence by >= 1 character.",
    "fully covered: merged retrieved spans contain a whole gold occurrence, so "
    "a sentence split across two adjacent Agentic GraphRAG windows still counts.",
    "RAPTOR summaries keep rank and token cost but earn no span (lower bound).",
    "Agentic GraphRAG: only the JSON text field; identical windows deduped; each window "
    "split at 'Page:' headers into anchored per-page segments.",
    "budget_recall: ranked lists cut to Agentic GraphRAG's per-question token volume "
    "(title lines and page headers excluded, overlapping windows counted once); "
    "the crossing item is truncated.",
    "Intervals are paired percentile bootstraps over the same questions.",
]


def _array(scores: Sequence[QuestionScore], metric: str) -> np.ndarray:
    return np.array([getattr(score, metric) for score in scores], dtype=float)


def _estimate(values: np.ndarray, samples: np.ndarray) -> Estimate:
    boots = values[samples].mean(axis=1)
    return Estimate(
        mean=float(values.mean()),
        low=float(np.quantile(boots, 0.025)),
        high=float(np.quantile(boots, 0.975)),
    )


@dataclass(slots=True)
class MethodRun:
    path: Path
    run: str
    saved: list[SavedRetrieval]
    scores: list[QuestionScore]
    corpus_matches: bool


def score_dataset(
    dataset: str,
    runs: list[MethodRun],
    corpus: Corpus,
    encoder,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> DatasetReport:
    graphrag = next(run for run in runs if run.saved[0].method == "graphrag")
    for run in runs:
        if [s.record_id for s in run.saved] != [s.record_id for s in graphrag.saved]:
            raise ValueError(f"{run.run} does not share Agentic GraphRAG's question order.")
    golds = [locate_gold(saved.supporting_sentences, corpus) for saved in graphrag.saved]
    graph_items = [graphrag_items(saved, corpus) for saved in graphrag.saved]
    budgets = [graphrag_tokens(items, corpus, encoder) for items in graph_items]
    for run in runs:
        method = run.saved[0].method
        for index, saved in enumerate(run.saved):
            items = (
                graph_items[index]
                if method == "graphrag"
                else ranked_items(saved, corpus)
            )
            run.scores.append(
                score_question(
                    saved, run.run, items, golds[index], budgets[index],
                    corpus, encoder,
                )
            )
    scored = [index for index, gold in enumerate(golds) if gold]
    samples = rng.integers(0, len(scored), size=(bootstrap_samples, len(scored)))
    summaries: list[MethodSummary] = []
    arrays: list[list[np.ndarray]] = []
    for run in runs:
        method = run.saved[0].method
        kept = [run.scores[index] for index in scored]
        values = [_array(kept, metric) for metric in BOOTSTRAP_METRICS]
        arrays.append([_array(kept, metric) for metric in DIFFERENCE_METRICS])
        estimates = [_estimate(value, samples) for value in values]
        kinds = [kind for score in kept for kind in score.item_kinds]
        lower = method == "raptor"
        summaries.append(
            MethodSummary(
                method=method,
                run=run.run,
                bound="lower_bound" if lower else "exact",
                note=(
                    "Leaf chunks only; the runs' raptor_tree.json files are gone, "
                    "so summaries cannot be credited."
                    if lower
                    else ""
                ),
                corpus_sha256_matches=run.corpus_matches,
                scored_questions=len(kept),
                gold_sentences=sum(score.gold_total for score in kept),
                gold_unlocated=sum(score.gold_unlocated for score in kept),
                gold_recall=estimates[0],
                full_span_recall=estimates[1],
                all_gold_hit=estimates[2],
                page_recall=estimates[3],
                budget_recall=estimates[4],
                mean_retrieved_tokens=fmean(score.retrieved_tokens for score in kept),
                mean_budget_tokens=fmean(score.budget_tokens for score in kept),
                item_kind_counts=[
                    KindCount(kind=kind, count=kinds.count(kind))
                    for kind in sorted(set(kinds))
                ],
            )
        )
    differences: list[Difference] = []
    baseline = runs.index(graphrag)  # only graphrag-vs-other pairs are reported
    for left, run in enumerate(runs):
        if left == baseline:
            continue
        for position, metric in enumerate(DIFFERENCE_METRICS):
            differences.append(
                Difference(
                    metric=metric,
                    method=graphrag.saved[0].method,
                    baseline=run.saved[0].method,
                    estimate=_estimate(
                        arrays[baseline][position] - arrays[left][position], samples
                    ),
                )
            )
    excluded = [index for index, gold in enumerate(golds) if not gold]
    return DatasetReport(
        dataset=dataset,
        questions=len(golds),
        scored_questions=len(scored),
        excluded_record_ids=[graphrag.saved[index].record_id for index in excluded],
        excluded_question_types=sorted(
            {graphrag.saved[index].question_type or "unknown" for index in excluded}
        ),
        excluded_mean_retrieved_tokens=[
            MethodTokens(
                method=run.saved[0].method,
                mean_retrieved_tokens=fmean(
                    run.scores[index].retrieved_tokens for index in excluded
                ),
            )
            for run in runs
            if excluded
        ],
        methods=summaries,
        differences=differences,
    )


def load_run(run_dir: Path) -> list[SavedRetrieval]:
    lines = (run_dir / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()
    saved = [SavedRetrieval.model_validate_json(line) for line in lines if line]
    if not saved or len({s.method for s in saved} | {s.dataset for s in saved}) != 2:
        raise ValueError(f"{run_dir} must hold one method on one dataset.")
    return saved


def _relative(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(path)


def _cell(estimate: Estimate) -> str:
    return f"{estimate.mean:.3f} [{estimate.low:.3f},{estimate.high:.3f}]"


def print_table(report: Report) -> None:
    header = (
        f"{'method':<10} {'gold recall [95% CI]':<22} {'full-span':>9} "
        f"{'all-gold':>8} {'page':>6} {'budget recall [95% CI]':<24} "
        f"{'budget tok':>10} {'ret tok':>8}"
    )
    for dataset in report.datasets:
        print(
            f"\n{dataset.dataset}: {dataset.scored_questions}/{dataset.questions} "
            "questions scored"
        )
        if dataset.excluded_record_ids:
            print(
                f"  excluded {len(dataset.excluded_record_ids)} with no gold "
                f"({', '.join(dataset.excluded_question_types)})"
            )
        print(header)
        for summary in dataset.methods:
            name = summary.method + ("*" if summary.bound == "lower_bound" else "")
            print(
                f"{name:<10} {_cell(summary.gold_recall):<22} "
                f"{summary.full_span_recall.mean:>9.3f} {summary.all_gold_hit.mean:>8.3f} "
                f"{summary.page_recall.mean:>6.3f} "
                f"{_cell(summary.budget_recall):<24} "
                f"{summary.mean_budget_tokens:>10.0f} {summary.mean_retrieved_tokens:>8.0f}"
            )
        print("  paired differences vs graphrag (method - graphrag):")
        for difference in dataset.differences:
            if difference.metric != "all_gold_hit":
                print(
                    f"    graphrag - {difference.baseline:<8} {difference.metric:<14} "
                    f"{_cell(difference.estimate)}"
                )
    print("\n* lower bound: RAPTOR summaries counted as misses (tree not tied to run)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "results" / "retrieval_quality" / stamp
    encoder = _tokenizer()
    rng = np.random.default_rng(args.seed)
    loaded = [load_run(path) for path in args.runs]
    datasets = sorted({saved[0].dataset for saved in loaded})
    reports: list[DatasetReport] = []
    all_scores: list[QuestionScore] = []
    for dataset in datasets:
        corpus = load_corpus(dataset)
        runs = sorted(
            (
                MethodRun(
                    path=item_path,
                    run=_relative(item_path),
                    saved=saved,
                    scores=[],
                    corpus_matches={s.corpus_sha256 for s in saved}
                    == {corpus.chunks_sha256},
                )
                for item_path, saved in zip(args.runs, loaded)
                if saved[0].dataset == dataset
            ),
            key=lambda run: METHOD_ORDER.index(run.saved[0].method),
        )
        reports.append(
            score_dataset(dataset, runs, corpus, encoder, args.bootstrap, rng)
        )
        all_scores.extend(score for run in runs for score in run.scores)
    report = Report(
        created_at=stamp,
        bootstrap_samples=args.bootstrap,
        seed=args.seed,
        runs=[_relative(path) for path in args.runs],
        definitions=DEFINITIONS,
        datasets=reports,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "per_question.jsonl").write_text(
        "".join(score.model_dump_json() + "\n" for score in all_scores),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(report.model_dump_json(indent=2))
    print_table(report)
    print(f"\nWrote {_relative(output)}")


if __name__ == "__main__":
    main()
