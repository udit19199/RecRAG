"""Score saved retrieval runs with one LLM judge applied identically to every method.

RAPTOR summaries cannot be traced to source pages in the saved runs, so an
offset or ID metric cannot treat all four architectures the same way. The
judge reads content instead: each gold sentence gets an integer 0-3, a
question takes its weakest gold sentence, and the retrieval score is the
sum over questions divided by 3 x questions, as a percentage.

Two stages keep the scale identical across methods:
1. Each gold sentence is split into claims once per question and frozen, so
   every method is judged against the same claims.
2. Per method and question, the judge marks each claim stated, missing or
   contradicted and must quote the retrieved text. Quotes are checked in
   code and the integer score is derived from verified quotes only, so the
   judge never picks the number itself.

Verdicts are cached by a hash of prompt version, model and payload, so a
rerun resumes and re-scoring costs nothing.

Run with ``uv run python -m evals.retrieval_judge runs/<method>/<run> ...``.
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import tomllib
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict

from benchmark import ROOT, Method, SavedRetrieval, Settings
from cost import MeteredClient, Pricing, QuestionCost, measure
from evals.retrieval_quality import (
    METHOD_ORDER,
    RANKED_METHODS,
    Corpus,
    Estimate,
    Item,
    _estimate,
    _relative,
    _tokenizer,
    graphrag_items,
    graphrag_tokens,
    load_corpus,
    load_run,
    locate_gold,
    normalize,
    ranked_items,
    score_question,
)

# Bump when a prompt or the scoring rule changes; it is part of every cache key.
PROMPT_VERSION = "2026-10-30.2"
CACHE_PATH = ROOT / "runs" / "retrieval_judge" / "cache.jsonl"
MAX_SCORE = 3

CLAIMS_PROMPT = (
    "You split gold evidence sentences into the separate claims each one makes. "
    "Cover every claim in the sentence: do not drop names, numbers, dates, "
    "places, or qualifiers such as 'former', 'first', or 'only'. Each claim is "
    "one short self-contained statement in plain words, naming its subject "
    "instead of using pronouns. Mark exactly one claim per sentence as main: "
    "the central fact the sentence is about. Return one entry per sentence, "
    "using the sentence numbers given."
)
JUDGE_PROMPT = (
    "You check whether retrieved text states the claims of gold evidence "
    "sentences. Use only the retrieved items. Do not use outside knowledge and "
    "do not reason your way to a claim the text does not state; plain "
    "rewording, synonyms, abbreviations, and different formatting are fine.\n"
    "For every claim give one status:\n"
    "- stated: some item says it, with every specific matching (names, "
    "numbers, dates, places, who did what, qualifiers).\n"
    "- contradicted: some item gives a different value for the same specific, "
    "such as another year or another person.\n"
    "- missing: neither of the above.\n"
    "If an item states the claim and another item contradicts it, the status "
    "is contradicted.\n"
    "Evidence for stated or contradicted claims: copy one continuous passage "
    "from one item, character for character, with no ellipses or edits, and "
    "give that item's number. Never join separate words into a quote that "
    "does not appear as written; if the words are apart, quote the longer "
    "continuous passage that contains them (for 'in 1992 and 1993', quote "
    "that, not 'in 1993'). A claim spread across items may take several "
    "quotes. Missing claims take no evidence.\n"
    "For each sentence also list mentions: exact quotes where the main people, "
    "organizations, works, or things that sentence is about appear, even if "
    "no claim is stated. Use an empty list when none appear."
)

# Gold sentences mark one route to the answer, but the pooled corpus often
# states the same links on other pages, and fixed-size windows can cut a gold
# sentence mid-way. Gold-sentence recall therefore under-credits methods that
# reach the answer another way, so the primary score asks whether the
# retrieved text supports the gold answer by any explicitly stated route.
SUPPORT_PROMPT = (
    "You check whether retrieved text supports the gold answer to a question. "
    "You get the question, the gold answer, the gold evidence sentences (one "
    "reference route to the answer), and numbered retrieved items. Use only "
    "the retrieved items. Do not use outside knowledge and do not assume a "
    "link the text does not state; plain rewording is fine.\n"
    "chain: quotes from the items that, read together, lead from the question "
    "to the gold answer. Follow the reference route where the items state it. "
    "Any other route the items state explicitly is equally valid, such as a "
    "different page stating the same link. Each quote is one continuous "
    "passage copied character for character from one item, with no ellipses "
    "or edits, plus that item's number. Never join separate words into a quote "
    "that does not appear as written. Use an empty chain when nothing helps.\n"
    "complete: true only if the chain alone establishes every link needed to "
    "reach the gold answer; false if any link is missing or must be assumed.\n"
    "answer: stated when an item states the gold answer as the answer to what "
    "the question asks; mentioned when the gold answer appears only in passing, "
    "such as in a list, an entity name, or unrelated context; missing "
    "otherwise. For a yes or no gold answer always use missing, because those "
    "are judged by the chain alone. Give quotes for stated or mentioned."
)
# No rule ties links to a named publication: GraphRAG's character windows
# mostly lack the "Page: title (source, date)" header (219 of 898 on
# MultiHop-RAG) while chunks repeat it, so such a rule scored a missing
# header rather than missing evidence.
YES_NO = {"yes", "no"}

ClaimStatus = Literal["stated", "missing", "contradicted"]
AnswerStatus = Literal["stated", "mentioned", "missing"]
View = Literal["full", "budget"]


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    main: bool


class SentenceClaims(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentence: int
    claims: list[Claim]


class ClaimSplit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentences: list[SentenceClaims]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item: int
    quote: str


class ClaimJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: int
    status: ClaimStatus
    evidence: list[Evidence]


class SentenceJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentence: int
    claims: list[ClaimJudgement]
    mentions: list[Evidence]


class Judgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentences: list[SentenceJudgement]


class AnswerJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: AnswerStatus
    evidence: list[Evidence]


class SupportJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chain: list[Evidence]
    complete: bool
    answer: AnswerJudgement


CallKind = Literal["claims", "judge", "support"]


class CacheEntry(BaseModel):
    key: str
    kind: CallKind
    prompt_version: str
    model: str
    dataset: str
    record_id: str
    method: Method | None
    response: str
    cost: QuestionCost


class Cache:
    """Append-only JSONL; a rerun skips every call whose key is already stored."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        self.entries = [CacheEntry.model_validate_json(line) for line in lines if line]
        self.keys = {entry.key for entry in self.entries}

    def get(self, key: str) -> CacheEntry | None:
        if key not in self.keys:
            return None
        # The newest entry wins, so a re-asked call replaces a rejected reply.
        return next(entry for entry in reversed(self.entries) if entry.key == key)

    def put(self, entry: CacheEntry) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(entry.model_dump_json() + "\n")
            self.entries.append(entry)
            self.keys.add(entry.key)


@dataclass(slots=True)
class Judge:
    model: ChatOpenAI
    model_name: str
    pricing: Pricing
    cache: Cache

    @classmethod
    def from_config(cls, cache: Cache) -> Judge:
        settings = Settings.model_validate(
            tomllib.loads((ROOT / "config.toml").read_text())
        )
        model = ChatOpenAI(
            model=settings.llm.model,
            timeout=settings.llm.timeout,
            use_responses_api=True,
            streaming=False,
            disable_streaming=True,
            # POST is not idempotent; a failed call is re-run from the cache.
            max_retries=0,
            http_client=MeteredClient(timeout=settings.llm.timeout),
            reasoning={"effort": settings.llm.reasoning_effort},
        )
        return cls(model, settings.llm.model, settings.pricing, cache)

    def call[T: BaseModel](
        self,
        kind: CallKind,
        schema: type[T],
        system: str,
        user: str,
        dataset: str,
        record_id: str,
        method: Method | None,
        check: Callable[[T], None],
    ) -> T:
        key = sha256(
            "\x00".join([PROMPT_VERSION, self.model_name, kind, system, user]).encode()
        ).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            response = schema.model_validate_json(cached.response)
            try:
                check(response)
                return response
            except ValueError:
                pass  # Stored before replies were checked; ask again.
        structured = self.model.with_structured_output(
            schema, method="json_schema", strict=True
        )
        with measure(self.pricing) as cost:
            response = schema.model_validate(
                structured.invoke([SystemMessage(system), HumanMessage(user)])
            )
        # Check before caching: a reply that skips sentences or claims must be
        # asked again on rerun, not replayed from the cache.
        check(response)
        self.cache.put(
            CacheEntry(
                key=key,
                kind=kind,
                prompt_version=PROMPT_VERSION,
                model=self.model_name,
                dataset=dataset,
                record_id=record_id,
                method=method,
                response=response.model_dump_json(),
                cost=cost,
            )
        )
        return response


def claims_payload(sentences: Sequence[str]) -> str:
    return "\n".join(f"[{index}] {sentence}" for index, sentence in enumerate(sentences))


def split_claims(judge: Judge, saved: SavedRetrieval) -> ClaimSplit:
    def check(split: ClaimSplit) -> None:
        numbers = sorted(entry.sentence for entry in split.sentences)
        if numbers != list(range(len(saved.supporting_sentences))):
            raise ValueError(f"{saved.record_id}: claim split skipped sentences.")
        for entry in split.sentences:
            if sum(claim.main for claim in entry.claims) != 1:
                raise ValueError(f"{saved.record_id}: sentence {entry.sentence} main claim.")

    return judge.call(
        "claims",
        ClaimSplit,
        CLAIMS_PROMPT,
        claims_payload(saved.supporting_sentences),
        saved.dataset,
        saved.record_id,
        None,
        check,
    )


def judgement_check(saved: SavedRetrieval, split: ClaimSplit) -> Callable[[Judgement], None]:
    def check(judgement: Judgement) -> None:
        numbers = sorted(entry.sentence for entry in judgement.sentences)
        if numbers != list(range(len(saved.supporting_sentences))):
            raise ValueError(f"{saved.record_id}: judge skipped sentences.")
        for entry in judgement.sentences:
            claims = next(s for s in split.sentences if s.sentence == entry.sentence)
            if sorted(c.claim for c in entry.claims) != list(range(len(claims.claims))):
                raise ValueError(f"{saved.record_id}: judge skipped claims.")

    return check


class _GraphRelationship(BaseModel):
    # Merged nodes keep every name they were extracted under, so an end can be a list.
    model_config = ConfigDict(extra="ignore")
    source: str | list[str]
    target: str | list[str]
    type: str

    def render(self) -> str:
        ends = [" / ".join([end] if isinstance(end, str) else end) for end in [self.source, self.target]]
        return f"{ends[0]} -[{self.type}]-> {ends[1]}"


class _GraphFact(BaseModel):
    model_config = ConfigDict(extra="ignore")
    relationships: list[_GraphRelationship]


class _GraphEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str
    entities: list[str | list[str]]
    graph_facts: list[_GraphFact]


def unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    return [value for value in values if not (value in seen or seen.add(value))]


@dataclass(slots=True)
class JudgeItem:
    """One retrieved item as the judge sees it, plus its in-budget text."""

    rank: int
    kind: str
    text: str
    full_key: str
    budget_key: str


def render_graphrag(passage: str) -> str:
    evidence = _GraphEvidence.model_validate_json(passage)
    lines = [evidence.text]
    facts = [
        relation.render()
        for fact in evidence.graph_facts
        for relation in fact.relationships
    ]
    if facts:
        lines.append("Graph facts:\n" + "\n".join(f"- {fact}" for fact in unique(facts)))
    if evidence.entities:
        lines.append(
            "Entities: "
            + ", ".join(
                entity if isinstance(entity, str) else " / ".join(entity)
                for entity in evidence.entities
            )
        )
    return "\n\n".join(lines)


def budget_prefixes(items: Sequence[Item], budget: int, encoder) -> list[str]:
    """Normalized text of each ranked item kept under the token budget.

    Mirrors ``retrieval_quality.budget_spans``: whole items until the budget
    runs out, the crossing item truncated, the rest dropped.
    """
    kept: list[str] = []
    used = 0
    for item in items:
        tokens = encoder.encode(item.text)
        if used + len(tokens) <= budget:
            kept.append(item.text)
            used += len(tokens)
        elif budget > used:
            kept.append(encoder.decode(tokens[: budget - used]).rstrip("�"))
            used = budget
        else:
            kept.append("")
    return kept


def judge_items(
    saved: SavedRetrieval, corpus: Corpus, budget: int, encoder
) -> list[JudgeItem]:
    if saved.method == "graphrag":
        # GraphRAG's own volume defines the budget, so both views match.
        rendered = unique([render_graphrag(passage) for passage in saved.retrieval.passages])
        return [
            JudgeItem(rank, "graphrag", text, normalize(text), normalize(text))
            for rank, text in enumerate(rendered)
        ]
    scored = ranked_items(saved, corpus)
    prefixes = budget_prefixes(scored, budget, encoder)
    items: list[JudgeItem] = []
    for item, passage, prefix in zip(scored, saved.retrieval.passages, prefixes):
        if item.kind == "summary":
            budget_key = prefix
        else:
            # Title lines are outside the token budget, as in retrieval_quality.
            title = normalize(passage[: passage.find("\n")])
            budget_key = f"{title} {prefix}" if prefix else ""
        items.append(
            JudgeItem(item.rank, item.kind, passage, normalize(passage), budget_key)
        )
    return items


def judge_payload(
    saved: SavedRetrieval, split: ClaimSplit, order: Sequence[JudgeItem]
) -> str:
    sentences = sorted(split.sentences, key=lambda entry: entry.sentence)
    gold = "\n\n".join(
        f"Sentence {entry.sentence}: {saved.supporting_sentences[entry.sentence]}\n"
        + "\n".join(
            f"  Claim {index}{' (main)' if claim.main else ''}: {claim.text}"
            for index, claim in enumerate(entry.claims)
        )
        for entry in sentences
    )
    retrieved = "\n\n".join(
        f"<item {number}>\n{item.text}\n</item {number}>"
        for number, item in enumerate(order, start=1)
    )
    return f"Gold sentences and claims:\n{gold}\n\nRetrieved items:\n{retrieved}"


def gold_answers(saved: SavedRetrieval) -> list[str]:
    return saved.expected_answers or (
        [saved.canonical_answer] if saved.canonical_answer else []
    )


def support_payload(saved: SavedRetrieval, order: Sequence[JudgeItem]) -> str:
    route = "\n".join(
        f"[{index}] {sentence}" for index, sentence in enumerate(saved.supporting_sentences)
    )
    retrieved = "\n\n".join(
        f"<item {number}>\n{item.text}\n</item {number}>"
        for number, item in enumerate(order, start=1)
    )
    return (
        f"Question: {saved.question}\n"
        f"Gold answer: {' | '.join(gold_answers(saved))}\n\n"
        f"Gold evidence sentences (reference route):\n{route}\n\n"
        f"Retrieved items:\n{retrieved}"
    )


def shuffled(items: Sequence[JudgeItem], saved: SavedRetrieval) -> list[JudgeItem]:
    # Seeded per question and method so the order is stable across reruns.
    order = list(items)
    random.Random(f"{saved.dataset}:{saved.record_id}:{saved.method}").shuffle(order)
    return order


class ClaimDetail(BaseModel):
    text: str
    main: bool
    judged_status: ClaimStatus
    full_status: ClaimStatus
    budget_status: ClaimStatus
    quotes: list[str]
    unverified_quotes: int


class GoldScore(BaseModel):
    sentence: str
    full_score: int
    budget_score: int
    offset_fully_covered: bool
    offset_covered: bool
    claims: list[ClaimDetail]
    mention_quotes: list[str]
    credited_kinds: list[str]


class SupportDetail(BaseModel):
    full_score: int
    budget_score: int
    complete: bool
    judged_answer: AnswerStatus
    chain_quotes: list[str]
    answer_quotes: list[str]
    unverified_quotes: int
    credited_kinds: list[str]


class QuestionScore(BaseModel):
    dataset: str
    method: Method
    run: str
    record_id: str
    question_type: str | None
    question: str
    support_score: int
    budget_support_score: int
    support: SupportDetail
    full_score: int
    budget_score: int
    budget_tokens: int
    has_untraceable_items: bool
    gold: list[GoldScore]


def verified(evidence: Evidence, order: Sequence[JudgeItem], view: View) -> bool:
    if not 1 <= evidence.item <= len(order):
        return False
    item = order[evidence.item - 1]
    text = item.full_key if view == "full" else item.budget_key
    quote = normalize(evidence.quote)
    return bool(quote) and quote in text


def claim_status(judged: ClaimJudgement, order: Sequence[JudgeItem], view: View) -> ClaimStatus:
    if judged.status == "missing":
        return "missing"
    if any(verified(evidence, order, view) for evidence in judged.evidence):
        return judged.status
    return "missing"


def sentence_score(
    statuses: Sequence[ClaimStatus], mains: Sequence[bool], mentioned: bool
) -> int:
    """3 all stated, 2 main stated, 1 mentioned or a minor claim, 0 otherwise.

    A verified contradiction anywhere scores 0: the answer model would see
    both versions and could pick the wrong one.
    """
    if "contradicted" in statuses:
        return 0
    if all(status == "stated" for status in statuses):
        return 3
    if any(status == "stated" for status, main in zip(statuses, mains) if main):
        return 2
    if mentioned or "stated" in statuses:
        return 1
    return 0


def support_value(
    judged: SupportJudgement, order: Sequence[JudgeItem], view: View, yes_no: bool
) -> int:
    """3 full chain and answer stated; 2 answer stated; 1 anything; 0 nothing.

    A yes or no answer is never written as such, so a complete chain alone
    earns 3 and score 2 cannot occur for it.
    """
    chain = [verified(evidence, order, view) for evidence in judged.chain]
    complete = judged.complete and bool(chain) and all(chain)
    answer = (
        judged.answer.status
        if any(verified(evidence, order, view) for evidence in judged.answer.evidence)
        else "missing"
    )
    if complete and (yes_no or answer == "stated"):
        return 3
    if answer == "stated":
        return 2
    if answer == "mentioned" or any(chain):
        return 1
    return 0


def support_detail(
    judged: SupportJudgement, order: Sequence[JudgeItem], yes_no: bool
) -> SupportDetail:
    quotes = [*judged.chain, *judged.answer.evidence]
    return SupportDetail(
        full_score=support_value(judged, order, "full", yes_no),
        budget_score=support_value(judged, order, "budget", yes_no),
        complete=judged.complete,
        judged_answer=judged.answer.status,
        chain_quotes=[evidence.quote for evidence in judged.chain],
        answer_quotes=[evidence.quote for evidence in judged.answer.evidence],
        unverified_quotes=sum(not verified(e, order, "full") for e in quotes),
        credited_kinds=sorted(
            {order[e.item - 1].kind for e in quotes if verified(e, order, "full")}
        ),
    )


def score_gold(
    sentence: str,
    claims: SentenceClaims,
    judged: SentenceJudgement,
    order: Sequence[JudgeItem],
    offset_full: bool,
    offset_any: bool,
) -> GoldScore:
    if sorted(entry.claim for entry in judged.claims) != list(range(len(claims.claims))):
        raise ValueError(f"Judge skipped claims of: {sentence}")
    by_claim = sorted(judged.claims, key=lambda entry: entry.claim)
    mains = [claim.main for claim in claims.claims]
    scores: list[int] = []
    for view in ["full", "budget"]:
        statuses = [claim_status(entry, order, view) for entry in by_claim]
        mentioned = any(verified(evidence, order, view) for evidence in judged.mentions)
        scores.append(sentence_score(statuses, mains, mentioned))
    credited = sorted(
        {
            order[evidence.item - 1].kind
            for entry in by_claim
            if entry.status == "stated"
            for evidence in entry.evidence
            if verified(evidence, order, "full")
        }
    )
    return GoldScore(
        sentence=sentence,
        full_score=scores[0],
        budget_score=scores[1],
        offset_fully_covered=offset_full,
        offset_covered=offset_any,
        claims=[
            ClaimDetail(
                text=claim.text,
                main=claim.main,
                judged_status=entry.status,
                full_status=claim_status(entry, order, "full"),
                budget_status=claim_status(entry, order, "budget"),
                quotes=[evidence.quote for evidence in entry.evidence],
                unverified_quotes=sum(
                    not verified(evidence, order, "full") for evidence in entry.evidence
                ),
            )
            for claim, entry in zip(claims.claims, by_claim)
        ],
        mention_quotes=[
            evidence.quote
            for evidence in judged.mentions
            if verified(evidence, order, "full")
        ],
        credited_kinds=credited,
    )


@dataclass(slots=True)
class MethodRun:
    run: str
    saved: list[SavedRetrieval]
    scores: list[QuestionScore | None]


@dataclass(slots=True)
class Task:
    label: str
    future: Future


def drain(tasks: Sequence[Task], every: int) -> None:
    for done, _ in enumerate(as_completed([task.future for task in tasks]), start=1):
        if done % every == 0:
            print(f"  {done}/{len(tasks)}", flush=True)


def outcome(task: Task):
    """The task's result, or None after reporting the error; a rerun resumes."""
    try:
        return task.future.result()
    except Exception as error:  # noqa: BLE001
        print(f"  failed {task.label}: {error}", file=sys.stderr)
        return None


def score_one(
    judge: Judge,
    saved: SavedRetrieval,
    run: str,
    split: ClaimSplit,
    corpus: Corpus,
    budget: int,
    graph_items: list[Item],
    encoder,
) -> QuestionScore:
    order = shuffled(judge_items(saved, corpus, budget, encoder), saved)
    judgement = judge.call(
        "judge",
        Judgement,
        JUDGE_PROMPT,
        judge_payload(saved, split, order),
        saved.dataset,
        saved.record_id,
        saved.method,
        judgement_check(saved, split),
    )
    answers = gold_answers(saved)
    if not answers:
        raise ValueError(f"{saved.record_id}: no gold answer.")
    yes_no = all(answer.strip().lower() in YES_NO for answer in answers)
    support = support_detail(
        judge.call(
            "support",
            SupportJudgement,
            SUPPORT_PROMPT,
            support_payload(saved, order),
            saved.dataset,
            saved.record_id,
            saved.method,
            lambda judged: None,
        ),
        order,
        yes_no,
    )
    # ID-style offset coverage is kept only as a check on the judge.
    offset_items = graph_items if saved.method == "graphrag" else ranked_items(saved, corpus)
    golds = locate_gold(saved.supporting_sentences, corpus)
    offset = score_question(saved, run, offset_items, golds, budget, corpus, encoder)
    claims = sorted(split.sentences, key=lambda entry: entry.sentence)
    judged = sorted(judgement.sentences, key=lambda entry: entry.sentence)
    gold = [
        score_gold(sentence, claim, entry, order, detail.fully_covered, detail.covered)
        for sentence, claim, entry, detail in zip(
            saved.supporting_sentences, claims, judged, offset.gold
        )
    ]
    return QuestionScore(
        dataset=saved.dataset,
        method=saved.method,
        run=run,
        record_id=saved.record_id,
        question_type=saved.question_type,
        question=saved.question,
        support_score=support.full_score,
        budget_support_score=support.budget_score,
        support=support,
        # A multi-hop question is only as answerable as its weakest gold sentence.
        full_score=min(score.full_score for score in gold),
        budget_score=min(score.budget_score for score in gold),
        budget_tokens=budget,
        has_untraceable_items=any(
            kind in {"summary", "unlocated"} for kind in offset.item_kinds
        ),
        gold=gold,
    )


class MethodSummary(BaseModel):
    method: Method
    run: str
    questions: int
    support_score: Estimate
    full_support_rate: Estimate
    budget_support_score: Estimate
    budget_full_support_rate: Estimate
    support_counts: list[int]
    support_unverified_quotes: int
    # Secondary: recall of the annotated gold sentences themselves.
    retrieval_score: Estimate
    full_retrieval_rate: Estimate
    budget_retrieval_score: Estimate
    budget_full_retrieval_rate: Estimate
    score_counts: list[int]
    unverified_quotes: int
    total_quotes: int


class Difference(BaseModel):
    metric: str
    method: Method
    baseline: Method
    estimate: Estimate


class Agreement(BaseModel):
    """Judge vs offset coverage on gold sentences with only traceable items."""

    method: Method
    gold_sentences: int
    offset_full_judge_3: int
    offset_full_judge_below_3: int
    offset_none_judge_0_or_1: int
    offset_none_judge_2_or_3: int
    offset_partial: int
    # Support vs gold-sentence recall over all scored questions.
    gold_complete_questions: int
    gold_complete_support_3: int
    gold_incomplete_questions: int
    gold_incomplete_support_3: int


class DatasetReport(BaseModel):
    dataset: str
    questions: int
    scored_questions: int
    failed_record_ids: list[str]
    methods: list[MethodSummary]
    differences: list[Difference]
    agreement: list[Agreement]


class Report(BaseModel):
    created_at: str
    prompt_version: str
    model: str
    bootstrap_samples: int
    seed: int
    runs: list[str]
    definitions: list[str]
    judge_cost_usd: float | None
    datasets: list[DatasetReport]


DEFINITIONS = [
    "Primary, answer support: the judge sees the question, gold answer, gold "
    "sentences as one reference route, and the items. It quotes a chain to the "
    "gold answer by the reference route or any route the items state, says "
    "whether the chain is complete, and whether the answer is stated, only "
    "mentioned, or missing.",
    "Support score: 3 complete verified chain and answer stated (yes/no "
    "answers: complete chain); 2 answer stated; 1 answer mentioned or some "
    "verified chain quote; 0 otherwise. Support score % = sum / (3 x "
    "questions) x 100; full support rate = share scoring 3.",
    "Secondary, gold-sentence recall, as follows.",
    "Gold sentences are split into claims once per question; every method is "
    "judged against the same frozen claims.",
    "Each claim is stated, missing, or contradicted; stated and contradicted "
    "need a quote that appears verbatim (after normalization) in the cited "
    "item, otherwise the claim counts as missing.",
    "Gold sentence score: 3 all claims stated; 2 main claim stated; 1 people "
    "or things mentioned or a minor claim stated; 0 otherwise. Any verified "
    "contradiction scores 0.",
    "Question score: the lowest gold sentence score.",
    "Retrieval score: sum of question scores / (3 x questions) x 100.",
    "Full retrieval rate: share of questions scoring 3, x 100.",
    "Budget view: ranked methods cut to GraphRAG's per-question token volume "
    "(as in retrieval_quality); quotes outside the cut do not count.",
    "Judge sees no method name, no question, and items in a seeded shuffle.",
    "Intervals are paired percentile bootstraps over the same questions.",
]


def _percent(values: Sequence[float], samples: np.ndarray) -> Estimate:
    estimate = _estimate(np.array(values, dtype=float), samples)
    return Estimate(mean=estimate.mean * 100, low=estimate.low * 100, high=estimate.high * 100)


def _metrics(scores: Sequence[QuestionScore]) -> list[list[float]]:
    return [
        [score.support_score / MAX_SCORE for score in scores],
        [float(score.support_score == MAX_SCORE) for score in scores],
        [score.budget_support_score / MAX_SCORE for score in scores],
        [float(score.budget_support_score == MAX_SCORE) for score in scores],
        [score.full_score / MAX_SCORE for score in scores],
        [float(score.full_score == MAX_SCORE) for score in scores],
        [score.budget_score / MAX_SCORE for score in scores],
        [float(score.budget_score == MAX_SCORE) for score in scores],
    ]


METRICS = [
    "support_score",
    "full_support_rate",
    "budget_support_score",
    "budget_full_support_rate",
    "retrieval_score",
    "full_retrieval_rate",
    "budget_retrieval_score",
    "budget_full_retrieval_rate",
]


def agreement(method: Method, scores: Sequence[QuestionScore]) -> Agreement:
    gold = [g for score in scores if not score.has_untraceable_items for g in score.gold]
    full = [g for g in gold if g.offset_fully_covered]
    none = [g for g in gold if not g.offset_covered]
    return Agreement(
        method=method,
        gold_sentences=len(gold),
        offset_full_judge_3=sum(g.full_score == 3 for g in full),
        offset_full_judge_below_3=sum(g.full_score < 3 for g in full),
        offset_none_judge_0_or_1=sum(g.full_score <= 1 for g in none),
        offset_none_judge_2_or_3=sum(g.full_score >= 2 for g in none),
        offset_partial=len(gold) - len(full) - len(none),
        gold_complete_questions=sum(s.full_score == 3 for s in scores),
        gold_complete_support_3=sum(
            s.full_score == 3 and s.support_score == 3 for s in scores
        ),
        gold_incomplete_questions=sum(s.full_score < 3 for s in scores),
        gold_incomplete_support_3=sum(
            s.full_score < 3 and s.support_score == 3 for s in scores
        ),
    )


def summarize(
    dataset: str, runs: list[MethodRun], bootstrap: int, rng: np.random.Generator
) -> DatasetReport:
    # Only questions every method scored, so comparisons stay paired.
    total = len(runs[0].saved)
    kept = [
        index
        for index in range(total)
        if runs[0].saved[index].supporting_sentences
        and all(run.scores[index] is not None for run in runs)
    ]
    failed = [
        runs[0].saved[index].record_id
        for index in range(total)
        if runs[0].saved[index].supporting_sentences
        and any(run.scores[index] is None for run in runs)
    ]
    samples = rng.integers(0, len(kept), size=(bootstrap, len(kept)))
    summaries: list[MethodSummary] = []
    arrays: list[list[list[float]]] = []
    for run in runs:
        scores = [run.scores[index] for index in kept]
        values = _metrics(scores)
        arrays.append(values)
        claims = [c for s in scores for g in s.gold for c in g.claims]
        summaries.append(
            MethodSummary(
                method=run.saved[0].method,
                run=run.run,
                questions=len(scores),
                support_score=_percent(values[0], samples),
                full_support_rate=_percent(values[1], samples),
                budget_support_score=_percent(values[2], samples),
                budget_full_support_rate=_percent(values[3], samples),
                support_counts=[
                    sum(score.support_score == value for score in scores)
                    for value in range(MAX_SCORE + 1)
                ],
                support_unverified_quotes=sum(
                    score.support.unverified_quotes for score in scores
                ),
                retrieval_score=_percent(values[4], samples),
                full_retrieval_rate=_percent(values[5], samples),
                budget_retrieval_score=_percent(values[6], samples),
                budget_full_retrieval_rate=_percent(values[7], samples),
                score_counts=[
                    sum(score.full_score == value for score in scores)
                    for value in range(MAX_SCORE + 1)
                ],
                unverified_quotes=sum(c.unverified_quotes for c in claims),
                total_quotes=sum(len(c.quotes) for c in claims),
            )
        )
    graph = next(i for i, run in enumerate(runs) if run.saved[0].method == "graphrag")
    differences = [
        Difference(
            metric=metric,
            method=runs[index].saved[0].method,
            baseline="graphrag",
            estimate=_percent(
                list(np.array(arrays[index][position]) - np.array(arrays[graph][position])),
                samples,
            ),
        )
        for index in range(len(runs))
        if index != graph
        for position, metric in enumerate(METRICS)
    ]
    return DatasetReport(
        dataset=dataset,
        questions=total,
        scored_questions=len(kept),
        failed_record_ids=failed,
        methods=summaries,
        differences=differences,
        agreement=[
            agreement(run.saved[0].method, [run.scores[index] for index in kept])
            for run in runs
        ],
    )


def _cell(estimate: Estimate) -> str:
    return f"{estimate.mean:5.1f} [{estimate.low:5.1f},{estimate.high:5.1f}]"


def print_table(report: Report) -> None:
    for dataset in report.datasets:
        print(
            f"\n{dataset.dataset}: {dataset.scored_questions}/{dataset.questions} "
            "questions scored"
            + (f", {len(dataset.failed_record_ids)} failed" if dataset.failed_record_ids else "")
        )
        print(
            f"{'method':<10} {'support score % [95% CI]':<27} "
            f"{'full support % [95% CI]':<27} {'budget score %':>14} "
            f"{'budget full %':>13}  scores 0/1/2/3"
        )
        for summary in dataset.methods:
            print(
                f"{summary.method:<10} {_cell(summary.support_score):<27} "
                f"{_cell(summary.full_support_rate):<27} "
                f"{summary.budget_support_score.mean:>14.1f} "
                f"{summary.budget_full_support_rate.mean:>13.1f}  "
                + "/".join(str(count) for count in summary.support_counts)
            )
        print("  secondary, gold-sentence recall:")
        for summary in dataset.methods:
            print(
                f"{summary.method:<10} {_cell(summary.retrieval_score):<27} "
                f"{_cell(summary.full_retrieval_rate):<27} "
                f"{summary.budget_retrieval_score.mean:>14.1f} "
                f"{summary.budget_full_retrieval_rate.mean:>13.1f}  "
                + "/".join(str(count) for count in summary.score_counts)
            )
        print("  paired differences (method - graphrag), percentage points:")
        for difference in dataset.differences:
            if difference.metric in {"support_score", "full_support_rate", "retrieval_score"}:
                print(
                    f"    {difference.method:<8} {difference.metric:<20} "
                    f"{_cell(difference.estimate)}"
                )
        print("  judge vs offset coverage (questions with only traceable items):")
        for row in dataset.agreement:
            print(
                f"    {row.method:<8} offset full: judge 3 {row.offset_full_judge_3}, "
                f"<3 {row.offset_full_judge_below_3} | offset none: judge 0-1 "
                f"{row.offset_none_judge_0_or_1}, 2-3 {row.offset_none_judge_2_or_3} "
                f"| partial {row.offset_partial}"
            )
        print("  support vs gold-sentence recall (questions):")
        for row in dataset.agreement:
            print(
                f"    {row.method:<8} all gold found: support 3 in "
                f"{row.gold_complete_support_3}/{row.gold_complete_questions} | "
                f"gold missing: support 3 in "
                f"{row.gold_incomplete_support_3}/{row.gold_incomplete_questions}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, help="Score only the first N questions.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "results" / "retrieval_judge" / stamp
    cache = Cache(CACHE_PATH)
    judge = Judge.from_config(cache)
    encoder = _tokenizer()
    rng = np.random.default_rng(args.seed)
    loaded = [load_run(path) for path in args.runs]
    reports: list[DatasetReport] = []
    all_scores: list[QuestionScore] = []
    failures = 0
    for dataset in sorted({saved[0].dataset for saved in loaded}):
        corpus = load_corpus(dataset)
        runs = sorted(
            (
                MethodRun(_relative(path), saved[: args.limit], [])
                for path, saved in zip(args.runs, loaded)
                if saved[0].dataset == dataset
            ),
            key=lambda run: METHOD_ORDER.index(run.saved[0].method),
        )
        methods = [run.saved[0].method for run in runs]
        if sorted(methods) != sorted(set(methods)) or "graphrag" not in methods:
            raise ValueError(f"{dataset}: give one run per method, including graphrag.")
        graph = next(run for run in runs if run.saved[0].method == "graphrag")
        for run in runs:
            if [s.record_id for s in run.saved] != [s.record_id for s in graph.saved]:
                raise ValueError(f"{run.run} does not share GraphRAG's question order.")
            run.scores = [None] * len(run.saved)
        graph_items = [graphrag_items(saved, corpus) for saved in graph.saved]
        budgets = [graphrag_tokens(items, corpus, encoder) for items in graph_items]
        gold_indexes = [i for i, saved in enumerate(graph.saved) if saved.supporting_sentences]
        print(f"{dataset}: splitting claims for {len(gold_indexes)} questions", flush=True)
        splits: list[ClaimSplit | None] = [None] * len(graph.saved)
        with ThreadPoolExecutor(args.workers) as pool:
            tasks = [
                Task(
                    f"claims {graph.saved[index].record_id}",
                    pool.submit(split_claims, judge, graph.saved[index]),
                )
                for index in gold_indexes
            ]
            drain(tasks, 25)
        for index, task in zip(gold_indexes, tasks):
            splits[index] = outcome(task)
            failures += splits[index] is None
        ready = [index for index in gold_indexes if splits[index] is not None]
        print(f"{dataset}: judging {len(ready) * len(runs)} method-question pairs", flush=True)
        for run in runs:
            with ThreadPoolExecutor(args.workers) as pool:
                tasks = [
                    Task(
                        f"{run.saved[index].method} {run.saved[index].record_id}",
                        pool.submit(
                            score_one,
                            judge,
                            run.saved[index],
                            run.run,
                            splits[index],
                            corpus,
                            budgets[index],
                            graph_items[index],
                            encoder,
                        ),
                    )
                    for index in ready
                ]
                print(f"  {run.saved[0].method}", flush=True)
                drain(tasks, 25)
            for index, task in zip(ready, tasks):
                run.scores[index] = outcome(task)
                failures += run.scores[index] is None
        reports.append(summarize(dataset, runs, args.bootstrap, rng))
        all_scores.extend(score for run in runs for score in run.scores if score is not None)
    costs = [entry.cost for entry in cache.entries]
    report = Report(
        created_at=stamp,
        prompt_version=PROMPT_VERSION,
        model=judge.model_name,
        bootstrap_samples=args.bootstrap,
        seed=args.seed,
        runs=[_relative(path) for path in args.runs],
        definitions=DEFINITIONS,
        judge_cost_usd=(
            sum(cost.estimated_usd or 0 for cost in costs)
            if all(cost.estimated_usd is not None for cost in costs)
            else None
        ),
        datasets=reports,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "per_question.jsonl").write_text(
        "".join(score.model_dump_json() + "\n" for score in all_scores), encoding="utf-8"
    )
    (output / "summary.json").write_text(report.model_dump_json(indent=2))
    # Hand-check queues: judge/offset disagreements and summary-credited sentences.
    (output / "disagreements.jsonl").write_text(
        "".join(
            score.model_dump_json() + "\n"
            for score in all_scores
            if not score.has_untraceable_items
            and any(
                (g.offset_fully_covered and g.full_score < 3)
                or (not g.offset_covered and g.full_score >= 2)
                for g in score.gold
            )
        ),
        encoding="utf-8",
    )
    (output / "summary_credit.jsonl").write_text(
        "".join(
            score.model_dump_json() + "\n"
            for score in all_scores
            if any("summary" in g.credited_kinds for g in score.gold)
        ),
        encoding="utf-8",
    )
    print_table(report)
    if report.judge_cost_usd is not None:
        print(f"\nJudge cost so far (all cached calls): ${report.judge_cost_usd:.2f}")
    print(f"Wrote {_relative(output)}")
    if failures:
        print(f"{failures} calls failed; rerun the same command to resume.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
