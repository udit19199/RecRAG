"""Dataset record types, loaders, and the shared source registry.

Keeping these small adapters together makes it easier to compare their record
shapes without scattering the active dataset interface across modules.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

MULTIHOP_RAG_QUERY_TYPES = (
    "inference_query",
    "comparison_query",
    "temporal_query",
    "null_query",
)


@dataclass(slots=True)
class SourcePage:
    title: str
    passages: list[str]


class DatasetRecord(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def question(self) -> str: ...

    @property
    def pages(self) -> Sequence[SourcePage]: ...

    @property
    def answer(self) -> str | None: ...

    def answer_aliases(self) -> Sequence[str]: ...

    def supporting_sentences(self) -> Sequence[str]: ...


@dataclass(slots=True, frozen=True)
class HotpotRecord:
    id: str
    question: str
    pages: list[SourcePage]
    question_type: str
    level: str
    answer: str
    evidence: list[str]

    def answer_aliases(self) -> list[str]:
        return [self.answer]

    def supporting_sentences(self) -> list[str]:
        return self.evidence


def load_hotpotqa_record(index: int) -> HotpotRecord:
    return _hotpot_record(_hotpot_table()[index])


@cache
def load_hotpotqa_corpus() -> list[SourcePage]:
    pages: list[SourcePage] = []
    seen: set[str] = set()
    for row in _hotpot_table():
        for page in _hotpot_record(row).pages:
            text = page.title + "\n" + "\n".join(page.passages)
            key = sha256(text.encode()).hexdigest()
            if key not in seen:
                pages.append(page)
                seen.add(key)
    return pages


@cache
def _hotpot_table():
    dataset_dir = Path(__file__).parent / "datasets/hotpotqa"
    for filename in (
        "hotpotqa_distractor_validation_sample_10.json",
        "hotpotqa_distractor_validation_first_1000.json",
    ):
        local_rows = dataset_dir / filename
        if local_rows.exists():
            return json.loads(local_rows.read_text(encoding="utf-8"))
    from datasets import load_dataset

    return load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")


def _hotpot_record(row) -> HotpotRecord:
    context = row["context"]
    facts = row["supporting_facts"]
    pages = [
        SourcePage(title=title, passages=sentences)
        for title, sentences in zip(context["title"], context["sentences"])
    ]
    return HotpotRecord(
        id=row["id"],
        question=row["question"],
        pages=pages,
        question_type=row["type"],
        level=row["level"],
        answer=row["answer"],
        evidence=[
            page.passages[index]
            for title, index in zip(facts["title"], facts["sent_id"])
            for page in pages
            if page.title == title
        ],
    )


@dataclass(slots=True, frozen=True)
class MultiHopRAGRecord:
    id: str
    question: str
    answer: str
    question_type: str
    pages: list[SourcePage]
    evidence: list[str]

    def answer_aliases(self) -> list[str]:
        return [self.answer]

    def supporting_sentences(self) -> list[str]:
        return self.evidence


def load_multihop_rag_record(index: int) -> MultiHopRAGRecord:
    row = _multihop_queries()[index]
    evidence = row["evidence_list"]
    return MultiHopRAGRecord(
        id=f"multihop-rag-{index}",
        question=row["query"],
        answer=row["answer"],
        question_type=_multihop_query_type(row["question_type"]),
        pages=_multihop_corpus(),
        evidence=[item["fact"] for item in evidence],
    )


def _multihop_query_type(value: str) -> str:
    if value not in MULTIHOP_RAG_QUERY_TYPES:
        raise ValueError(
            f"Unknown MultiHop-RAG question_type {value!r}; expected one of "
            f"{', '.join(MULTIHOP_RAG_QUERY_TYPES)}."
        )
    return value


@cache
def _multihop_queries():
    from huggingface_hub import hf_hub_download

    # Read the same raw query file as the reference, preserving its row order.
    path = hf_hub_download(
        repo_id="yixuantt/MultiHopRAG", repo_type="dataset", filename="MultiHopRAG.json"
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


@cache
def _multihop_corpus() -> list[SourcePage]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="yixuantt/MultiHopRAG", repo_type="dataset", filename="corpus.json"
    )
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        SourcePage(
            title=f"{row['title']} ({row['source']}, {row['published_at']})",
            passages=[row["body"]],
        )
        for row in rows
    ]


@dataclass(slots=True, frozen=True)
class NaturalQuestionsRecord:
    id: str
    question: str
    pages: list[SourcePage]
    answer: str
    aliases: list[str]
    evidence: list[str]

    def answer_aliases(self) -> list[str]:
        return self.aliases

    def supporting_sentences(self) -> list[str]:
        return self.evidence


@dataclass(slots=True, frozen=True)
class NovelQARecord:
    id: str
    question: str
    pages: list[SourcePage]
    question_type: str
    answer: str | None = None
    complexity: str | None = None

    def answer_aliases(self) -> list[str]:
        return [self.answer] if self.answer else []

    def supporting_sentences(self) -> list[str]:
        return []


@dataclass(slots=True, frozen=True)
class _NovelQAQuestion:
    id: str
    book_id: str
    question: str
    question_type: str
    answer: str | None = None
    complexity: str | None = None


def load_novelqa_record(index: int) -> NovelQARecord:
    row = _novelqa_questions()[index]
    return NovelQARecord(
        id=row.id,
        question=row.question,
        pages=[_novelqa_page(row.book_id)],
        question_type=row.question_type,
        answer=row.answer,
        complexity=row.complexity,
    )


@cache
def _novelqa_root() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id="NovelQA/NovelQA",
            repo_type="dataset",
            allow_patterns=["bookmeta.json", "Books/PublicDomain/*.txt", "Data/**"],
        )
    )


@cache
def _novelqa_questions() -> list[_NovelQAQuestion]:
    questions: list[_NovelQAQuestion] = []
    root = _novelqa_root()
    book_ids = {path.stem for path in (root / "Books").rglob("*.txt")}
    for path in sorted((root / "Data").rglob("*.json")):
        if path.stem not in book_ids:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = (
            payload
            if isinstance(payload, list)
            else [
                {"QID": qid, **row}
                for qid, row in payload.items()
                if isinstance(row, dict)
            ]
        )
        for row in rows:
            qid = str(row["QID"])
            question = row["Question"]
            options = row.get("Options", {})
            if options:
                question += "\nOptions:\n" + "\n".join(
                    f"{label}. {text}" for label, text in options.items()
                )
            questions.append(
                _NovelQAQuestion(
                    id=f"novelqa-{path.stem}-{qid}",
                    book_id=path.stem,
                    question=question,
                    question_type=str(row.get("Aspect", "unknown")),
                    answer=row.get("Gold"),
                    complexity=row.get("Complexity"),
                )
            )
    if not questions:
        raise ValueError(
            "NovelQA has no public-domain question files with matching book text. "
            "Check Hugging Face access to NovelQA/NovelQA."
        )
    return questions


@cache
def _novelqa_page(book_id: str) -> SourcePage:
    root = _novelqa_root()
    path = next(
        path for path in (root / "Books").rglob("*.txt") if path.stem == book_id
    )
    metadata = json.loads((root / "bookmeta.json").read_text(encoding="utf-8"))
    title = book_id
    if isinstance(metadata, dict):
        for key, row in metadata.items():
            if isinstance(row, dict) and str(row.get("BID", key)) == book_id:
                title = row.get("title", book_id)
                break
    else:
        for row in metadata:
            if isinstance(row, dict) and str(row.get("BID", "")) == book_id:
                title = row.get("title", book_id)
                break
    return SourcePage(title=title, passages=[path.read_text(encoding="utf-8")])


def load_natural_questions_record(index: int) -> NaturalQuestionsRecord:
    return _natural_questions_record(_natural_questions_table()[index])


@cache
def _natural_questions_table():
    from datasets import load_dataset

    return load_dataset(
        "google-research-datasets/natural_questions", "default", split="validation"
    )


def _natural_questions_record(row) -> NaturalQuestionsRecord:
    document = row["document"]
    tokens = document["tokens"]
    text = " ".join(
        token
        for token, is_html in zip(tokens["token"], tokens["is_html"])
        if not is_html and token
    )
    annotation_columns = row["annotations"]
    aliases = []
    evidence = []
    long_answers = []
    for index, yes_no in enumerate(annotation_columns["yes_no_answer"]):
        if yes_no in {0, 1}:
            answer = "no" if yes_no == 0 else "yes"
            if answer not in aliases:
                aliases.append(answer)
        short_answers = annotation_columns["short_answers"]
        answer_texts = (
            short_answers[index]["text"]
            if isinstance(short_answers, list)
            else short_answers["text"][index]
        )
        for answer_text in answer_texts:
            if answer_text and answer_text not in aliases:
                aliases.append(answer_text)
        long_answer = annotation_columns["long_answer"]
        start = (
            long_answer[index]["start_token"]
            if isinstance(long_answer, list)
            else long_answer["start_token"][index]
        )
        end = (
            long_answer[index]["end_token"]
            if isinstance(long_answer, list)
            else long_answer["end_token"][index]
        )
        if start >= 0 and end > start:
            passage = " ".join(
                token
                for token, is_html in zip(
                    tokens["token"][start:end], tokens["is_html"][start:end]
                )
                if not is_html and token
            )
            if passage and passage not in evidence:
                evidence.append(passage)
            if passage and passage not in long_answers:
                long_answers.append(passage)
    # Later annotators may provide short answers. Long evidence must not mask them.
    if not aliases:
        aliases.extend(long_answers)
    if not aliases:
        aliases.append("No answer")
    return NaturalQuestionsRecord(
        id=str(row["id"]),
        question=row["question"]["text"],
        pages=[SourcePage(title=document["title"], passages=[text])],
        answer=aliases[0],
        aliases=aliases,
        evidence=evidence,
    )


@dataclass(slots=True, frozen=True)
class DatasetSource:
    name: str
    display_name: str
    load_record: Callable[[int], DatasetRecord]


DATASET_SOURCES: list[DatasetSource] = [
    DatasetSource(
        name="hotpotqa",
        display_name="HotpotQA",
        load_record=load_hotpotqa_record,
    ),
    DatasetSource(
        name="multihop_rag",
        display_name="MultiHop-RAG",
        load_record=load_multihop_rag_record,
    ),
    DatasetSource(
        name="natural_questions",
        display_name="Natural Questions",
        load_record=load_natural_questions_record,
    ),
    DatasetSource(
        name="novelqa",
        display_name="NovelQA",
        load_record=load_novelqa_record,
    ),
]


def get_source(name: str, *, paper_data: Path | None = None) -> DatasetSource:
    for source in DATASET_SOURCES:
        if source.name == name:
            if paper_data is not None:
                if name not in {"multihop_rag", "natural_questions", "novelqa"}:
                    raise ValueError(
                        "--paper-data supports MultiHop-RAG, NQ, and NovelQA."
                    )
                root = paper_data.resolve()
                if not root.is_relative_to(Path(__file__).resolve().parent):
                    raise ValueError(
                        "Keep processed dataset files inside this repository."
                    )
                return DatasetSource(
                    name=name,
                    display_name=source.display_name,
                    load_record=lambda index, root=root: _paper_records(name, root)[
                        index
                    ],
                )
            return source
    raise ValueError(f"Unknown dataset: {name}")


@dataclass(slots=True, frozen=True)
class PaperRecord:
    id: str
    question: str
    pages: list[SourcePage]
    answer: str | None
    aliases: list[str]
    evidence: list[str]
    question_type: str | None = None
    complexity: str | None = None

    def answer_aliases(self) -> list[str]:
        return self.aliases

    def supporting_sentences(self) -> list[str]:
        return self.evidence


def _paper_json(root: Path, filename: str):
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing processed paper input: {path}")
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_hook=lambda fields: SimpleNamespace(**fields),
    )


def _paper_text(value) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "\n\n".join(value)
    raise ValueError(
        "Processed context must be text or a list of text passages. The authors' JSONReader is unpublished."
    )


@cache
def _paper_records(name: str, root: Path) -> list[PaperRecord]:
    records: list[PaperRecord] = []
    if name == "multihop_rag":
        corpus = _paper_json(root, "corpus.json")
        pages = [
            SourcePage(
                title=f"{row.title} ({row.source}, {row.published_at})",
                passages=[row.body],
            )
            for row in corpus
        ]
        for index, row in enumerate(_paper_json(root, "MultiHopRAG.json")):
            records.append(
                PaperRecord(
                    id=str(getattr(row, "id", index)),
                    question=row.query,
                    pages=pages,
                    answer=row.answer,
                    aliases=[row.answer],
                    evidence=[item.fact for item in getattr(row, "evidence_list", [])],
                    question_type=_multihop_query_type(row.question_type),
                )
            )
    elif name == "natural_questions":
        for document_id, row in enumerate(_paper_json(root, "NQ.json")):
            context = getattr(
                row, "context", getattr(row, "content", getattr(row, "text", None))
            )
            pages = [
                SourcePage(
                    title=str(getattr(row, "title", document_id)),
                    passages=[_paper_text(context)],
                )
            ]
            if len(row.questions) != len(row.answer):
                raise ValueError("NQ questions and answers have different lengths.")
            for index, question in enumerate(row.questions):
                raw_answer = row.answer[index]
                aliases = raw_answer if isinstance(raw_answer, list) else [raw_answer]
                records.append(
                    PaperRecord(
                        id=f"{document_id}_{index}",
                        question=question,
                        pages=pages,
                        answer=aliases[0] if aliases else None,
                        aliases=aliases,
                        evidence=[],
                    )
                )
    else:
        contexts = _paper_json(root, "NovelQA_contexts.json")
        questions = _paper_json(root, "NovelQA_qa.json")
        if not isinstance(contexts, SimpleNamespace) or not isinstance(
            questions, SimpleNamespace
        ):
            raise ValueError(
                "NovelQA processed files must map book IDs to contexts and questions."
            )
        for book_id, book_questions in vars(questions).items():
            context = getattr(
                contexts, book_id, getattr(contexts, f"{book_id}.txt", None)
            )
            pages = [SourcePage(title=book_id, passages=[_paper_text(context)])]
            for question_id, row in vars(book_questions).items():
                options = getattr(row, "Options", None)
                question = row.Question
                if options:
                    question += "\nOptions:\n" + "\n".join(
                        f"{label}. {text}" for label, text in vars(options).items()
                    )
                answer = getattr(row, "Gold", None)
                records.append(
                    PaperRecord(
                        id=f"{book_id}-{question_id}",
                        question=question,
                        pages=pages,
                        answer=answer,
                        aliases=[answer] if answer else [],
                        evidence=[],
                        question_type=getattr(row, "Aspect", None),
                        complexity=getattr(row, "Complexity", None),
                    )
                )
    if not records:
        raise ValueError("Processed paper inputs contain no questions.")
    return records
