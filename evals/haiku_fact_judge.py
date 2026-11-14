"""Gold facts found, judged by Claude Haiku subagents. No OpenAI calls.

The one retrieval measure: for each question, the share of its gold facts
that the retrieved text states. Two stages, the same for every method:

1. claims: each question's gold sentences are split into short facts once,
   and those facts are frozen, so every method is judged on the same facts.
2. judge: per method and question, each fact is marked stated, missing or
   contradicted, with a quote copied from the retrieved text. A fact counts
   as found only when the judge says stated AND its quote appears in the
   retrieved item (normalized substring check in code). The judge never
   picks the score.

The retrieved text is what the answer model was given. Items are shuffled
with a fixed seed so rank order is hidden. Identical (question, retrieved
text) pairs are judged once.

Every file a subagent reads holds the exact instructions it follows
(``guide``), and every reply is kept as written, so the judging can be
audited and repeated. Workflow, from the repo root:

  uv run python -m evals.haiku_fact_judge claims OUT RUN_DIR... [--type T]
  (Haiku subagents write OUT/claims/verdict-*.json)
  uv run python -m evals.haiku_fact_judge judge OUT RUN_DIR... [--type T]
  (Haiku subagents write OUT/judge/verdict-*.json)
  uv run python -m evals.haiku_fact_judge score OUT
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from benchmark import SavedRetrieval
from evals.retrieval_judge import render_graphrag
from evals.retrieval_quality import load_run, normalize

PROMPT_VERSION = "2026-11-10.haiku.2"
JUDGE_MODEL = "claude-haiku (Claude Code subagent)"

CLAIMS_GUIDE = """You split gold evidence sentences into the separate facts each one states.

For every item in this file, split each numbered sentence into facts. Cover
every fact in the sentence: do not drop names, numbers, dates, places, or
qualifiers such as 'former', 'first', or 'only'. Each fact is one short
self-contained statement in plain words that names its subject instead of
using a pronoun. Do not add anything the sentence does not say.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "sentences": [["fact", "fact"], ["fact"]]}
The outer list has one entry per sentence, in sentence order (sentence 0 first).
Use only ids from this file and do not skip any item or sentence."""

JUDGE_GUIDE = """You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item."""

Status = Literal["stated", "missing", "contradicted"]


class ClaimsTask(BaseModel):
    id: str
    sentences: list[str]


class ClaimsVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    sentences: list[list[str]]


class Quote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item: int
    quote: str


class FactVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact: str
    status: Status
    evidence: list[Quote] = []


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    facts: list[FactVerdict]


class JudgeTask(BaseModel):
    id: str
    gold: list[dict]  # [{"sentence": str, "facts": {"S.F": str}}]
    items: list[str]  # retrieved items, numbered from 1 in this order


def _hash(*parts: str) -> str:
    return sha256("\0".join(parts).encode()).hexdigest()[:16]


def answer_items(saved: SavedRetrieval) -> list[str]:
    """The retrieved text the answer model was given, one entry per item."""
    if saved.method == "graphrag":
        texts = [render_graphrag(passage) for passage in saved.retrieval.passages]
    else:
        texts = list(saved.retrieval.passages)
    seen: set[str] = set()
    unique = [t for t in texts if t.strip() and not (t in seen or seen.add(t))]
    random.Random(f"{saved.dataset}:{saved.record_id}:{saved.method}").shuffle(unique)
    return unique


def _runs(run_dirs: list[Path], kind: str | None) -> list[tuple[str, SavedRetrieval]]:
    rows = []
    for run_dir in run_dirs:
        for saved in load_run(run_dir):
            if kind is None or saved.question_type == kind:
                rows.append((str(run_dir), saved))
    return rows


def _write_batches(folder: Path, guide: str, tasks: list[BaseModel], size: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for number, start in enumerate(range(0, len(tasks), size)):
        (folder / f"items-{number:03d}.json").write_text(
            json.dumps(
                {
                    "prompt_version": PROMPT_VERSION,
                    "guide": guide,
                    "items": [t.model_dump() for t in tasks[start : start + size]],
                },
                ensure_ascii=False,
                indent=1,
            )
        )


def export_claims(out: Path, run_dirs: list[Path], kind: str | None, size: int) -> None:
    tasks: dict[str, ClaimsTask] = {}
    for _, saved in _runs(run_dirs, kind):
        if saved.supporting_sentences:
            tasks.setdefault(
                saved.record_id,
                ClaimsTask(id=saved.record_id, sentences=saved.supporting_sentences),
            )
    _write_batches(out / "claims", CLAIMS_GUIDE, list(tasks.values()), size)
    print(f"{len(tasks)} questions to split -> {out / 'claims'}")


def merged_single_passage(verdict: ClaimsVerdict, sentences: int) -> ClaimsVerdict:
    """One gold passage split into several fact lists becomes one list.

    Natural Questions evidence is one long-answer paragraph per question; the
    judge sometimes returns one fact list per sentence inside it (13 Nov 2026,
    7 of 358 NQ questions). Every fact still comes from that one passage, so
    the lists are joined in order. Nothing is added or dropped, and items with
    several gold sentences are never changed."""
    if sentences == 1 and len(verdict.sentences) > 1 and all(verdict.sentences):
        return ClaimsVerdict(
            id=verdict.id, sentences=[[f for part in verdict.sentences for f in part]]
        )
    return verdict


def load_claims(out: Path) -> dict[str, list[list[str]]]:
    tasks = {
        item["id"]: item
        for path in sorted((out / "claims").glob("items-*.json"))
        for item in json.loads(path.read_text())["items"]
    }
    claims: dict[str, list[list[str]]] = {}
    for path in sorted((out / "claims").glob("verdict-*.json")):
        for row in json.loads(path.read_text()):
            verdict = merged_single_passage(
                ClaimsVerdict.model_validate(row), len(tasks[row["id"]]["sentences"])
            )
            task = tasks[verdict.id]
            if len(verdict.sentences) != len(task["sentences"]) or not all(
                verdict.sentences
            ):
                raise ValueError(f"{path.name}: {verdict.id} skipped a sentence.")
            claims[verdict.id] = verdict.sentences
    missing = set(tasks) - set(claims)
    if missing:
        raise ValueError(
            f"No claims for {len(missing)} questions: {sorted(missing)[:5]}"
        )
    return claims


def _wrapped(text: str) -> str:
    """Break long lines at spaces so a reader can page through the file.

    Quotes are checked after whitespace is collapsed, so line breaks inside a
    quote do not change whether it is found."""
    return "\n".join(
        textwrap.fill(line, 160, break_long_words=False, break_on_hyphens=False)
        if line.strip()
        else line
        for line in text.splitlines()
    )


def render_task(task: JudgeTask) -> str:
    gold = "\n".join(
        f"Sentence {s}: {g['sentence']}\n"
        + "\n".join(f"  Fact {label}: {fact}" for label, fact in g["facts"].items())
        for s, g in enumerate(task.gold)
    )
    items = "\n\n".join(
        f"<item {number}>\n{_wrapped(text)}\n</item {number}>"
        for number, text in enumerate(task.items, start=1)
    )
    return (
        f"=== ITEM id={task.id} ===\nGold sentences and facts:\n{gold}\n\n"
        f"Retrieved items:\n{items}\n=== END ITEM id={task.id} ===\n"
    )


def _write_judge_batches(
    folder: Path, tasks: list[JudgeTask], max_chars: int, first: int = 0
) -> None:
    """Batches of about max_chars of retrieved text; one large task goes alone.

    items-N.json is what the scorer verifies quotes against. items-N.md holds
    the same content with the guide on top, and is what the subagent reads."""
    folder.mkdir(parents=True, exist_ok=True)
    batches: list[list[JudgeTask]] = []
    used = 0
    for task in tasks:
        size = sum(len(text) for text in task.items)
        if not batches or used + size > max_chars:
            batches.append([])
            used = 0
        batches[-1].append(task)
        used += size
    for number, batch in enumerate(batches, start=first):
        name = f"items-{number:03d}"
        (folder / f"{name}.json").write_text(
            json.dumps(
                {
                    "prompt_version": PROMPT_VERSION,
                    "guide": JUDGE_GUIDE,
                    "items": [task.model_dump() for task in batch],
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        (folder / f"{name}.md").write_text(
            f"Prompt version {PROMPT_VERSION}\n\n{JUDGE_GUIDE}\n\n"
            f"This file has {len(batch)} items.\n\n"
            + "\n".join(render_task(task) for task in batch)
        )


def export_judge(
    out: Path, run_dirs: list[Path], kind: str | None, size: int, add: bool = False
) -> None:
    """Write judge tasks. With add=True, earlier files and verdicts are kept:
    only tasks not already exported are written, numbered after the last file,
    and the new (run, question) links are appended."""
    claims = load_claims(out)
    tasks: dict[str, JudgeTask] = {}
    links = []
    for run, saved in _runs(run_dirs, kind):
        if not saved.supporting_sentences:
            continue
        facts = claims[saved.record_id]
        gold = [
            {
                "sentence": sentence,
                "facts": {f"{s}.{f}": fact for f, fact in enumerate(facts[s])},
            }
            for s, sentence in enumerate(saved.supporting_sentences)
        ]
        items = answer_items(saved)
        task_id = _hash(saved.record_id, json.dumps(gold), *items)
        tasks.setdefault(task_id, JudgeTask(id=task_id, gold=gold, items=items))
        links.append(
            {
                "run": run,
                "method": saved.method,
                "record_id": saved.record_id,
                "question_type": saved.question_type,
                "task": task_id,
            }
        )
    folder = out / "judge"
    links_path = folder / "links.jsonl"
    if add:
        done = {
            item["id"]
            for path in folder.glob("items-*.json")
            for item in json.loads(path.read_text())["items"]
        }
        first = 1 + max(
            (int(path.stem.split("-")[1]) for path in folder.glob("items-*.json")),
            default=-1,
        )
        old_links = links_path.read_text().splitlines() if links_path.exists() else []
        seen = {
            (json.loads(line)["run"], json.loads(line)["record_id"])
            for line in old_links
        }
        links = [link for link in links if (link["run"], link["record_id"]) not in seen]
        new_tasks = [task for task in tasks.values() if task.id not in done]
        _write_judge_batches(folder, new_tasks, size, first)
        links_path.write_text(
            "".join(line + "\n" for line in old_links)
            + "".join(json.dumps(link) + "\n" for link in links)
        )
        print(
            f"added {len(links)} (run, question) pairs, {len(new_tasks)} new tasks, files from {first:03d}"
        )
        return
    _write_judge_batches(folder, list(tasks.values()), size)
    links_path.write_text("".join(json.dumps(link) + "\n" for link in links))
    print(
        f"{len(links)} (run, question) pairs, {len(tasks)} distinct -> {out / 'judge'}"
    )


def _verified(quote: Quote, items: list[str]) -> bool:
    if not 1 <= quote.item <= len(items):
        return False
    text = normalize(quote.quote)
    return bool(text) and text in normalize(items[quote.item - 1])


def score(out: Path) -> None:
    tasks = {
        item["id"]: JudgeTask.model_validate(item)
        for path in sorted((out / "judge").glob("items-*.json"))
        for item in json.loads(path.read_text())["items"]
    }
    verdicts: dict[str, JudgeVerdict] = {}
    for path in sorted((out / "judge").glob("verdict-*.json")):
        for row in json.loads(path.read_text()):
            verdict = JudgeVerdict.model_validate(row)
            labels = {label for g in tasks[verdict.id].gold for label in g["facts"]}
            given = [fact.fact for fact in verdict.facts]
            if sorted(given) != sorted(labels):
                raise ValueError(
                    f"{path.name}: {verdict.id} facts {given} != {sorted(labels)}"
                )
            verdicts[verdict.id] = verdict
    missing = set(tasks) - set(verdicts)
    if missing:
        raise ValueError(f"No verdict for {len(missing)} tasks: {sorted(missing)[:5]}")

    per_task = {}
    for task_id, verdict in verdicts.items():
        items = tasks[task_id].items
        facts = []
        for fact in verdict.facts:
            ok = [_verified(q, items) for q in fact.evidence]
            found = fact.status == "stated" and any(ok)
            facts.append(
                {
                    "fact": fact.fact,
                    "judged": fact.status,
                    "found": found,
                    "quotes": len(ok),
                    "unverified_quotes": ok.count(False),
                }
            )
        per_task[task_id] = facts

    rows = []
    for line in (out / "judge" / "links.jsonl").read_text().splitlines():
        link = json.loads(line)
        facts = per_task[link["task"]]
        rows.append(
            {
                **link,
                "share": sum(f["found"] for f in facts) / len(facts),
                "facts": facts,
            }
        )
    (out / "per_question.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    by_run: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        by_run.setdefault((row["run"], row["question_type"]), []).append(row["share"])
    print("run | type | questions | gold facts found >=50% / >=75% / 100%")
    for (run, kind), shares in sorted(by_run.items()):
        counts = [sum(s >= t - 1e-9 for s in shares) for t in (0.5, 0.75, 1.0)]
        print(
            f"{run} | {kind} | {len(shares)} | {counts[0]} / {counts[1]} / {counts[2]}"
        )
    judged = Counter(f["judged"] for facts in per_task.values() for f in facts)
    unverified = sum(
        f["judged"] == "stated" and not f["found"]
        for facts in per_task.values()
        for f in facts
    )
    print(
        f"judge statuses {dict(judged)}; stated but quote not found in text: {unverified}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["claims", "judge", "score"])
    parser.add_argument("out", type=Path)
    parser.add_argument("run_dirs", type=Path, nargs="*")
    parser.add_argument("--type", dest="kind")
    parser.add_argument(
        "--add",
        action="store_true",
        help="judge: keep earlier files and verdicts, export only new tasks",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=10,
        help="claims: questions per file; judge: characters of retrieved text per file",
    )
    args = parser.parse_args()
    if args.command == "claims":
        export_claims(args.out, args.run_dirs, args.kind, args.batch)
    elif args.command == "judge":
        export_judge(args.out, args.run_dirs, args.kind, args.batch, args.add)
    else:
        score(args.out)


if __name__ == "__main__":
    main()
