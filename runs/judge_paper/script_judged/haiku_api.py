"""Haiku gold-fact judge by direct API call (replaces the subagent launcher).

Same judge as before: model Claude Haiku, the same items-N.md file (prompt
version and JUDGE_GUIDE on top, then the items), the same verdict format and
the same checks. Only the delivery changes: instead of a Claude Code subagent
reading the file with its Read tool and writing verdict-N.json, the whole file
is sent as one user message and the reply is the JSON array.

The call goes to the Anthropic Messages API at $ANTHROPIC_BASE_URL (the local
relay) with $ANTHROPIC_AUTH_TOKEN. Every raw reply is kept in
judge/api_raw/verdict-N.json (model, request settings, usage, text), so each
verdict can be traced to the exact call.

A file whose verdict already passes the check is skipped, so the script can be
rerun any time (it holds no state outside the files). A reply that fails to
parse or fails the check is retried once; a file that fails twice is reported
and left without a verdict.

Answers (--stage answers): the same for answers/items-answers-N.json. The
message is the file's "guide" followed by its "answers" list as JSON; the
reply must hold one {"id", "correct": true|false} per answer id. Written to
answers/verdicts-answers-N.json, raw replies to answers/api_raw/.

Usage (repo root):
  PYTHONPATH=. .venv/bin/python runs/judge_paper/scripts/haiku_api.py FIRST LAST [--stage judge|answers] [--minutes M] [--out-dir DIR]
--minutes stops starting new files after M minutes (default 8), so each call of
the script stays short. --out-dir writes verdicts elsewhere (for agreement tests).
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from evals.haiku_fact_judge import JudgeVerdict

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 32000
FOLDER = Path("runs/judge_paper/judge")
INSTRUCTION = (
    "\n\nReply with the JSON array described in the instructions at the top, one "
    "object per item in this file, and nothing else: no prose, no code fence. "
    "Quotes must be copied character for character from the item text; a line "
    "break in the file counts as a space."
)


def call(text: str) -> dict:
    body = {"model": MODEL, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": text + INSTRUCTION}]}
    req = urllib.request.Request(
        os.environ["ANTHROPIC_BASE_URL"].rstrip("/") + "/v1/messages",
        data=json.dumps(body).encode(),
        headers={"x-api-key": os.environ["ANTHROPIC_AUTH_TOKEN"], "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
    )
    for wait in (0, 20, 60, 120):
        time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return {"request": {k: v for k, v in body.items() if k != "messages"} | {"instruction": INSTRUCTION},
                        "response": json.load(r)}
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            err = e
            if isinstance(e, urllib.error.HTTPError) and e.code < 500 and e.code != 429:
                break
    raise RuntimeError(f"API call failed: {err!r}")


def check(rows: list, items: dict) -> None:
    """Same rules as check_verdicts.py (judge stage)."""
    ids = [r["id"] for r in rows]
    if sorted(ids) != sorted(items):
        raise ValueError(f"ids differ: missing {sorted(set(items) - set(ids))[:3]}, "
                         f"unknown {sorted(set(ids) - set(items))[:3]}, repeated {len(ids) - len(set(ids))}")
    for r in rows:
        v = JudgeVerdict.model_validate(r)
        labels = sorted(l for g in items[v.id]["gold"] for l in g["facts"])
        if sorted(f.fact for f in v.facts) != labels:
            raise ValueError(f"{v.id}: fact labels differ")


def check_answers(rows: list, items: dict) -> None:
    ids = [r["id"] for r in rows]
    if sorted(ids) != sorted(items):
        raise ValueError(f"ids differ: missing {sorted(set(items) - set(ids))[:3]}, unknown {sorted(set(ids) - set(items))[:3]}")
    for r in rows:
        if set(r) != {"id", "correct"} or not isinstance(r["correct"], bool):
            raise ValueError(f"bad row {r}")


def parse(text: str) -> list:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--stage", choices=["judge", "answers"], default="judge")
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args()
    folder = FOLDER if a.stage == "judge" else FOLDER.parent / "answers"
    a.out_dir = a.out_dir or folder
    raw_dir = a.out_dir / "api_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for n in range(a.first, a.last + 1):
        if a.stage == "judge":
            items_path, out = folder / f"items-{n:03d}.json", a.out_dir / f"verdict-{n:03d}.json"
        else:
            items_path, out = folder / f"items-answers-{n:03d}.json", a.out_dir / f"verdicts-answers-{n:03d}.json"
        if not items_path.exists():
            continue
        data = json.loads(items_path.read_text())
        if a.stage == "judge":
            items, ok_check = {i["id"]: i for i in data["items"]}, check
            text = (folder / f"items-{n:03d}.md").read_text()
        else:
            items, ok_check = {i["id"]: i for i in data["answers"]}, check_answers
            assert not data.get("evidence"), "evidence items are not handled"
            text = data["guide"] + "\n\nItems (answers list from items.json):\n" + json.dumps(data["answers"], ensure_ascii=False, indent=1)
        try:
            ok_check(json.loads(out.read_text()), items)
            print(f"skip {n} (verdict ok)", flush=True)
            continue
        except Exception:  # noqa: BLE001 - missing or bad: judge it
            pass
        if time.time() - start > a.minutes * 60:
            print(f"stop at {n}: time budget used", flush=True)
            break
        for attempt in (1, 2):
            t0 = time.time()
            try:
                got = call(text)
                reply = "".join(c.get("text", "") for c in got["response"]["content"])
                (raw_dir / f"{out.stem}.try{attempt}.json").write_text(json.dumps(got, ensure_ascii=False, indent=1))
                rows = parse(reply)
                ok_check(rows, items)
            except Exception as e:  # noqa: BLE001 - retry once, then report
                print(f"BAD {n} try {attempt}: {repr(e)[:200]}", flush=True)
                continue
            out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
            u = got["response"]["usage"]
            print(f"ok {n}: {len(rows)} items, {got['response']['model']}, in {u['input_tokens']} out {u['output_tokens']}, "
                  f"{time.time() - t0:.0f}s", flush=True)
            break


if __name__ == "__main__":
    main()
