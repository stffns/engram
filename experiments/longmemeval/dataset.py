"""Dataset loading for LongMemEval.

Two paths:

- ``load_fixture(path)`` — load a tiny synthetic JSON file. Used by tests
  and by anyone who wants to dry-run the runner without touching the real
  dataset. The fixture format is a strict subset of the LongMemEval JSON
  schema, so the same parser handles both.
- ``load_longmemeval(cache_dir, subset)`` — download the real LongMemEval
  dataset from HuggingFace. **Currently a stub** — implemented in a
  follow-up commit, once we've actually validated the parser against the
  fixture path. The stub raises ``NotImplementedError`` with the exact
  steps to fetch the file manually in the meantime.

The Conversation dataclass is the *only* thing the runner depends on. As
long as both loaders return ``list[Conversation]``, we can swap dataset
sources without touching the eval logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Turn:
    """One message inside a session."""

    role: str
    content: str


@dataclass
class Conversation:
    """One LongMemEval question with its full haystack of sessions.

    The runner ingests every turn of every haystack session into a fresh
    ``Memory`` instance, then asks the question and checks whether any of
    the top-k recall hits come from a session in ``answer_session_ids``.
    """

    question_id: str
    question: str
    answer: str
    answer_session_ids: list[str]
    haystack_sessions: dict[str, list[Turn]] = field(default_factory=dict)
    question_type: str | None = None

    @property
    def n_turns(self) -> int:
        return sum(len(s) for s in self.haystack_sessions.values())

    @property
    def n_sessions(self) -> int:
        return len(self.haystack_sessions)


def _parse_record(record: dict[str, Any]) -> Conversation:
    """Parse one JSON record into a ``Conversation``.

    Tolerates both the fixture format (``haystack_sessions`` as a dict of
    session_id → turns) and the LongMemEval format (``haystack_sessions``
    as a list of session arrays paired with ``haystack_session_ids``).
    """
    qid = str(record["question_id"])
    question = record["question"]
    answer = record.get("answer", "")
    answer_session_ids = [str(s) for s in record.get("answer_session_ids", [])]
    question_type = record.get("question_type")

    raw_sessions = record["haystack_sessions"]
    sessions: dict[str, list[Turn]] = {}

    if isinstance(raw_sessions, dict):
        for sid, turns in raw_sessions.items():
            sessions[str(sid)] = [Turn(role=t["role"], content=t["content"]) for t in turns]
    elif isinstance(raw_sessions, list):
        ids = [str(s) for s in record.get("haystack_session_ids", [])]
        if len(ids) != len(raw_sessions):
            raise ValueError(
                f"haystack_session_ids ({len(ids)}) does not match "
                f"haystack_sessions ({len(raw_sessions)}) for {qid}"
            )
        for sid, turns in zip(ids, raw_sessions, strict=True):
            sessions[sid] = [Turn(role=t["role"], content=t["content"]) for t in turns]
    else:
        raise TypeError(f"unexpected haystack_sessions type for {qid}: {type(raw_sessions)}")

    return Conversation(
        question_id=qid,
        question=question,
        answer=answer,
        answer_session_ids=answer_session_ids,
        haystack_sessions=sessions,
        question_type=question_type,
    )


def load_fixture(path: str | Path) -> list[Conversation]:
    """Load a tiny synthetic LongMemEval-shaped fixture from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"fixture root must be a list, got {type(data).__name__}")
    return [_parse_record(rec) for rec in data]


def load_longmemeval(
    cache_dir: str | Path | None = None,  # noqa: ARG001
    subset: str = "longmemeval_s",  # noqa: ARG001
) -> list[Conversation]:
    """Download and parse the real LongMemEval dataset.

    **Stub.** Real implementation lands in a follow-up once the runner is
    validated against the fixture path. To run a real benchmark today,
    download the JSON manually and use ``load_fixture(path)``.

    See: https://huggingface.co/datasets/xiaowu0162/longmemeval
    """
    raise NotImplementedError(
        "load_longmemeval is not implemented yet. Download the dataset "
        "manually from https://huggingface.co/datasets/xiaowu0162/longmemeval "
        "and use load_fixture(path) instead. Tracking in experiments/"
        "longmemeval/README.md."
    )
