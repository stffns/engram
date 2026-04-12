"""LongMemEval R@k runner.

The job: take a list of ``Conversation`` objects, ingest each one's
haystack into a fresh ``Memory``, recall the question, and check whether
any of the top-k hits come from a session in ``answer_session_ids``.

Three baselines are wired in:

- ``vstash`` — bypass engram entirely; ingest straight into
  ``vstash.Memory``. Tells us how much engram's loop costs vs the bare
  substrate.
- ``engram-always`` — engram with ``AlwaysWrite``. The "store-everything"
  control: every event lands, no filtering.
- ``engram-heuristic`` — engram with the default ``HeuristicWriteDecider``
  (Phase 1). The thing we actually ship.

Each baseline reports R@k with a 95% bootstrap confidence interval. The
runner is intentionally CLI-only and prints to stdout — appending to
``RESULTS.md`` is a manual step (CONSTITUTION §9 honesty discipline:
results land in the file with full provenance, not auto-appended).
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import vstash

from engram import AlwaysWrite, HeuristicWriteDecider, Memory
from experiments.retrieval.longmemeval.dataset import (
    Conversation,
    Turn,
    load_fixture,
    load_longmemeval,
)

# Title format used to tag every ingested turn so we can attribute recall
# hits back to a session. Pinned here so the eval and the ingest agree.
_TITLE_FMT = "{qid}::{sid}::{turn_idx}"


def _title_for(qid: str, sid: str, turn_idx: int) -> str:
    return _TITLE_FMT.format(qid=qid, sid=sid, turn_idx=turn_idx)


def _session_from_title(title: str | None) -> str | None:
    if not title or "::" not in title:
        return None
    parts = title.split("::", 2)
    if len(parts) < 2:
        return None
    return parts[1]


_SPECIAL_TOKENS = re.compile(r"<\|[a-z_]+\|>")


def _format_turn(turn: Turn) -> str:
    text = f"{turn.role}: {turn.content}"
    # Strip tiktoken special tokens that appear in some LongMemEval
    # haystacks (e.g. <|endoftext|>) — these crash vstash's chunk_text.
    return _SPECIAL_TOKENS.sub("", text)


# --------------------------------------------------------------------- adapters


@dataclass
class IngestStats:
    attempted: int
    written: int
    skipped: int


class _Adapter:
    """Wraps a memory backend so the runner doesn't care which one it is."""

    name: str

    def remember(self, text: str, *, title: str) -> bool:
        """Returns True if the row was actually written, False if skipped."""
        raise NotImplementedError

    def remember_batch(self, items: list[tuple[str, str]]) -> int:
        """Batch ingest [(text, title), ...]. Returns count written.

        Default implementation falls back to sequential remember().
        Adapters with native batch support override this.
        """
        written = 0
        for text, title in items:
            if self.remember(text, title=title):
                written += 1
        return written

    def recall(self, query: str, *, top_k: int) -> list[Any]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _VstashAdapter(_Adapter):
    name = "vstash"

    def __init__(self, project: str, db: Path) -> None:
        self._m = vstash.Memory(project=project, db=db, collection="default")

    def remember(self, text: str, *, title: str) -> bool:
        self._m.remember(text, title=title, collection="default")
        return True

    def remember_batch(self, items: list[tuple[str, str]]) -> int:
        """Batch ingest using vstash 0.28.0 store-level API.

        Falls back to sequential remember() if batch API is unavailable
        or if accessing store internals fails (vstash API change).
        """
        try:
            store = self._m._store
            batch_fn = getattr(store, "add_documents_batch", None)
            if batch_fn is None:
                return super().remember_batch(items)

            from vstash.embed import embed_texts
            from vstash.ingest import chunk_text

            model = store.get_meta("embedding_model")
            if not model:
                return super().remember_batch(items)

            docs = []
            for text, title in items:
                chunks = chunk_text(text)
                embeddings = embed_texts(chunks, model)
                docs.append({
                    "path": f"text://{title}",
                    "title": title,
                    "chunks": chunks,
                    "embeddings": embeddings,
                    "source_type": "text",
                    "collection": "default",
                    "project": store._project,
                    "layer": "episodic",
                })
            if docs:
                batch_fn(docs)
            return len(docs)
        except Exception:
            return super().remember_batch(items)

    def recall(self, query: str, *, top_k: int) -> list[Any]:
        return self._m.search(query, top_k=top_k, collection="default")

    def close(self) -> None:
        self._m.close()


class _EngramAdapter(_Adapter):
    def __init__(self, project: str, db: Path, decider) -> None:
        self._m = Memory(project=project, db=db, write_decider=decider)

    def remember(self, text: str, *, title: str) -> bool:
        return self._m.remember(text, title=title).written

    def recall(self, query: str, *, top_k: int) -> list[Any]:
        return self._m.recall(query, top_k=top_k)

    def close(self) -> None:
        self._m.close()


class _EngramAlwaysAdapter(_EngramAdapter):
    name = "engram-always"

    def __init__(self, project: str, db: Path) -> None:
        super().__init__(project, db, AlwaysWrite())


class _EngramHeuristicAdapter(_EngramAdapter):
    name = "engram-heuristic"

    def __init__(self, project: str, db: Path) -> None:
        super().__init__(project, db, HeuristicWriteDecider())


_ADAPTERS: dict[str, Callable[[str, Path], _Adapter]] = {
    "vstash": _VstashAdapter,
    "engram-always": _EngramAlwaysAdapter,
    "engram-heuristic": _EngramHeuristicAdapter,
}


# --------------------------------------------------------------------- eval core


@dataclass
class QuestionResult:
    question_id: str
    hit: bool
    ingest: IngestStats
    elapsed_s: float


@dataclass
class BaselineResult:
    baseline: str
    n_questions: int
    top_k: int
    r_at_k: float
    ci_low: float
    ci_high: float
    total_elapsed_s: float
    per_question: list[QuestionResult]


def eval_question(
    adapter: _Adapter,
    conv: Conversation,
    *,
    top_k: int,
) -> QuestionResult:
    t0 = time.perf_counter()

    # Collect all turns, then batch-ingest if the adapter supports it.
    items: list[tuple[str, str]] = []
    for sid, turns in conv.haystack_sessions.items():
        for turn_idx, turn in enumerate(turns):
            items.append((_format_turn(turn), _title_for(conv.question_id, sid, turn_idx)))

    attempted = len(items)
    written = adapter.remember_batch(items)

    hits = adapter.recall(conv.question, top_k=top_k)
    answer_set = set(conv.answer_session_ids)
    hit = any((_session_from_title(getattr(h, "title", None)) in answer_set) for h in hits)

    return QuestionResult(
        question_id=conv.question_id,
        hit=hit,
        ingest=IngestStats(attempted=attempted, written=written, skipped=attempted - written),
        elapsed_s=time.perf_counter() - t0,
    )


def bootstrap_ci(
    values: list[bool],
    *,
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a 0/1 vector."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_iter):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_iter)]
    hi = means[int((1 - alpha / 2) * n_iter) - 1]
    return (lo, hi)


def eval_baseline(
    baseline: str,
    conversations: list[Conversation],
    *,
    top_k: int,
    db_dir: Path,
) -> BaselineResult:
    if baseline not in _ADAPTERS:
        raise ValueError(f"unknown baseline {baseline!r}; choose from {sorted(_ADAPTERS)}")
    factory = _ADAPTERS[baseline]

    t0 = time.perf_counter()
    per_question: list[QuestionResult] = []

    for conv in conversations:
        # One DB per question per baseline so questions never see each
        # other's haystacks. The runner is correctness-first; if this is
        # too slow on the real 500-question set, we revisit (CONSTITUTION
        # §9 — measure before optimizing).
        db = db_dir / f"{baseline}_{conv.question_id}.db"
        adapter = factory(f"lme_{baseline}_{conv.question_id}", db)
        try:
            per_question.append(eval_question(adapter, conv, top_k=top_k))
        finally:
            adapter.close()

    hits = [q.hit for q in per_question]
    r_at_k = sum(hits) / len(hits) if hits else 0.0
    lo, hi = bootstrap_ci(hits)

    return BaselineResult(
        baseline=baseline,
        n_questions=len(per_question),
        top_k=top_k,
        r_at_k=r_at_k,
        ci_low=lo,
        ci_high=hi,
        total_elapsed_s=time.perf_counter() - t0,
        per_question=per_question,
    )


def format_result(result: BaselineResult) -> str:
    return (
        f"baseline={result.baseline:<20} "
        f"n={result.n_questions:<4} "
        f"R@{result.top_k}={result.r_at_k:.3f} "
        f"95% CI=[{result.ci_low:.3f}, {result.ci_high:.3f}] "
        f"elapsed={result.total_elapsed_s:.1f}s"
    )


# --------------------------------------------------------------------------- CLI


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="experiments.retrieval.longmemeval.runner",
        description="LongMemEval R@k runner for engram baselines.",
    )
    p.add_argument(
        "--baseline",
        action="append",
        choices=sorted(_ADAPTERS),
        help="Baseline to evaluate. Repeat to run multiple. Default: all three.",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Path to a LongMemEval-shaped JSON file. Defaults to the tiny "
        "synthetic fixture (sanity, not signal).",
    )
    src.add_argument(
        "--subset",
        choices=["longmemeval_oracle", "longmemeval_s", "longmemeval_m"],
        default=None,
        help="Real LongMemEval subset to download from HuggingFace and run "
        "against. Mutually exclusive with --fixture.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for question sampling when --questions caps the run.",
    )
    p.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Cap on the number of questions to evaluate (None = all).",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--db-dir",
        type=Path,
        default=None,
        help="Where to put per-question SQLite DBs. Default: a temp dir.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import tempfile

    args = _parse_args(argv)
    baselines = args.baseline or sorted(_ADAPTERS)

    if args.subset is not None:
        conversations = load_longmemeval(args.subset)
        source = f"subset={args.subset}"
    else:
        fixture_path = args.fixture or (
            Path(__file__).parent / "fixtures" / "tiny.json"
        )
        conversations = load_fixture(fixture_path)
        source = f"fixture={fixture_path}"

    if args.questions is not None and args.questions < len(conversations):
        # Deterministic sample so reruns at the same seed give the same
        # subset — important for honest comparisons across baselines and
        # commits.
        rng = random.Random(args.seed)
        conversations = rng.sample(conversations, args.questions)

    if not conversations:
        print("no conversations to evaluate", file=sys.stderr)
        return 2

    print(
        f"# LongMemEval — {len(conversations)} questions, top_k={args.top_k}, "
        f"{source}, seed={args.seed}"
    )

    with tempfile.TemporaryDirectory(prefix="engram_lme_") as td:
        db_dir = args.db_dir or Path(td)
        db_dir.mkdir(parents=True, exist_ok=True)
        for baseline in baselines:
            result = eval_baseline(
                baseline,
                conversations,
                top_k=args.top_k,
                db_dir=db_dir,
            )
            print(format_result(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
