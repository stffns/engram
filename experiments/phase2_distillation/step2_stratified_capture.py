"""Phase 2 distillation -- Step 2 stratified capture.

Builds on the Step 1+2 lite probe
(``probe_reasoning_channel.py`` + writeup at
``notes/2026-04-30-phase2-distillation-probe-go.md``). Probe
confirmed reasoning channel populates 10/10 on a small sample, but
LME shape coverage was uneven -- the first 5 LME conversations all
happened to be ``single-session-user``. This script closes that gap
by sampling 5 calls per shape across LoCoMo (5 categories) and LME
(6 question_types), giving N=55 stratified.

Question being answered: does the reasoning channel populate
**uniformly** across shapes, or is the per-shape rate / quality
sensitive to question type? In particular:

- temporal-reasoning shapes -- does the trace do explicit date math?
- knowledge-update shapes (the documented teacher -11pp regression)
  -- does the trace flag the contradiction-then-refuse pattern?
- multi-session shapes -- does the trace synthesize across distant
  context windows?

Sampling design (deterministic seed=42):

- LoCoMo: 5 questions per category x 5 categories = 25 calls. All
  drawn from a single conversation (conv-26 has 5+ Q in each
  category, so one ingest covers all 25 LoCoMo calls).
- LME: 5 conversations per question_type x 6 types = 30 calls. Each
  LME conversation has one question, so 30 separate ingests are
  required. Stratified sample drawn from longmemeval_s.

Cost: ~55 paid Cerebras gpt-oss-120b calls, ~$1-2. Wallclock ~30-40
min (LME ingest dominates).

Usage::

    CEREBRAS_API_KEY=... python -m experiments.phase2_distillation.step2_stratified_capture

Outputs go to ``experiments/phase2_distillation/runs/step2_<ts>/``.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import vstash
from vstash.chat import _build_messages  # probe-only: see CLAUDE.md

from experiments.midloop_concept.medlocal.cerebras_midloop import (
    cerebras_chat_capture,
)
from experiments.retrieval.locomo.runner import load_locomo
from experiments.retrieval.locomo.runner_phase2 import (
    _format_turn as _locomo_format_turn,
)
from experiments.retrieval.longmemeval.dataset import load_longmemeval
from experiments.retrieval.longmemeval.runner import (
    _format_turn as _lme_format_turn,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCOMO_DATA = REPO_ROOT / "experiments" / "retrieval" / "locomo" / "data" / "locomo10.json"
LME_CACHE = REPO_ROOT / "experiments" / "retrieval" / "longmemeval" / ".cache"

MODEL = "gpt-oss-120b"
TOP_K = 8
VEC_WEIGHT = 0.5
FTS_WEIGHT = 0.5
MAX_TOKENS = 4096
TEMPERATURE = 0.2

LOCOMO_PER_CATEGORY = 5
LOCOMO_CATEGORIES = ["single_hop", "temporal", "multi_hop", "open_domain", "adversarial"]
LOCOMO_CONV_ID = "conv-26"  # picked because it has 5+ Q in every category

LME_PER_TYPE = 5
LME_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]

SEED = 42


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _stratified_locomo_picks(conv) -> list:
    """Deterministic 5-per-category sample from a single conversation.

    LoCoMo QA pairs have no stable id; sort by (category_name,
    question text) so the seeded shuffle is reproducible across
    Python versions and against any benchmark file refresh that
    might re-order ``qa_pairs``. Mirrors the determinism guarantee
    in ``_stratified_lme_picks``.
    """
    rng = random.Random(SEED)
    by_cat: dict[str, list] = {}
    for qa in sorted(conv.qa_pairs, key=lambda q: (q.category_name, q.question)):
        by_cat.setdefault(qa.category_name, []).append(qa)
    picks = []
    for cat in LOCOMO_CATEGORIES:
        pool = by_cat.get(cat, [])
        if len(pool) < LOCOMO_PER_CATEGORY:
            raise RuntimeError(
                f"conv {conv.sample_id} lacks {LOCOMO_PER_CATEGORY} Q in category "
                f"{cat!r}; only {len(pool)} found. Pick a different conv."
            )
        # rng.sample (not shuffle) avoids in-place mutation so this
        # helper is safe to copy-paste into contexts that reuse pools.
        picks.extend(rng.sample(pool, LOCOMO_PER_CATEGORY))
    return picks


def _stratified_lme_picks(convs) -> list:
    """Deterministic 5-per-question_type sample from longmemeval_s."""
    rng = random.Random(SEED)
    by_type: dict[str, list] = {}
    for c in convs:
        by_type.setdefault(c.question_type, []).append(c)
    picks = []
    for qt in LME_TYPES:
        pool = by_type.get(qt, [])
        if len(pool) < LME_PER_TYPE:
            raise RuntimeError(
                f"LME has only {len(pool)} conversations of type {qt!r}; "
                f"need {LME_PER_TYPE}."
            )
        # Sort first for determinism, then shuffle with the seeded RNG.
        pool = sorted(pool, key=lambda c: c.question_id)
        rng.shuffle(pool)
        picks.extend(pool[:LME_PER_TYPE])
    return picks


def _ingest_locomo_per_session(mem: vstash.Memory, conv) -> int:
    n = 0
    for session in conv.sessions:
        sid = f"session_{session.index}"
        lines = [_locomo_format_turn(t) for t in session.turns]
        text = f"[{session.date_time}]\n" + "\n".join(lines)
        mem.remember(text, title=f"{conv.sample_id}::{sid}")
        n += 1
    return n


def _ingest_lme_per_turn(mem: vstash.Memory, conv) -> tuple[int, int]:
    """Ingest one LME conversation. Returns (written, skipped).

    A single malformed turn (e.g. an unexpected tiktoken token that
    survived the special-token strip) must not abort the run -- 30
    LME ingests means 30 chances to hit one bad row, and a hard
    crash there would orphan a $1-2 paid run. Catch per-turn so
    the rest of the conversation still goes in. The skip count is
    surfaced in the row metadata so the writeup can flag any
    conversation with substantial drop.
    """
    n_written = 0
    n_skipped = 0
    for sid, turns in conv.haystack_sessions.items():
        for turn_idx, turn in enumerate(turns):
            text = _lme_format_turn(turn)
            if not text.strip():
                continue
            title = f"{conv.question_id}::{sid}::turn_{turn_idx}"
            try:
                mem.remember(text, title=title)
                n_written += 1
            except Exception as exc:  # noqa: BLE001 -- per-turn fail-soft
                print(
                    f"  [lme {conv.question_id} {title}] ingest skipped: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                n_skipped += 1
    return n_written, n_skipped


def _run(out_dir: Path) -> list[dict]:
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise SystemExit("CEREBRAS_API_KEY required")

    rows: list[dict] = []
    rows_path = out_dir / "rows.jsonl"
    rows_fh = rows_path.open("w", encoding="utf-8")

    def _record(row: dict) -> None:
        rows.append(row)
        rows_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows_fh.flush()

    # ----- LoCoMo (single ingest, 25 calls) ------------------------ #
    print(f"[locomo] loading {LOCOMO_DATA}", flush=True)
    locomo_convs = load_locomo(LOCOMO_DATA)
    locomo_conv = next(c for c in locomo_convs if c.sample_id == LOCOMO_CONV_ID)
    locomo_picks = _stratified_locomo_picks(locomo_conv)
    print(
        f"[locomo] conv={locomo_conv.sample_id} sessions={len(locomo_conv.sessions)} "
        f"picks={len(locomo_picks)} ({len(LOCOMO_CATEGORIES)} categories x "
        f"{LOCOMO_PER_CATEGORY})",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="step2_locomo_") as tmpdir:
        db = Path(tmpdir) / "step2.db"
        mem = vstash.Memory(project="step2_locomo", db=db, collection="default")
        n_ing = _ingest_locomo_per_session(mem, locomo_conv)
        print(f"[locomo] ingested {n_ing} sessions", flush=True)

        for i, qa in enumerate(locomo_picks):
            print(
                f"[locomo {i+1:2d}/{len(locomo_picks)}] cat={qa.category_name:12s} "
                f"q={qa.question[:80]!r}",
                flush=True,
            )
            chunks = mem.search(
                qa.question,
                top_k=TOP_K,
                vec_weight=VEC_WEIGHT,
                fts_weight=FTS_WEIGHT,
            )
            messages = _build_messages(qa.question, chunks, history=None)
            res = cerebras_chat_capture(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            _record({
                "source": "locomo",
                "sample_id": locomo_conv.sample_id,
                "category": qa.category,
                "category_name": qa.category_name,
                "question": qa.question,
                "ground_truth": qa.answer,
                "n_chunks_retrieved": len(chunks),
                "content": res["content"],
                "reasoning": res["reasoning"],
                "reasoning_present": res["reasoning_present"],
                "usage": res["usage"],
                "wall_s": res["wall_s"],
            })
        mem.close()

    # ----- LME (one ingest per call, 30 calls) --------------------- #
    print(f"[lme] loading from cache {LME_CACHE}", flush=True)
    lme_convs = load_longmemeval(subset="longmemeval_s", cache_dir=LME_CACHE, download=False)
    lme_picks = _stratified_lme_picks(lme_convs)
    print(
        f"[lme] picks={len(lme_picks)} ({len(LME_TYPES)} types x {LME_PER_TYPE})",
        flush=True,
    )

    for i, conv in enumerate(lme_picks):
        try:
            with tempfile.TemporaryDirectory(prefix="step2_lme_") as tmpdir:
                db = Path(tmpdir) / "step2.db"
                mem = vstash.Memory(project="step2_lme", db=db, collection="default")
                n_written, n_skipped = _ingest_lme_per_turn(mem, conv)
                print(
                    f"[lme {i+1:2d}/{len(lme_picks)}] qid={conv.question_id} "
                    f"type={conv.question_type:30s} ingested "
                    f"{n_written} turns ({n_skipped} skipped)",
                    flush=True,
                )
                chunks = mem.search(
                    conv.question,
                    top_k=TOP_K,
                    vec_weight=VEC_WEIGHT,
                    fts_weight=FTS_WEIGHT,
                )
                messages = _build_messages(conv.question, chunks, history=None)
                res = cerebras_chat_capture(
                    model=MODEL,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                _record({
                    "source": "lme",
                    "question_id": conv.question_id,
                    "question_type": conv.question_type,
                    "question": conv.question,
                    "ground_truth": conv.answer,
                    "n_chunks_retrieved": len(chunks),
                    "n_turns_ingested": n_written,
                    "n_turns_skipped": n_skipped,
                    "content": res["content"],
                    "reasoning": res["reasoning"],
                    "reasoning_present": res["reasoning_present"],
                    "usage": res["usage"],
                    "wall_s": res["wall_s"],
                })
                mem.close()
        except Exception as exc:  # noqa: BLE001 -- per-conv fail-soft
            print(
                f"[lme {i+1:2d}/{len(lme_picks)}] qid={conv.question_id} "
                f"FAILED: {type(exc).__name__}: {exc}",
                flush=True,
            )
            _record({
                "source": "lme",
                "question_id": conv.question_id,
                "question_type": conv.question_type,
                "question": conv.question,
                "ground_truth": conv.answer,
                "error": f"{type(exc).__name__}: {exc}",
            })

    rows_fh.close()
    return rows


def _summarize(rows: list[dict]) -> dict:
    # Note: the per-shape metric is `frac_reasoning_nonempty`, not
    # `frac_content_nonempty`. A burned-budget call (visible content
    # empty, hidden reasoning populated) correctly counts as
    # reasoning-present here. If a future caller wants to filter
    # those rows out, they should check `len(content)` directly on
    # rows.jsonl rather than change this metric.
    def _per_shape(rows: list[dict], key: str) -> dict:
        out: dict = {}
        groups: dict[str, list] = {}
        for r in rows:
            shape = r.get(key)
            if shape:
                groups.setdefault(shape, []).append(r)
        for shape, group in groups.items():
            n = len(group)
            n_with_text = sum(1 for r in group if (r.get("reasoning") or "").strip())
            r_lens = [len(r.get("reasoning") or "") for r in group]
            c_lens = [len(r.get("content") or "") for r in group]
            out[shape] = {
                "n": n,
                "n_reasoning_nonempty": n_with_text,
                "frac_reasoning_nonempty": n_with_text / n if n else 0.0,
                "reasoning_chars_median": statistics.median(r_lens) if r_lens else 0,
                "reasoning_chars_min": min(r_lens) if r_lens else 0,
                "reasoning_chars_max": max(r_lens) if r_lens else 0,
                "content_chars_median": statistics.median(c_lens) if c_lens else 0,
            }
        return out

    locomo_rows = [r for r in rows if r["source"] == "locomo"]
    lme_rows = [r for r in rows if r["source"] == "lme"]

    n = len(rows)
    n_with_text = sum(1 for r in rows if (r.get("reasoning") or "").strip())
    return {
        "n_calls": n,
        "n_reasoning_nonempty": n_with_text,
        "frac_reasoning_nonempty": n_with_text / n if n else 0.0,
        "by_locomo_category": _per_shape(locomo_rows, "category_name"),
        "by_lme_question_type": _per_shape(lme_rows, "question_type"),
        "totals_by_source": {
            "locomo": len(locomo_rows),
            "lme": len(lme_rows),
        },
    }


def main() -> int:
    stamp = _now_stamp()
    out_dir = REPO_ROOT / "experiments" / "phase2_distillation" / "runs" / f"step2_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[step2] writing to {out_dir}", flush=True)
    print(
        f"[step2] config: seed={SEED} top_k={TOP_K} max_tokens={MAX_TOKENS} "
        f"temp={TEMPERATURE} model={MODEL}",
        flush=True,
    )
    print(
        f"[step2] N total = {len(LOCOMO_CATEGORIES) * LOCOMO_PER_CATEGORY} "
        f"locomo + {len(LME_TYPES) * LME_PER_TYPE} lme = "
        f"{len(LOCOMO_CATEGORIES) * LOCOMO_PER_CATEGORY + len(LME_TYPES) * LME_PER_TYPE}",
        flush=True,
    )

    t0 = time.perf_counter()
    rows = _run(out_dir)
    wall = time.perf_counter() - t0
    print(f"[step2] {len(rows)} rows in {wall/60:.1f} min", flush=True)

    summary = _summarize(rows)
    summary["wall_minutes"] = wall / 60
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[step2] summary -> {summary_path}", flush=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
