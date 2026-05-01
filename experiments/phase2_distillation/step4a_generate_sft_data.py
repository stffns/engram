"""Phase 2 distillation -- Step 4a: SFT corpus generation.

Generates supervised fine-tuning examples for the local student
by running the teacher (gpt-oss-120b on Cerebras) over the
canonical Phase 2 retrieval pipeline and capturing both
``message.content`` (visible answer) and ``message.reasoning``
(hidden trace).

Each row carries everything the trainer needs to reconstruct the
prompt the student will see at inference time and the trace +
answer the student should learn to produce::

    {
      "qid": "...",
      "source": "locomo" | "lme",
      "question": "...",
      "ground_truth": "...",
      "category_name": "..." | null,    # locomo
      "question_type": "..." | null,    # lme
      "messages_input": [system, user],  # canonical builder prompt
      "teacher_reasoning": "...",
      "teacher_content": "...",
      "teacher_usage": {...},
      "wall_s": ...
    }

Sampling design (deterministic seed):

- LoCoMo: stratified across the 10 conversations and 5 categories.
  Default ``--n-locomo 200`` distributes proportionally to the
  per-conv category counts.
- LME: stratified by question_type. Default ``--n-lme 0`` (LoCoMo-
  only) because LME requires per-question haystack ingest (~50s
  each), making LME-heavy corpora prohibitively slow for a Step 4
  smoke. Bump explicitly when running overnight.

Resume-safe: rows.jsonl is appended to. On re-run, qids already
present are skipped. The CLI pins ``--out-dir`` so resume works
across invocations of the same corpus.

CLAUDE.md global rule: any script that generates paid-API calls in
bulk needs a code review before running. The cost cap below is the
operational guard; review is the correctness guard.

Usage::

    # Smoke (5 calls, ~$0.10, ~2 min)
    python -m experiments.phase2_distillation.step4a_generate_sft_data \\
        --smoke --out-dir experiments/phase2_distillation/sft_smoke

    # Production (1K examples, several hours, ~$10-30 depending on
    # Cerebras tier)
    python -m experiments.phase2_distillation.step4a_generate_sft_data \\
        --n-locomo 800 --n-lme 200 \\
        --out-dir experiments/phase2_distillation/sft_corpus_v1
"""

from __future__ import annotations

import argparse
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
from vstash.chat import _build_messages

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

# Hard cost cap. Halts before issuing a call if the projected cost
# (calls remaining x average tokens x rate) would exceed this. Set
# to a generous bound for the production run; tighter for smoke.
HARD_COST_CAP_USD = 50.0

# Cerebras gpt-oss-120b pricing (as of 2026-04-30, confirmed by
# Jay). Used only for the cost cap projection, not actual billing.
ROUGH_USD_PER_1M_INPUT = 0.35
ROUGH_USD_PER_1M_OUTPUT = 0.75


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _qid(row: dict) -> str:
    if row.get("source") == "lme":
        return f"lme:{row.get('question_id')}"
    return f"locomo:{row.get('sample_id')}::{(row.get('question') or '')[:80]}"


def _existing_qids(rows_path: Path) -> set[str]:
    if not rows_path.exists():
        return set()
    out: set[str] = set()
    with rows_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.add(_qid(row))
    return out


def _stratified_locomo_picks(
    convs: list, n_total: int, seed: int
) -> list[tuple]:
    """Distribute ``n_total`` picks across the 10 LoCoMo
    conversations and 5 categories. Each pick is a (conv, qa)
    tuple. Sort+seed pattern matches step2 for determinism.
    """
    if n_total <= 0:
        return []
    rng = random.Random(seed)
    # Build per (conv, cat) pools, sorted for determinism.
    pools: dict[tuple, list] = {}
    for conv in convs:
        for qa in sorted(conv.qa_pairs, key=lambda q: (q.category_name, q.question)):
            key = (conv.sample_id, qa.category_name)
            pools.setdefault(key, []).append((conv, qa))
    # Round-robin allocation across keys to keep balance until we
    # run out of picks. If a pool is exhausted it is skipped.
    keys = sorted(pools.keys())
    for k in keys:
        rng.shuffle(pools[k])
    picks: list[tuple] = []
    cursor = 0
    while len(picks) < n_total:
        moved = False
        for k in keys:
            if not pools[k]:
                continue
            picks.append(pools[k].pop(0))
            moved = True
            if len(picks) >= n_total:
                break
        if not moved:
            break
    return picks


def _stratified_lme_picks(
    convs: list, n_total: int, seed: int
) -> list:
    if n_total <= 0:
        return []
    rng = random.Random(seed)
    by_type: dict[str, list] = {}
    for c in convs:
        by_type.setdefault(c.question_type, []).append(c)
    keys = sorted(by_type.keys())
    for k in keys:
        by_type[k] = sorted(by_type[k], key=lambda c: c.question_id)
        rng.shuffle(by_type[k])
    picks: list = []
    while len(picks) < n_total:
        moved = False
        for k in keys:
            if not by_type[k]:
                continue
            picks.append(by_type[k].pop(0))
            moved = True
            if len(picks) >= n_total:
                break
        if not moved:
            break
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
    n_w = 0
    n_s = 0
    for sid, turns in conv.haystack_sessions.items():
        for turn_idx, turn in enumerate(turns):
            text = _lme_format_turn(turn)
            if not text.strip():
                continue
            title = f"{conv.question_id}::{sid}::turn_{turn_idx}"
            try:
                mem.remember(text, title=title)
                n_w += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  ingest skipped: {type(exc).__name__}: {exc}", flush=True)
                n_s += 1
    return n_w, n_s


def _estimate_cost_usd(rows_so_far: list[dict], remaining: int) -> float:
    """Project remaining cost from average usage of rows already
    captured. Returns 0 if no rows yet (caller should treat that
    as 'unknown -- proceed but watch')."""
    if not rows_so_far or remaining <= 0:
        return 0.0
    in_toks = [
        (r.get("teacher_usage") or {}).get("prompt_tokens", 0) or 0
        for r in rows_so_far
    ]
    out_toks = [
        (r.get("teacher_usage") or {}).get("completion_tokens", 0) or 0
        for r in rows_so_far
    ]
    if not in_toks:
        return 0.0
    avg_in = statistics.mean(in_toks)
    avg_out = statistics.mean(out_toks)
    proj = remaining * (
        avg_in * ROUGH_USD_PER_1M_INPUT / 1_000_000
        + avg_out * ROUGH_USD_PER_1M_OUTPUT / 1_000_000
    )
    return proj


class _CostCapExceeded(Exception):
    """Raised by the inner processors to halt the whole run when
    the projected cost exceeds the cap. Caught by ``main()`` so we
    do not silently continue into the LME phase after the cap is
    hit during LoCoMo (per PR #49 review)."""


def _is_auth_error(exc: BaseException) -> bool:
    """Detect Cerebras auth / permissions failures so they halt the
    run instead of being silently retried qid-by-qid (would burn the
    full pick list with N "call failed" prints and zero rows). The
    helper retries 5xx/429 internally; 4xx propagate.
    """
    msg = str(exc).lower()
    if any(s in msg for s in ("401", "403", "unauthor", "forbidden", "invalid api key", "authentication")):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int) and status in (401, 403):
        return True
    return False


def _process_locomo(
    picks: list[tuple],
    fh,
    rows: list[dict],
    seen_qids: set[str],
    cost_cap_usd: float,
    total_target: int,
) -> None:
    """Group picks by sample_id so each conversation is ingested
    once. Within a conv, fire all picks. Skip qids already in
    rows.jsonl.

    ``total_target`` is the global pick count (locomo + lme) so the
    cost projection covers the whole job, not just the tail of the
    current conv.
    """
    by_conv: dict[str, list] = {}
    conv_obj_by_id: dict[str, object] = {}
    for conv, qa in picks:
        conv_obj_by_id[conv.sample_id] = conv
        by_conv.setdefault(conv.sample_id, []).append(qa)

    for conv_id, qas in by_conv.items():
        # Filter out qids already done before paying for ingest.
        pending = []
        for qa in qas:
            qid = f"locomo:{conv_id}::{(qa.question or '')[:80]}"
            if qid not in seen_qids:
                pending.append(qa)
        if not pending:
            print(f"[locomo {conv_id}] all {len(qas)} qids already done, skipping",
                  flush=True)
            continue
        conv = conv_obj_by_id[conv_id]
        with tempfile.TemporaryDirectory(prefix="step4_locomo_") as tmpdir:
            db = Path(tmpdir) / "step4.db"
            mem = vstash.Memory(project="step4_locomo", db=db, collection="default")
            n_ing = _ingest_locomo_per_session(mem, conv)
            print(
                f"[locomo {conv_id}] ingested {n_ing} sessions, "
                f"{len(pending)}/{len(qas)} qids pending",
                flush=True,
            )

            for i, qa in enumerate(pending):
                # Cost projection guard against the GLOBAL remaining
                # count, not the per-conv remainder. Otherwise the
                # guard only fires once you have already paid for all
                # earlier conversations.
                global_remaining = max(0, total_target - len(rows))
                proj = _estimate_cost_usd(rows, global_remaining)
                if proj > cost_cap_usd:
                    print(
                        f"[locomo {conv_id}] cost cap reached: "
                        f"projected ${proj:.2f} > cap ${cost_cap_usd:.2f} "
                        f"(global remaining {global_remaining}); halting "
                        "the whole run (LME phase will NOT run).",
                        flush=True,
                    )
                    mem.close()
                    raise _CostCapExceeded(proj)
                qid = f"locomo:{conv_id}::{(qa.question or '')[:80]}"
                print(
                    f"[locomo {conv_id} {i+1}/{len(pending)}] cat={qa.category_name:12s} "
                    f"q={qa.question[:70]!r}",
                    flush=True,
                )
                chunks = mem.search(
                    qa.question,
                    top_k=TOP_K,
                    vec_weight=VEC_WEIGHT,
                    fts_weight=FTS_WEIGHT,
                )
                messages = _build_messages(qa.question, chunks, history=None)
                try:
                    res = cerebras_chat_capture(
                        model=MODEL,
                        messages=messages,
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                    )
                except Exception as exc:  # noqa: BLE001
                    if _is_auth_error(exc):
                        print(
                            f"  call failed (auth): {type(exc).__name__}: {exc}; "
                            "halting -- bad key would burn the whole pick list",
                            flush=True,
                        )
                        mem.close()
                        raise
                    print(
                        f"  call failed: {type(exc).__name__}: {exc}; "
                        "skipping qid",
                        flush=True,
                    )
                    continue
                row = {
                    "qid": qid,
                    "source": "locomo",
                    "sample_id": conv_id,
                    "category": qa.category,
                    "category_name": qa.category_name,
                    "question": qa.question,
                    "ground_truth": qa.answer,
                    "n_chunks_retrieved": len(chunks),
                    "messages_input": messages,
                    "teacher_content": res["content"],
                    "teacher_reasoning": res["reasoning"],
                    "teacher_usage": res["usage"],
                    "wall_s": res["wall_s"],
                }
                rows.append(row)
                seen_qids.add(qid)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            mem.close()


def _process_lme(
    picks: list,
    fh,
    rows: list[dict],
    seen_qids: set[str],
    cost_cap_usd: float,
    total_target: int,
) -> None:
    for i, conv in enumerate(picks):
        qid = f"lme:{conv.question_id}"
        if qid in seen_qids:
            print(f"[lme {i+1}/{len(picks)}] qid={conv.question_id} already done, skipping",
                  flush=True)
            continue
        global_remaining = max(0, total_target - len(rows))
        proj = _estimate_cost_usd(rows, global_remaining)
        if proj > cost_cap_usd:
            print(
                f"[lme] cost cap reached: projected ${proj:.2f} > cap "
                f"${cost_cap_usd:.2f} (global remaining {global_remaining}); halting.",
                flush=True,
            )
            raise _CostCapExceeded(proj)
        try:
            with tempfile.TemporaryDirectory(prefix="step4_lme_") as tmpdir:
                db = Path(tmpdir) / "step4.db"
                mem = vstash.Memory(project="step4_lme", db=db, collection="default")
                n_w, n_s = _ingest_lme_per_turn(mem, conv)
                print(
                    f"[lme {i+1:3d}/{len(picks)}] qid={conv.question_id} "
                    f"type={conv.question_type:30s} ingested {n_w} turns "
                    f"({n_s} skipped)",
                    flush=True,
                )
                chunks = mem.search(
                    conv.question,
                    top_k=TOP_K,
                    vec_weight=VEC_WEIGHT,
                    fts_weight=FTS_WEIGHT,
                )
                messages = _build_messages(conv.question, chunks, history=None)
                try:
                    res = cerebras_chat_capture(
                        model=MODEL,
                        messages=messages,
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                    )
                except Exception as exc:  # noqa: BLE001
                    if _is_auth_error(exc):
                        print(
                            f"  call failed (auth): {type(exc).__name__}: {exc}; "
                            "halting -- bad key would burn the whole pick list",
                            flush=True,
                        )
                        mem.close()
                        raise
                    raise
                row = {
                    "qid": qid,
                    "source": "lme",
                    "question_id": conv.question_id,
                    "question_type": conv.question_type,
                    "question": conv.question,
                    "ground_truth": conv.answer,
                    "n_chunks_retrieved": len(chunks),
                    "n_turns_ingested": n_w,
                    "n_turns_skipped": n_s,
                    "messages_input": messages,
                    "teacher_content": res["content"],
                    "teacher_reasoning": res["reasoning"],
                    "teacher_usage": res["usage"],
                    "wall_s": res["wall_s"],
                }
                rows.append(row)
                seen_qids.add(qid)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                mem.close()
        except _CostCapExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            if _is_auth_error(exc):
                # Don't swallow auth errors at the outer level --
                # the inner re-raise was deliberate. Re-propagate
                # so main() halts.
                raise
            print(
                f"[lme {i+1:3d}/{len(picks)}] qid={conv.question_id} "
                f"FAILED: {type(exc).__name__}: {exc}",
                flush=True,
            )


def _summarize(rows: list[dict]) -> dict:
    by_source = {"locomo": 0, "lme": 0}
    n_with_reasoning = 0
    n_truncated = 0
    n_empty_content = 0
    in_toks = []
    out_toks = []
    for r in rows:
        src = r.get("source")
        by_source[src] = by_source.get(src, 0) + 1
        if (r.get("teacher_reasoning") or "").strip():
            n_with_reasoning += 1
        if not (r.get("teacher_content") or "").strip():
            n_empty_content += 1
        u = r.get("teacher_usage") or {}
        in_toks.append(u.get("prompt_tokens", 0) or 0)
        out_toks.append(u.get("completion_tokens", 0) or 0)
    avg_in = statistics.mean(in_toks) if in_toks else 0
    avg_out = statistics.mean(out_toks) if out_toks else 0
    cost = (
        sum(in_toks) * ROUGH_USD_PER_1M_INPUT / 1_000_000
        + sum(out_toks) * ROUGH_USD_PER_1M_OUTPUT / 1_000_000
    )
    return {
        "n_rows": len(rows),
        "by_source": by_source,
        "n_with_reasoning": n_with_reasoning,
        "n_empty_content": n_empty_content,
        "avg_prompt_tokens": int(avg_in),
        "avg_completion_tokens": int(avg_out),
        "estimated_cost_usd": round(cost, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-locomo", type=int, default=200)
    ap.add_argument("--n-lme", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="N=5 LoCoMo, N=0 LME, capped budget $1.")
    ap.add_argument("--out-dir", type=Path, required=False,
                    help="Directory for rows.jsonl. Required outside --smoke.")
    ap.add_argument("--cost-cap-usd", type=float, default=HARD_COST_CAP_USD)
    args = ap.parse_args()

    if args.smoke:
        args.n_locomo = 5
        args.n_lme = 0
        args.cost_cap_usd = 1.0
        if args.out_dir is None:
            args.out_dir = REPO_ROOT / "experiments" / "phase2_distillation" / "sft_smoke"
    if args.out_dir is None:
        sys.exit("--out-dir required (or use --smoke)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "rows.jsonl"
    print(f"[step4a] out_dir={args.out_dir}", flush=True)
    print(
        f"[step4a] config: n_locomo={args.n_locomo} n_lme={args.n_lme} "
        f"seed={args.seed} cost_cap=${args.cost_cap_usd:.2f}",
        flush=True,
    )

    if not os.environ.get("CEREBRAS_API_KEY"):
        sys.exit("CEREBRAS_API_KEY required")

    seen = _existing_qids(rows_path)
    if seen:
        print(f"[step4a] resuming -- {len(seen)} qids already in rows.jsonl", flush=True)

    print("[step4a] loading data", flush=True)
    locomo_convs = load_locomo(LOCOMO_DATA)
    locomo_picks = _stratified_locomo_picks(locomo_convs, args.n_locomo, args.seed)
    print(f"[step4a] locomo picks: {len(locomo_picks)}", flush=True)

    lme_picks = []
    if args.n_lme > 0:
        lme_convs = load_longmemeval(
            subset="longmemeval_s", cache_dir=LME_CACHE, download=False
        )
        lme_picks = _stratified_lme_picks(lme_convs, args.n_lme, args.seed)
        print(f"[step4a] lme picks: {len(lme_picks)}", flush=True)

    rows: list[dict] = []
    # Re-load existing rows so cost projection from this session
    # plus prior rows is honest.
    if rows_path.exists():
        with rows_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    t0 = time.perf_counter()
    # total_target is the global pick count for cost projection.
    # We DON'T add len(seen) -- on resume the picks list still
    # contains qids already in seen (they get filtered inside the
    # processors), so the budget should be the total target N, not
    # N + already-done.
    total_target = len(locomo_picks) + len(lme_picks)
    cost_cap_hit = False
    with rows_path.open("a", encoding="utf-8") as fh:
        try:
            if locomo_picks:
                _process_locomo(locomo_picks, fh, rows, seen, args.cost_cap_usd, total_target)
            if lme_picks:
                _process_lme(lme_picks, fh, rows, seen, args.cost_cap_usd, total_target)
        except _CostCapExceeded:
            cost_cap_hit = True
            print("[step4a] halted by cost cap; partial rows.jsonl preserved.", flush=True)
    wall = time.perf_counter() - t0
    print(f"[step4a] {len(rows)} rows total, +wall={wall/60:.1f}min", flush=True)

    summary = _summarize(rows)
    summary["wall_minutes_this_session"] = wall / 60
    summary["cost_cap_hit"] = cost_cap_hit
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 1 if cost_cap_hit else 0


if __name__ == "__main__":
    sys.exit(main())
