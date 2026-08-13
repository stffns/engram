"""Phase 2 distillation -- Step 1+2 probe.

Question this script answers: when gpt-oss-120b on Cerebras is asked
the canonical builder prompt over benchmark-shaped retrieval contexts
(LoCoMo per-session, LME per-turn), does ``message.reasoning``
populate, or is it empty in this domain?

If the channel populates with detailed traces, the larger Phase 2
distillation plan (Q -> reasoning_trace -> A SFT) is justified.
If the channel is empty or trivial, the plan reduces to direct Q -> A
distillation and the choice of teacher / student needs reconsidering.

Probe size: 5 LoCoMo per-session questions (conv 0 of locomo10.json)
+ 5 LME per-turn questions (first 5 conversations of longmemeval_s).
Each question retrieves top_k=8 with vec=0.5 / fts=0.5, builds the
canonical vstash builder prompt, and calls Cerebras directly with
max_tokens=4096 (reasoning headroom) and temperature=0.2 (vstash
default). We capture content, reasoning, and usage per call.

This script reaches into ``vstash.chat._build_messages`` so the
prompt shape is identical to production. CLAUDE.md hard rule allows
private-API access for one-off probes under ``experiments/`` /
``notes/``; no production code path imports this module.

Usage::

    CEREBRAS_API_KEY=... python -m experiments.phase2_distillation.probe_reasoning_channel

Outputs go to ``experiments/phase2_distillation/runs/probe_<ts>/``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import vstash
from vstash.chat import _build_messages  # probe-only: see module docstring

from experiments.retrieval.locomo.runner import load_locomo
from experiments.retrieval.locomo.runner_phase2 import _format_turn as _locomo_format_turn
from experiments.retrieval.longmemeval.dataset import load_longmemeval
from experiments.retrieval.longmemeval.runner import _format_turn as _lme_format_turn

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCOMO_DATA = REPO_ROOT / "experiments" / "retrieval" / "locomo" / "data" / "locomo10.json"
LME_CACHE = REPO_ROOT / "experiments" / "retrieval" / "longmemeval" / ".cache"

MODEL = "gpt-oss-120b"
TOP_K = 8
VEC_WEIGHT = 0.5
FTS_WEIGHT = 0.5
MAX_TOKENS = 4096
TEMPERATURE = 0.2

N_LOCOMO = 5
N_LME = 5


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _pick_locomo_questions(conv) -> list:
    """5 questions spread across categories present in conv 0."""
    by_cat: dict[int, list] = {}
    for qa in conv.qa_pairs:
        by_cat.setdefault(qa.category, []).append(qa)
    picks = []
    for cat in sorted(by_cat):
        if len(picks) >= N_LOCOMO:
            break
        picks.append(by_cat[cat][0])
    while len(picks) < N_LOCOMO:
        for qas in by_cat.values():
            for qa in qas:
                if qa not in picks:
                    picks.append(qa)
                    if len(picks) >= N_LOCOMO:
                        break
            if len(picks) >= N_LOCOMO:
                break
    return picks[:N_LOCOMO]


def _ingest_locomo_per_session(mem: vstash.Memory, conv) -> int:
    n = 0
    for session in conv.sessions:
        sid = f"session_{session.index}"
        lines = [_locomo_format_turn(t) for t in session.turns]
        text = f"[{session.date_time}]\n" + "\n".join(lines)
        mem.remember(text, title=f"{conv.sample_id}::{sid}")
        n += 1
    return n


def _ingest_lme_per_turn(mem: vstash.Memory, conv) -> int:
    n = 0
    for sid, turns in conv.haystack_sessions.items():
        for turn_idx, turn in enumerate(turns):
            text = _lme_format_turn(turn)
            if not text.strip():
                continue
            title = f"{conv.question_id}::{sid}::turn_{turn_idx}"
            mem.remember(text, title=title)
            n += 1
    return n


_SENTINEL = object()


def _call_cerebras(client, messages):
    """Direct Cerebras call. Captures content + reasoning + usage.

    Distinguishes "reasoning attribute absent from SDK response"
    from "attribute present but None / empty". This is the load-
    bearing signal of the probe -- a field-absent result means the
    SDK doesn't surface reasoning on this model; a field-present-but-
    empty result means the model chose not to populate it.
    """
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    wall = time.perf_counter() - t0
    msg = resp.choices[0].message
    content = getattr(msg, "content", None)
    raw_reasoning = getattr(msg, "reasoning", _SENTINEL)
    reasoning_present = raw_reasoning is not _SENTINEL
    reasoning = raw_reasoning if reasoning_present else None

    usage = getattr(resp, "usage", None)
    usage_dict = None
    if usage is not None:
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            usage_dict["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)

    return {
        "content": content,
        "reasoning": reasoning,
        "reasoning_present": reasoning_present,
        "usage": usage_dict,
        "wall_s": wall,
    }


def _run_probe(out_dir: Path) -> list[dict]:
    from cerebras.cloud.sdk import Cerebras
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise SystemExit("CEREBRAS_API_KEY required")
    client = Cerebras(api_key=api_key)

    rows: list[dict] = []
    rows_path = out_dir / "rows.jsonl"
    rows_fh = rows_path.open("w", encoding="utf-8")

    def _record(row: dict) -> None:
        rows.append(row)
        rows_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows_fh.flush()

    # ----- LoCoMo --------------------------------------------------- #
    print(f"[locomo] loading {LOCOMO_DATA}", flush=True)
    locomo_convs = load_locomo(LOCOMO_DATA)
    locomo_conv = locomo_convs[0]
    locomo_qs = _pick_locomo_questions(locomo_conv)
    print(f"[locomo] conv={locomo_conv.sample_id} picked {len(locomo_qs)} questions", flush=True)

    with tempfile.TemporaryDirectory(prefix="probe_locomo_") as tmpdir:
        db = Path(tmpdir) / "probe.db"
        mem = vstash.Memory(project="probe_locomo", db=db, collection="default")
        n_ing = _ingest_locomo_per_session(mem, locomo_conv)
        print(f"[locomo] ingested {n_ing} sessions", flush=True)

        for i, qa in enumerate(locomo_qs):
            print(f"[locomo {i+1}/{N_LOCOMO}] cat={qa.category_name} q={qa.question[:80]!r}", flush=True)
            chunks = mem.search(
                qa.question,
                top_k=TOP_K,
                vec_weight=VEC_WEIGHT,
                fts_weight=FTS_WEIGHT,
            )
            messages = _build_messages(qa.question, chunks, history=None)
            res = _call_cerebras(client, messages)
            _record({
                "source": "locomo",
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

    # ----- LME ------------------------------------------------------ #
    print(f"[lme] loading from cache {LME_CACHE}", flush=True)
    lme_convs = load_longmemeval(subset="longmemeval_s", cache_dir=LME_CACHE, download=False)
    lme_picks = lme_convs[:N_LME]
    print(f"[lme] picked {len(lme_picks)} conversations (one Q each)", flush=True)

    for i, conv in enumerate(lme_picks):
        with tempfile.TemporaryDirectory(prefix="probe_lme_") as tmpdir:
            db = Path(tmpdir) / "probe.db"
            mem = vstash.Memory(project="probe_lme", db=db, collection="default")
            n_ing = _ingest_lme_per_turn(mem, conv)
            print(f"[lme {i+1}/{N_LME}] qid={conv.question_id} type={conv.question_type} ingested {n_ing} turns", flush=True)
            chunks = mem.search(
                conv.question,
                top_k=TOP_K,
                vec_weight=VEC_WEIGHT,
                fts_weight=FTS_WEIGHT,
            )
            messages = _build_messages(conv.question, chunks, history=None)
            res = _call_cerebras(client, messages)
            _record({
                "source": "lme",
                "question_id": conv.question_id,
                "question_type": conv.question_type,
                "question": conv.question,
                "ground_truth": conv.answer,
                "n_chunks_retrieved": len(chunks),
                "content": res["content"],
                "reasoning": res["reasoning"],
                "reasoning_present": res["reasoning_present"],
                "usage": res["usage"],
                "wall_s": res["wall_s"],
            })
            mem.close()

    rows_fh.close()
    return rows


def _summarize(rows: list[dict]) -> dict:
    import statistics

    n = len(rows)
    n_attr_present = sum(1 for r in rows if r.get("reasoning_present"))
    n_nonempty = sum(1 for r in rows if (r.get("reasoning") or "").strip())
    reasoning_lens = [len(r.get("reasoning") or "") for r in rows]
    content_lens = [len(r.get("content") or "") for r in rows]
    reasoning_token_counts = [
        (r["usage"] or {}).get("reasoning_tokens")
        for r in rows
        if (r["usage"] or {}).get("reasoning_tokens") is not None
    ]
    summary = {
        "n_calls": n,
        "n_reasoning_attr_present": n_attr_present,
        "n_reasoning_nonempty_text": n_nonempty,
        "frac_reasoning_nonempty": n_nonempty / n if n else 0.0,
        "reasoning_chars": {
            "min": min(reasoning_lens) if reasoning_lens else 0,
            "median": statistics.median(reasoning_lens) if reasoning_lens else 0,
            "max": max(reasoning_lens) if reasoning_lens else 0,
        },
        "content_chars": {
            "min": min(content_lens) if content_lens else 0,
            "median": statistics.median(content_lens) if content_lens else 0,
            "max": max(content_lens) if content_lens else 0,
        },
        "reasoning_tokens_in_usage": {
            "n_reported": len(reasoning_token_counts),
            "values": reasoning_token_counts,
        },
        "by_source": {
            "locomo_nonempty": sum(1 for r in rows if r["source"] == "locomo" and (r.get("reasoning") or "").strip()),
            "lme_nonempty": sum(1 for r in rows if r["source"] == "lme" and (r.get("reasoning") or "").strip()),
        },
    }
    return summary


def main() -> int:
    stamp = _now_stamp()
    out_dir = REPO_ROOT / "experiments" / "phase2_distillation" / "runs" / f"probe_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probe] writing to {out_dir}", flush=True)

    rows = _run_probe(out_dir)
    rows_path = out_dir / "rows.jsonl"
    print(f"[probe] wrote {len(rows)} rows -> {rows_path}", flush=True)

    summary = _summarize(rows)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[probe] summary -> {summary_path}", flush=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
