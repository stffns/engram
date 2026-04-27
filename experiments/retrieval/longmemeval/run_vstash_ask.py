"""Benchmark vstash.Memory.ask() on LongMemEval.

Tests the out-of-the-box vstash RAG endpoint -- retrieve top-k chunks +
generate answer via the configured inference backend -- against the
same LongMemEval substrate we've been evaluating pipeline_runner on.

Key differences from pipeline_runner:
- vstash.ask() uses vstash's own prompt template (not our
  PIPELINE_BUILDER_SYSTEM).
- vstash.ask() does its own retrieval internally (we control top_k).
- No brief synthesis; pure retrieval-then-generate.

Config: forces backend=cerebras + model=llama3.1-8b so we compare the
prompt/retrieval differences vs our pipeline_runner (same Builder
model, different wrapper). Override via VSTASH_BACKEND and
VSTASH_MODEL env vars for other backends.

Usage:
    python3 experiments/retrieval/longmemeval/run_vstash_ask.py \\
      --seed 44 --n 30 --top-k 10 --tag vstash_ask_cerebras
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ENGRAM = Path(__file__).resolve().parent.parent.parent.parent
if str(ENGRAM) not in sys.path:
    sys.path.insert(0, str(ENGRAM))

import os
import tempfile

# Force the backend BEFORE vstash loads its config.
os.environ.setdefault("VSTASH_BACKEND", "cerebras")
os.environ.setdefault("VSTASH_MODEL", "llama3.1-8b")

import vstash  # noqa: E402
import vstash.chat as _vstash_chat  # noqa: E402

from experiments.retrieval.longmemeval.dataset import load_longmemeval  # noqa: E402
from experiments.retrieval.longmemeval.runner import _format_turn  # noqa: E402
from experiments.retrieval.longmemeval.mode_a_eval import (  # noqa: E402
    _oracle_client,
    oracle_score,
)


# Hybrid prompt: vstash's "say so if no answer" (trust-first, preserves
# the +50% trust_score of vstash.ask default) + pipeline_runner's
# aggressive extraction rules (quote values, combine facts, compute).
# Hypothesis: correct_rate goes from 17/30 (default) toward 20/30 while
# trust stays >= +45%.
HYBRID_SYSTEM_PROMPT = """You are a precise document assistant. Answer questions based strictly on the provided context.

Rules (in priority order):
1. Quote specific numbers, names, dates, and titles VERBATIM from the context when they contain the answer.
2. If the answer requires combining facts across multiple context chunks (sums, counts, date ordering, ratios from numerator+denominator), do the combination and state the result with a brief justification.
3. If the context does NOT explicitly contain the answer, or answering would require guessing or inferring from absent evidence, respond EXACTLY: "not enough information in memory". Do not substitute a related or adjacent fact.
4. Prefer admitting insufficient information over stating a fact you cannot directly support from the context.
5. Always cite which source chunk each fact comes from (use the title shown in brackets).
6. Keep answers under 150 words."""


def _ingest(
    mem: vstash.Memory, conv, collection: str,
    *, granularity: str = "per-turn",
) -> int:
    """Ingest the conversation's haystack into vstash.

    `per-turn`    : one remember per turn (legacy default; pre-fragments
                    the dialogue and bypasses vstash's chunker).
    `per-session` : one remember per session (joined turns); vstash
                    chunks the block. Discovered 2026-04-25 to be
                    ~+13pp on LoCoMo correct rate.
    """
    n = 0
    if granularity == "per-turn":
        for sid, turns in conv.haystack_sessions.items():
            for i, turn in enumerate(turns):
                mem.remember(
                    _format_turn(turn),
                    title=f"{conv.question_id}::{sid}::{i}",
                    collection=collection,
                )
                n += 1
    elif granularity == "per-session":
        for sid, turns in conv.haystack_sessions.items():
            text = "\n".join(_format_turn(t) for t in turns)
            mem.remember(
                text,
                title=f"{conv.question_id}::{sid}",
                collection=collection,
            )
            n += 1
    else:
        raise ValueError(f"unknown granularity: {granularity}")
    return n


def _override_inference_config(mem: vstash.Memory, backend: str, model: str):
    """Swap the inference backend on this Memory's cfg without rewriting
    vstash.toml. vstash.Memory holds `_cfg` frozen at construction;
    rebuild it with the desired inference section.
    """
    from vstash.config import InferenceConfig
    old = mem._cfg
    new_inference = InferenceConfig(backend=backend, model=model)
    # Use pydantic's model_copy with deep=False to override just the
    # inference section while preserving every other field (forward
    # compat with future VstashConfig additions).
    new_cfg = old.model_copy(update={"inference": new_inference})
    object.__setattr__(mem, "_cfg", new_cfg)


def run_one(
    conv, score_fn, top_k: int, backend: str, model: str,
    *, granularity: str = "per-turn",
    vec_weight: float | None = None,
    fts_weight: float | None = None,
    retrieval_mode: str | None = None,
) -> dict:
    """``score_fn`` is a callable ``(question, ground_truth, candidate) -> dict``
    so the caller can swap Gemini (``oracle_score(client, ...)``) or
    Cerebras (``_CerebrasOracle().score(...)``) without changing this body.
    """
    t_q = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="vstash_ask_") as td:
        db_path = Path(td) / "mem.db"
        mem = vstash.Memory(db=str(db_path))
        _override_inference_config(mem, backend, model)
        collection = "default"

        t0 = time.perf_counter()
        n_ing = _ingest(mem, conv, collection, granularity=granularity)
        ingest_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        ask_kwargs = {"top_k": top_k, "collection": collection}
        if vec_weight is not None:
            ask_kwargs["vec_weight"] = vec_weight
        if fts_weight is not None:
            ask_kwargs["fts_weight"] = fts_weight
        if retrieval_mode is not None:
            ask_kwargs["retrieval_mode"] = retrieval_mode
        try:
            answer = mem.ask(conv.question, **ask_kwargs)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return {
                "qid": conv.question_id,
                "question": conv.question,
                "error": f"{type(exc).__name__}: {exc}",
                "wall_s": time.perf_counter() - t_q,
            }
        ask_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        gt = conv.answer if isinstance(conv.answer, str) else str(conv.answer)
        oracle = score_fn(conv.question, gt, answer)
        oracle_s = time.perf_counter() - t0

    return {
        "qid": conv.question_id,
        "question_type": conv.question_type,
        "question": conv.question,
        "ground_truth": gt,
        "n_ingested": n_ing,
        "vstash_answer": answer,
        "oracle": oracle,
        "wall_s_ingest": ingest_s,
        "wall_s_ask": ask_s,
        "wall_s_oracle": oracle_s,
        "wall_s_total": time.perf_counter() - t_q,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="longmemeval_s")
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--backend", default=os.environ.get("VSTASH_BACKEND", "cerebras"))
    ap.add_argument("--model", default=os.environ.get("VSTASH_MODEL", "llama3.1-8b"))
    ap.add_argument(
        "--oracle", choices=["gemini", "cerebras"], default="gemini",
        help="Oracle backend for grading. 'cerebras' uses llama3.1-8b "
             "as a fallback when Gemini quota is exhausted. The "
             "Cerebras-graded baseline at "
             "experiments/retrieval/longmemeval/pipeline_runs/"
             "vstash_ask_seed44_n30_vstash_ask_cerebras_*.jsonl is "
             "the apples-to-apples comparison point for --oracle cerebras runs.",
    )
    ap.add_argument("--tag", default="vstash_ask")
    ap.add_argument("--out", type=Path,
                    default=Path("experiments/retrieval/longmemeval/pipeline_runs"))
    ap.add_argument("--only-qids", default=None)
    ap.add_argument("--prompt-variant", choices=["default", "hybrid"],
                    default="default",
                    help="'default' uses vstash's built-in SYSTEM_PROMPT (trust-first). "
                         "'hybrid' combines vstash's honest-hedge rule with "
                         "pipeline_runner's extraction rules.")
    ap.add_argument(
        "--granularity",
        choices=["per-turn", "per-session"],
        default="per-turn",
        help="per-turn (legacy default, pre-fragments dialogue and "
             "bypasses vstash's chunker) or per-session (one remember "
             "per session, lets vstash chunk; +13pp on LoCoMo per "
             "2026-04-25 finding).",
    )
    ap.add_argument(
        "--vec-weight", type=float, default=None,
        help="vstash hybrid retrieval vec weight. Default ~0.86. "
             "vec=0.5 fts=0.5 unlocked +10pp temporal on LoCoMo (2026-04-25).",
    )
    ap.add_argument(
        "--fts-weight", type=float, default=None,
        help="vstash hybrid retrieval fts weight. Default ~0.14.",
    )
    ap.add_argument(
        "--retrieval-mode",
        choices=["hybrid", "vec_only", "fts_only"],
        default=None,
    )
    args = ap.parse_args()

    if args.prompt_variant == "hybrid":
        _vstash_chat.SYSTEM_PROMPT = HYBRID_SYSTEM_PROMPT
        print(f"[prompt] using HYBRID prompt ({len(HYBRID_SYSTEM_PROMPT)} chars)",
              flush=True)
    else:
        print(f"[prompt] using vstash DEFAULT SYSTEM_PROMPT", flush=True)

    convs = load_longmemeval(args.subset)
    if args.only_qids:
        wanted = {q.strip() for q in args.only_qids.split(",") if q.strip()}
        index = {c.question_id: c for c in convs}
        sampled = [index[q] for q in wanted if q in index]
    else:
        import random
        rnd = random.Random(args.seed)
        sampled = rnd.sample(convs, min(args.n, len(convs)))
    print(
        f"[vstash-ask] backend={args.backend} model={args.model} top_k={args.top_k} "
        f"seed={args.seed} n={len(sampled)}",
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out / f"vstash_ask_seed{args.seed}_n{len(sampled)}_{args.tag}_{ts}.jsonl"
    print(f"[out] {out_path}", flush=True)

    if args.oracle == "cerebras":
        from experiments.retrieval.locomo.runner_rerank import _CerebrasOracle
        oracle = _CerebrasOracle()
        print("[oracle] using Cerebras llama3.1-8b (Gemini quota fallback)",
              flush=True)
        _score_fn = lambda q, gt, ans: oracle.score(q, gt, ans)
    else:
        oracle = _oracle_client()
        print("[oracle] using Gemini (mode_a_eval default)", flush=True)
        _score_fn = lambda q, gt, ans: oracle_score(oracle, q, gt, ans)

    from experiments.retrieval.oracle_health import (
        OracleHealthError,
        OracleHealthGuard,
    )
    health = OracleHealthGuard()
    try:
        health.preflight(_score_fn)
        print(f"[oracle] pre-flight OK", flush=True)
    except OracleHealthError as exc:
        print(f"[oracle] PRE-FLIGHT FAILED: {exc}", flush=True)
        return 2

    t_all = time.perf_counter()
    correct = 0
    total = 0
    from collections import Counter
    verdicts = Counter()
    with out_path.open("w") as f:
        for i, conv in enumerate(sampled):
            print(f"\n[{i+1}/{len(sampled)}] qid={conv.question_id}", flush=True)
            row = run_one(
                conv, _score_fn, args.top_k, args.backend, args.model,
                granularity=args.granularity,
                vec_weight=args.vec_weight,
                fts_weight=args.fts_weight,
                retrieval_mode=args.retrieval_mode,
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            try:
                health.record(row.get("oracle"))
            except OracleHealthError as exc:
                print(f"\n[oracle] HEALTH ABORT: {exc}", flush=True)
                return 2
            v = (row.get("oracle") or {}).get("verdict")
            verdicts[v] += 1
            total += 1
            if v in ("supports", "partial"):
                correct += 1
            print(
                f"  -> verdict={v} wall={row.get('wall_s_total', 0):.1f}s "
                f"(ingest {row.get('wall_s_ingest', 0):.1f}s, "
                f"ask {row.get('wall_s_ask', 0):.1f}s, "
                f"oracle {row.get('wall_s_oracle', 0):.1f}s)",
                flush=True,
            )
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"  total wall   : {time.perf_counter() - t_all:.1f}s", flush=True)
    print(f"  verdicts     : {dict(verdicts)}", flush=True)
    if not health.summary_safe:
        print(f"\n  {health.warning()}\n", flush=True)
        print(
            f"  correct      : SUPPRESSED ({health.errors}/{health.total} oracle errors)",
            flush=True,
        )
        print(f"  trust_score  : SUPPRESSED", flush=True)
    else:
        print(f"  correct      : {correct}/{total} = {correct/max(1,total)*100:.1f}%", flush=True)
        print(f"  trust_score  : {(correct - verdicts['contradicts'])/max(1,total)*100:+.1f}%", flush=True)
    print(f"[out] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
