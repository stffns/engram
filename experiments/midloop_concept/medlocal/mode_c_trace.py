"""Mode C per-stage debug trace.

Picks one LongMemEval question where the winning config (E4B +
multi-chunk splice, a.k.a. H3+H12) oracle'd as ``supports``, rebuilds
the haystack into a fresh vstash db, and walks the Mode C pipeline
step-by-step printing what each stage sees and produces. Intended as
the "show me what actually happens at each stage" artifact for the
Mode C writeup.

Default question: the Imagine Dragons single-session-user lookup
(``qid=4fd1909e``) -- short, single-firing, clean winning case.

Stages printed:

  [1] Chat-template applied to the user question
  [2] Generation streams tokens; decider fires at t=T
  [3] vstash 3-way-dual retrieval returns a top-5 pool
  [4] Multi-chunk policy selects qualifying chunks
  [5] Splice is applied; generation continues
  [6] Final answer + oracle verdict

Run:
  python experiments/midloop_concept/medlocal/mode_c_trace.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import vstash

from experiments.midloop_concept.medlocal.cerebras_midloop import (
    retrieve as cerebras_retrieve,
)
from experiments.midloop_concept.medlocal.mode_c_demo import (
    MAX_TOTAL_TOKENS,
    MULTI_SPLICE_BUDGET_TOKENS,
    MULTI_SPLICE_MAX_CHUNKS,
    MULTI_SPLICE_SCORE_THRESHOLD,
    PROMPT_PREFACE,
    RETRIEVAL_POOL,
    SPLICE_ENVELOPE,
    _apply_chat,
    _encode,
    load_model,
    run_mode_c,
)
from experiments.retrieval.longmemeval.dataset import load_longmemeval
from experiments.retrieval.longmemeval.mode_a_eval import (
    _ingest,
    _oracle_client,
    oracle_score,
)


DEFAULT_QID = "4fd1909e"
DEFAULT_MODEL = Path.home() / ".lmstudio/models/lmstudio-community/gemma-4-E4B-it-MLX-4bit"


def _divider(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", default=DEFAULT_QID)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--subset", default="longmemeval_s")
    args = parser.parse_args()

    convs = load_longmemeval(subset=args.subset)
    conv = next((c for c in convs if c.question_id == args.qid), None)
    if conv is None:
        print(f"qid not found: {args.qid}")
        return 2

    _divider("[0] QUESTION")
    print(f"  qid:  {conv.question_id}")
    print(f"  type: {conv.question_type}")
    print(f"  Q:    {conv.question}")
    print(f"  GT:   {conv.answer}")

    model, tokenizer = load_model(args.model)

    _divider("[1] CHAT-TEMPLATE APPLIED TO QUESTION")
    chatted = _apply_chat(tokenizer, conv.question)
    print(f"  full prompt length: {len(chatted)} chars")
    print("  preamble (first 240 chars):")
    print(f"    {chatted[:240]!r}")
    print("  tail with question (last 240 chars):")
    print(f"    {chatted[-240:]!r}")

    tmp_db = Path.home() / ".merken" / f"mode_c_trace_{conv.question_id}.db"
    if tmp_db.exists():
        tmp_db.unlink()

    _divider("[2] INGEST HAYSTACK INTO VSTASH")
    mem = vstash.Memory(db=str(tmp_db), project=conv.question_id, collection="default")
    try:
        t0 = time.perf_counter()
        _ingest(mem, conv)
        print(f"  ingest wall: {time.perf_counter()-t0:.1f}s")
    finally:
        mem.close()

    _divider("[3] RUN MODE C (H3+H12 = E4B + multi-chunk)")
    t0 = time.perf_counter()
    result = run_mode_c(
        model_path=args.model,
        db_path=tmp_db,
        project=conv.question_id,
        question=conv.question,
        model=model,
        tokenizer=tokenizer,
        # H3+H12 config: no forced-first-fire, no question-only.
    )
    mc_wall = time.perf_counter() - t0
    print(f"  [stream] done in {mc_wall:.1f}s  "
          f"tokens={result.total_tokens_sampled}  "
          f"splices={len(result.splices)}  "
          f"terminated={result.terminated_reason}")

    _divider("[4] SPLICE EVENTS (what the decider fired on)")
    if not result.splices:
        print("  (no splices this run)")
    for i, s in enumerate(result.splices):
        print(
            f"  [{i}] trigger_t={s.trigger_token_index} "
            f"reason={s.reason}  "
            f"source={s.source_id}  score="
            f"{s.source_score:.4f}  "
            f"splice_tokens={s.splice_tokens}"
        )
        print(f"       window tail: {s.window_text[-80:]!r}")
    # Group splices by firing to see what the multi-chunk policy
    # actually packed together. Splices sharing the same
    # trigger_token_index + reason came from a single firing.
    by_fire: dict[tuple[int, str], list] = {}
    for s in result.splices:
        key = (s.trigger_token_index, s.reason)
        by_fire.setdefault(key, []).append(s)
    _divider("[5] MULTI-CHUNK GROUPING (H12 policy)")
    for key, group in by_fire.items():
        print(f"  firing at t={key[0]} ({key[1]}):")
        for s in group:
            print(
                f"    src={s.source_id}  score={s.source_score:.4f}  "
                f"tok={s.splice_tokens}"
            )
        total_tok = sum(s.splice_tokens for s in group)
        print(f"    -> {len(group)} chunks, {total_tok} splice tokens combined")

    _divider("[6] RETRIEVAL POOL AT FIRST FIRING (what vstash returned)")
    # Re-run the retrieval call with the first firing's window to
    # show the full top-5 and which rows the policy accepted /
    # rejected. This is the "what did retrieval see" artifact.
    if result.splices:
        first_window = result.splices[0].window_text
        mem2 = vstash.Memory(db=str(tmp_db), project=conv.question_id, collection="default")
        try:
            ret_query = f"{conv.question}\n{first_window}"
            pool = cerebras_retrieve(
                mem2, ret_query, top_k=RETRIEVAL_POOL, retrieval_mode="dual"
            )
            for i, e in enumerate(pool):
                score = e.get("score")
                passes = (
                    isinstance(score, (int, float))
                    and score >= MULTI_SPLICE_SCORE_THRESHOLD
                )
                tag = "KEEP" if passes else "drop"
                src = e.get("source_id", "?")
                text = (e.get("text", "") or "")[:120].replace("\n", " ")
                print(
                    f"  rank={i+1} [{tag}] score="
                    f"{score if isinstance(score,(int,float)) else '?':.4f}  "
                    f"src={src}"
                )
                print(f"           text: {text!r}")
        finally:
            mem2.close()
    else:
        print("  (no splices; no retrieval pool to inspect)")

    _divider("[7] FINAL ANSWER TEXT (what the model output)")
    raw = result.answer_text
    print(f"  raw length: {len(raw)} chars")
    # Strip thinking preamble + tail-truncate like the benchmark.
    stripped = raw.rsplit("<channel|>", 1)[-1] if "<channel|>" in raw else raw
    stripped = stripped[-2000:]
    print("  stripped (oracle view, last 500 chars):")
    for line in stripped[-500:].splitlines():
        print(f"    | {line}")

    _divider("[8] ORACLE VERDICT")
    oracle = _oracle_client()
    o = oracle_score(oracle, conv.question, str(conv.answer), stripped)
    print(f"  verdict: {o.get('verdict')}")
    print(f"  rationale (truncated): {(o.get('rationale') or '')[:300]}")

    # Cleanup.
    if tmp_db.exists():
        tmp_db.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
