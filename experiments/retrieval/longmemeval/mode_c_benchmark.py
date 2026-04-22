"""Mode C benchmark on LongMemEval -- head-to-head with RAG / Mode A.

The ``mode_a_eval.py`` grid runner handles Cerebras-backed
conditions (control / rag / rag_specific / mode_a). Mode C runs
on a local mlx model with streaming generation + KV-splice;
architecturally distinct enough that it gets its own runner.

This script reuses the LongMemEval question-and-haystack dataset
and the same Gemini 2.5 Flash oracle so results are comparable
to the N=50 runs already in ``mode_a_eval_grids/``.

Per question:

  1. Load the question's haystack into a fresh vstash db
     (matches ``mode_a_eval.py`` ingestion shape).
  2. Run ``mode_c_demo.run_mode_c`` end-to-end against that db.
  3. Oracle the answer with Gemini.
  4. Write one JSONL audit row capturing the question, ground
     truth, mode_c output, splice events, oracle verdict, and
     wall-time.

Default N=30 (subset of the seed=42 sample used for RAG
baseline) to keep wall time bounded. Each mode_c generation
takes ~15-20s on gemma-4-E2B-it local; 30 questions ~ 7-10
minutes plus ingestion. Oracle spend ~\\$0.01 per question.

Run:
  python experiments/retrieval/longmemeval/mode_c_benchmark.py \\
      --n 30 --seed 42 \\
      --out experiments/retrieval/longmemeval/mode_c_runs
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import vstash

from experiments.midloop_concept.medlocal.mode_c_demo import (
    SPLICE_ENVELOPE_V1,
    SPLICE_ENVELOPE_V2,
    load_model,
    run_mode_c,
)
from experiments.retrieval.longmemeval.dataset import (
    Conversation,
    load_longmemeval,
)
from experiments.retrieval.longmemeval.mode_a_eval import (
    _ingest,
    _oracle_client,
    oracle_score,
)


def _load_questions(subset: str, n: int, seed: int) -> list[Conversation]:
    conversations = load_longmemeval(subset=subset)
    rnd = random.Random(seed)
    return rnd.sample(conversations, min(n, len(conversations)))


def _load_questions_by_qid(subset: str, qids: list[str]) -> list[Conversation]:
    """Load a specific list of qids in the order given. Used by smoke
    tests that want to verify a fix moves target-fail qids without
    breaking a known-winner qid -- far cheaper than N=30."""
    index = {c.question_id: c for c in load_longmemeval(subset=subset)}
    missing = [q for q in qids if q not in index]
    if missing:
        raise ValueError(f"qid(s) not in {subset}: {missing}")
    return [index[q] for q in qids]


def _correct(v: str) -> bool:
    return v in ("supports", "partial")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="longmemeval_s")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/retrieval/longmemeval/mode_c_runs"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home()
        / ".lmstudio/models/lmstudio-community/gemma-4-E2B-it-MLX-4bit",
    )
    parser.add_argument(
        "--force-first-fire",
        type=int,
        default=None,
        help=(
            "H1: force the decider to fire at this token index "
            "regardless of claim patterns. Bypasses the late-firing "
            "issue where the heuristic detector only trips after "
            "the thinking preamble hardens into a committed answer."
        ),
    )
    parser.add_argument(
        "--question-only-retrieval",
        action="store_true",
        help=(
            "H2: search vstash with the question only (not "
            "question + window_text). Avoids query drift into "
            "the model's meta-reasoning."
        ),
    )
    parser.add_argument(
        "--relative-threshold-factor",
        type=float,
        default=None,
        help=(
            "H14: multi-chunk score cutoff becomes "
            "top1_score * FACTOR instead of the absolute 0.0161. "
            "Typical values 0.5-0.7. Adapts to query-noise regime."
        ),
    )
    parser.add_argument(
        "--retrieval-window-tokens",
        type=int,
        default=None,
        help=(
            "H15: retrieval query uses only the last N chars of "
            "the decider window (not the full 40-token window). "
            "Typical value: 80. Reduces noise inflation of "
            "irrelevant chunks."
        ),
    )
    parser.add_argument(
        "--score-threshold-override",
        type=float,
        default=None,
        help=(
            "H16: replace the default absolute 0.0161 threshold "
            "with a caller-supplied value. Used with "
            "--question-only-retrieval to lower the cutoff once "
            "window_text is no longer inflating scores."
        ),
    )
    parser.add_argument(
        "--tag",
        default=None,
        help=(
            "suffix appended to the output filename so multiple "
            "experiment conditions can coexist in the same output "
            "directory"
        ),
    )
    parser.add_argument(
        "--qids",
        default=None,
        help=(
            "comma-separated list of question_ids to run instead of "
            "the seed-sampled N. Smoke mode: verify a fix moves "
            "target-failing qids without breaking a known-good qid, "
            "before paying for an N=30 grid cell. Overrides --n and "
            "--seed."
        ),
    )
    parser.add_argument(
        "--prompt-preface",
        default=None,
        help=(
            "H6 variant prefix injected before the user question. "
            "Overrides the default PROMPT_PREFACE (which includes "
            "the 'not in memory' escape hatch the Builder takes as "
            "a refusal license). Omit to use the default."
        ),
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Qwen3+ family: pass enable_thinking=False to the chat "
            "template so the model emits an empty <think></think> "
            "pair and skips the thinking preamble. Saves the ~300 "
            "token budget that otherwise disappears into "
            "meta-reasoning before the answer body."
        ),
    )
    parser.add_argument(
        "--splice-envelope",
        choices=["v1", "v2"],
        default="v1",
        help=(
            "Envelope format for KV-splice payloads. v1 = original "
            "'[Source: X] {text}' (gemma-friendly but breaks on "
            "Qwen which hallucinates additional blocks). v2 = "
            "fenced '<<<MEMORY_EXCERPT>>>...<<<END_MEMORY_EXCERPT>>>' "
            "block that resists being read as a conversational turn."
        ),
    )
    parser.add_argument(
        "--strip-turn-prefixes",
        action="store_true",
        help=(
            "Strip leading 'user:' / 'assistant:' role markers "
            "from chunk text before splicing. LongMemEval stores "
            "transcripts with those prefixes literally and Qwen "
            "reads them as ChatML turn markers -- stripping them "
            "prevents the model from continuing a hallucinated "
            "dialogue."
        ),
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    if args.qids:
        # Smoke mode: pinned qids, seed is irrelevant; count reflects
        # the pinned list size, not the benchmark default N.
        pinned_n = len([q.strip() for q in args.qids.split(",") if q.strip()])
        out_path = args.out / f"mode_c_smoke_n{pinned_n}{tag}.jsonl"
    else:
        out_path = args.out / f"mode_c_n{args.n}_seed{args.seed}{tag}.jsonl"

    print(f"[config] subset={args.subset} n={args.n} seed={args.seed}")
    print(f"[config] out={out_path}")
    print(f"[config] model={args.model}")

    if args.qids:
        qid_list = [q.strip() for q in args.qids.split(",") if q.strip()]
        sampled = _load_questions_by_qid(args.subset, qid_list)
        print(f"[dataset] smoke mode: {len(sampled)} qid(s) pinned")
    else:
        sampled = _load_questions(args.subset, args.n, args.seed)
        print(f"[dataset] sampled {len(sampled)} conversations")

    # Load the mlx model ONCE up front and reuse across all N
    # questions -- each load is ~2-3s and doing N loads would
    # blow ~90s of pure wasted wall on N=30.
    model, tokenizer = load_model(args.model)

    oracle = _oracle_client()

    tmp_root = Path.home() / ".merken" / f"longmemeval_mode_c_{args.seed}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    fout = out_path.open("w")
    rows: list[dict] = []

    try:
        for i, conv in enumerate(sampled):
            gt_text = str(conv.answer) if conv.answer is not None else ""
            print(
                f"\n{'='*72}\n[{i+1}/{len(sampled)}] qid={conv.question_id} "
                f"type={conv.question_type}\n{'='*72}"
            )
            print(f"Q: {conv.question[:200]}")
            print(f"GT: {gt_text[:200]}")

            db_path = tmp_root / f"{conv.question_id}.db"
            if db_path.exists():
                db_path.unlink()
            mem = vstash.Memory(
                db=str(db_path),
                project=conv.question_id,
                collection="default",
            )

            try:
                t_ingest = time.perf_counter()
                _ingest(mem, conv)
                ingest_s = time.perf_counter() - t_ingest
                print(f"[ingest] {ingest_s:.1f}s")
            finally:
                mem.close()

            try:
                t_mc = time.perf_counter()
                mc_result = run_mode_c(
                    model_path=args.model,
                    db_path=db_path,
                    project=conv.question_id,
                    question=conv.question,
                    model=model,
                    tokenizer=tokenizer,
                    force_first_fire_at_token=args.force_first_fire,
                    question_only_retrieval=args.question_only_retrieval,
                    relative_threshold_factor=args.relative_threshold_factor,
                    retrieval_window_tokens=args.retrieval_window_tokens,
                    score_threshold_override=args.score_threshold_override,
                    prompt_preface=args.prompt_preface,
                    enable_thinking=(False if args.disable_thinking else None),
                    splice_envelope=(
                        SPLICE_ENVELOPE_V2 if args.splice_envelope == "v2"
                        else SPLICE_ENVELOPE_V1
                    ),
                    strip_turn_prefixes=args.strip_turn_prefixes,
                )
                mc_wall = time.perf_counter() - t_mc
                print(
                    f"[mode_c] {mc_wall:.1f}s  "
                    f"tokens={mc_result.total_tokens_sampled}  "
                    f"splices={len(mc_result.splices)}  "
                    f"termination={mc_result.terminated_reason}"
                )

                # Mode C answer_text can be 3000-4000 chars and
                # the model frequently emits MULTIPLE answer blocks
                # before the budget runs out:
                #
                #   <|channel>thought ... <channel|>ANSWER<turn|>
                #   <|channel>thought ... <channel|>ANSWER<turn|>
                #   <|channel>thought ... (CUT OFF by budget)
                #
                # The previous extraction used
                # ``rsplit("<channel|>", 1)[-1]`` which returned
                # whatever came after the LAST ``<channel|>`` --
                # when the last block was an incomplete thinking
                # preamble truncated by the budget, the oracle saw
                # only garbage even though correct answer blocks
                # existed earlier in the output. This was confirmed
                # on qid=3b6f954b (Melbourne) Variant A run:
                # the model emitted "University of Melbourne" twice
                # as completed answer blocks but the oracle was
                # fed a cut-off thinking block and verdicted
                # ``neutral``.
                #
                # Also: after an answer block the model sometimes
                # spams ``<turn|>`` repeatedly until budget runs
                # out. That spam crashes the oracle into
                # ``oracle_parse_failure`` even when the answer
                # itself is correct (confirmed on 4fd1909e /
                # Imagine Dragons Variant B run).
                #
                # Fix: find every complete ``<channel|>...<turn|>``
                # block in the output and return the LAST one
                # (latest committed answer). Strip trailing
                # ``<turn|>`` repetitions so the oracle doesn't
                # choke on transcript-end spam.
                import re as _re
                raw = mc_result.answer_text
                answer_blocks = _re.findall(
                    r"<channel\|>(.*?)<turn\|>", raw, flags=_re.DOTALL
                )
                if answer_blocks:
                    answer_for_oracle = answer_blocks[-1].strip()
                else:
                    # Fallback: no complete block, use prior logic.
                    answer_for_oracle = (
                        raw.rsplit("<channel|>", 1)[-1]
                        if "<channel|>" in raw else raw
                    )
                # Defensive: some outputs interleave multiple
                # ``<turn|>`` tokens inside the answer body;
                # collapse consecutive runs so the oracle candidate
                # is legible. Also strip Qwen ChatML markers
                # (<|im_end|>, <|im_start|>, <think>...</think>)
                # when a Qwen3+ Builder is in use -- the oracle
                # should never see transcript scaffolding.
                answer_for_oracle = _re.sub(
                    r"(<turn\|>)+", "", answer_for_oracle
                )
                answer_for_oracle = _re.sub(
                    r"<\|im_(start|end)\|>", "", answer_for_oracle
                )
                answer_for_oracle = _re.sub(
                    r"<think>.*?</think>", "", answer_for_oracle, flags=_re.DOTALL
                )
                # V2 envelope cleanup (shipped 2026-04-22, H11):
                # when the Builder (notably Qwen3+) regurgitates
                # the splice envelope as part of its own output,
                # the oracle was receiving up to 2000 chars of
                # chunk content instead of the model's actual
                # answer. Strip complete <<<MEMORY_EXCERPT ...
                # <<<END_MEMORY_EXCERPT>>> blocks AND orphaned
                # envelope markers (budget truncated the closing
                # tag).
                answer_for_oracle = _re.sub(
                    r"<<<MEMORY_EXCERPT[^>]*>>>.*?<<<END_MEMORY_EXCERPT>>>",
                    "", answer_for_oracle, flags=_re.DOTALL
                )
                answer_for_oracle = _re.sub(
                    r"<<<(MEMORY_EXCERPT[^>]*|END_MEMORY_EXCERPT)>>>",
                    "", answer_for_oracle
                )
                # V1 envelope cleanup: strip [Source: X] headers
                # and the chunk text that follows if the model
                # regurgitated them (same failure mode as V2 but
                # for the old envelope, in case older Builders
                # still produce such outputs).
                answer_for_oracle = _re.sub(
                    r"\[Source: [^\]]+\]\s*\n[^\[]*",
                    "", answer_for_oracle
                )
                answer_for_oracle = answer_for_oracle.strip()[-2000:]
                o = oracle_score(
                    oracle, conv.question, gt_text, answer_for_oracle
                )
                print(f"[oracle] verdict={o.get('verdict')}")

                row = {
                    "audit_id": uuid.uuid4().hex[:12],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "question_id": conv.question_id,
                    "question_type": conv.question_type,
                    "question": conv.question,
                    "ground_truth": gt_text,
                    "mode_c": {
                        "answer_text": mc_result.answer_text,
                        "total_tokens_sampled": mc_result.total_tokens_sampled,
                        "wall_s": mc_wall,
                        "terminated_reason": mc_result.terminated_reason,
                        "splices": [
                            {
                                "trigger_token_index": s.trigger_token_index,
                                "reason": s.reason,
                                "source_id": s.source_id,
                                "source_score": s.source_score,
                                "splice_tokens": s.splice_tokens,
                            }
                            for s in mc_result.splices
                        ],
                        "oracle": o,
                    },
                    "ingest_s": ingest_s,
                }
                fout.write(json.dumps(row, default=str) + "\n")
                fout.flush()
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                print(f"[error] qid={conv.question_id}: {exc!r}")
                traceback.print_exc()
                fout.write(json.dumps({
                    "question_id": conv.question_id,
                    "question": conv.question,
                    "error": repr(exc),
                }, default=str) + "\n")
                fout.flush()
            finally:
                if db_path.exists():
                    db_path.unlink()
    finally:
        fout.close()

    # Summary
    n_ok = sum(1 for r in rows if "mode_c" in r)
    n_correct = sum(
        1 for r in rows
        if "mode_c" in r
        and _correct(r["mode_c"]["oracle"]["verdict"])
    )
    total_splices = sum(
        len(r["mode_c"].get("splices", [])) for r in rows if "mode_c" in r
    )
    total_tokens = sum(
        r["mode_c"].get("total_tokens_sampled", 0) for r in rows if "mode_c" in r
    )
    total_wall = sum(r["mode_c"].get("wall_s", 0) for r in rows if "mode_c" in r)

    print("\n" + "=" * 72)
    print("MODE C BENCHMARK SUMMARY")
    print("=" * 72)
    if n_ok == 0:
        print("  no completed rows")
        return 1
    print(f"  n_completed:        {n_ok}/{len(sampled)}")
    print(
        f"  correct_rate:       {n_correct/n_ok*100:.1f}%  "
        f"({n_correct}/{n_ok})"
    )
    print(f"  avg tokens/q:       {total_tokens/n_ok:.0f}")
    print(f"  avg wall/q:         {total_wall/n_ok:.1f}s")
    print(f"  splices total:      {total_splices}")
    print(f"  splices/q:          {total_splices/n_ok:.2f}")
    print()
    print("Comparison anchors (from earlier LongMemEval runs on same seed 42):")
    print("  RAG-k3 temp=0.0:    70.0% correct / 1394 tok / 1.1s")
    print("  RAG-k5 temp=0.1-3:  72-74% correct / 2290 tok / 1.2s")
    print("  Mode A v4:          64-71% correct / 9048 tok / 5.3s")
    print(f"  Mode C (this run):  {n_correct/n_ok*100:.1f}% / "
          f"{total_tokens/n_ok:.0f} tok / {total_wall/n_ok:.1f}s")
    print(f"\nlog: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
