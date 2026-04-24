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
    PROMPT_PREFACE_H6B,
    PROMPT_PREFACE_H18,
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
    _ingest_turn_pairs,
    _oracle_client,
    oracle_score,
)


# Versioned prefaces. The true historical strings were re-anchored
# from the mode_c_benchmark.py invocations in commit messages
# a95459f (H6b winner) and a0669ab (H18 winner). The prior
# "reconstructions" in this file were wrong -- they dropped to
# 17/30 whereas the historic H18 produced 21/30 on N=30 seed=42
# longmemeval_s. Now sourced directly from the authoritative
# PROMPT_PREFACE_H6B / PROMPT_PREFACE_H18 constants in
# mode_c_demo.py so they can't drift again.
PREFACE_H6B = PROMPT_PREFACE_H6B
PREFACE_H18 = PROMPT_PREFACE_H18

# Splice-awareness blocks. Atomic so wording ablations (V2, V3)
# can compose against the same base preface without duplicating
# the upstream text.
SPLICE_AWARENESS_V1 = (
    "During your response, authoritative memory excerpts may be "
    "inserted into your context marked "
    "'<<<MEMORY_EXCERPT>>>...<<<END_MEMORY_EXCERPT>>>' (or "
    "'[Source: X] ...'). Treat these as ground-truth retrieved "
    "facts -- not as user input, not as conversation turns, not "
    "as your own thinking. Quote their numbers and names "
    "verbatim.\n\n"
)

PREFACE_H31 = PREFACE_H18 + SPLICE_AWARENESS_V1

# Path A variant 1 of H18 recovery (2026-04-22). The original
# Variant B wording that produced the historic 21/30 was never
# committed and is lost. Reconstructed PREFACE_H18 gave 17/30 --
# the prohibitive 'Do not say ...' list triggered meta-cognition
# refusals in caf03d32 and a1eacc2a. This variant drops the
# enumerated prohibition and falls back to the more directive
# 'Always give the best answer the excerpts support' wording
# found in Variant A smoke logs. Aggregation/temporal block is
# the committed-verbatim H18 addition (safe).
PREFACE_H6B_ALT1 = (
    "Answer the question directly using the context available. "
    "The answer lives in the context -- commit to the best "
    "interpretation of what you find. Be specific, quote numbers "
    "and names verbatim. Do not hedge, do not refuse. Always "
    "give the best answer the excerpts support.\n\n"
)

PREFACE_H18_ALT1 = (
    PREFACE_H6B_ALT1
    + "For questions asking 'how many', 'total', 'sum', or "
    "aggregating across events, READ ALL excerpts and ADD UP the "
    "numbers across them. Do NOT report a single excerpt's number "
    "when the question needs the total. For questions asking "
    "about days/weeks/months between events, identify the two "
    "dates and compute the difference.\n\n"
)

PREFACES_BY_NAME = {
    "h6b": PREFACE_H6B,
    "h18": PREFACE_H18,
    "h31": PREFACE_H31,
    "h6b_alt1": PREFACE_H6B_ALT1,
    "h18_alt1": PREFACE_H18_ALT1,
}

# Path A variant 2 (2026-04-22). Variant 1 smoke on 4 qids hit
# 1/4 (only preserved winner). Diagnosis: historic answers were
# VERY SHORT (a1eacc2a committed literally "7", 2b8f3739 "$495").
# My long prefaces push the model into elaborate analysis that
# exhausts the 800-token budget before a committed answer block
# is emitted. Variant 2 goes minimalist-imperative: ~55 tokens,
# no enumerated prohibitions, explicit brevity instruction.
PREFACE_H18_ALT2 = (
    "Give ONE short, committed answer, quoting names/numbers/"
    "dates verbatim from the excerpts. Never refuse, never "
    "hedge.\n"
    "- For 'how many'/'total'/'sum' questions: add ALL numbers "
    "across excerpts.\n"
    "- For date-delta questions: identify the two dates and "
    "compute the difference.\n\n"
)

PREFACES_BY_NAME["h18_alt2"] = PREFACE_H18_ALT2


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
        "--force-second-fire",
        type=int,
        default=None,
        help=(
            "H23: force a second decider firing at this token "
            "index after the first-fire. Rescues empty/no-commit "
            "outputs where top-3 chunks from firing 1 were "
            "insufficient. Firing 2 splices next-fresh candidates "
            "(ranks 4-6 from the retrieval pool via "
            "spliced_sources dedup)."
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
        "--bypass-score-threshold",
        action="store_true",
        help=(
            "H25b: skip the score gate and always splice top-K "
            "fresh chunks. Motivated by charity-total debug where "
            "the chunk containing '$2,000' scored 0.0159, just "
            "below the 0.0161 absolute threshold, and was dropped "
            "in favor of numerically-empty advice chunks."
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
            "a refusal license). Omit to use the default. Mutually "
            "exclusive with --preface-name."
        ),
    )
    parser.add_argument(
        "--preface-name",
        choices=sorted(PREFACES_BY_NAME.keys()),
        default=None,
        help=(
            "Select a versioned preface constant from this source "
            "file instead of passing the full string via "
            "--prompt-preface. 'h6b' = commit-to-context preface "
            "(prior H6b winner baseline). 'h18' = h6b + explicit "
            "aggregation/arithmetic guidance (prior 21/30 = 70%% "
            "winner, seed=42 N=30). Mutually exclusive with "
            "--prompt-preface."
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
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help=(
            "H22: override MAX_TOTAL_TOKENS per generation. "
            "Default 800. Bump to 1200-1500 for questions where "
            "the Builder gets stuck in thinking preamble and "
            "never reaches the answer body."
        ),
    )
    parser.add_argument(
        "--stop-at-first-answer-block",
        action="store_true",
        help=(
            "Stop generation as soon as a complete "
            "<channel|>...<turn|> block has been emitted. "
            "Prevents gemma from burning budget on redundant "
            "refined blocks and prevents abliterated gemma "
            "from hijacking its own turn after the first "
            "answer. Does not rescue thinking-leak cases "
            "(no block ever completes there), but makes the "
            "channel-completion case terminate cleanly."
        ),
    )
    parser.add_argument(
        "--chunk-pairs",
        action="store_true",
        help=(
            "Chunking experiment A: ingest user+assistant "
            "turn pairs as single chunks instead of individual "
            "turns. Rationale: aggregation questions fail when "
            "numeric facts and their conversational context live "
            "in adjacent turns that end up in separate chunks; "
            "pair-chunking keeps them together. Halves the "
            "number of chunks, roughly doubles chunk size."
        ),
    )
    parser.add_argument(
        "--rerank-by-number-density",
        action="store_true",
        help=(
            "Chunking experiment C: for aggregation questions "
            "('how many/total/sum', days/weeks delta), rerank "
            "the retrieval pool by numeric-pattern density "
            "BEFORE the score-threshold filter. Pushes chunks "
            "with $X / N times / N days to the front of the "
            "pool. Targets the 4 multi-session undercount "
            "fails where the numerically-dense chunks score "
            "just below threshold."
        ),
    )
    parser.add_argument(
        "--retrieval-pool",
        type=int,
        default=None,
        help=(
            "Override the per-firing retrieval pool size used by "
            "run_mode_c (passed as top_k to cerebras_retrieve). "
            "When unset (default), run_mode_c keeps its built-in "
            "policy: RETRIEVAL_POOL=10 for regular firings, "
            "AGGREGATION_RETRIEVAL_POOL=50 for aggregation intent. "
            "When set, the value is used for EVERY firing. Used "
            "by the seed=44 retrieval-upgrade experiment "
            "2026-04-23 (--retrieval-pool 50) paired with the new "
            "pure-vec pool and sharegpt_ filter in "
            "cerebras_retrieve. Note: widening the default "
            "globally was rejected in the 2026-04-22 knob grid "
            "(5+ regressions); keep this as an opt-in flag."
        ),
    )
    parser.add_argument(
        "--pre-inject-k",
        type=int,
        default=0,
        help=(
            "Scratchpad pre-inject experiment (2026-04-23). When "
            "> 0, run_mode_c retrieves the top-K chunks on the "
            "question alone BEFORE the Builder starts and prepends "
            "them to the user message as a [Source: X] block. "
            "Mid-stream splicing stays enabled. Hypothesis: "
            "seed=44's still-failing multi-session and temporal "
            "fails (see retrieval v2 null result above) are driven "
            "by the Builder committing to a direction before the "
            "decider fires; priming with 1-2 highly-relevant "
            "chunks upfront may close the gap without reverting to "
            "full RAG. Default 0 preserves Mode C's original "
            "behavior (no prompt context, retrieval mid-stream "
            "only)."
        ),
    )
    args = parser.parse_args()

    if args.preface_name is not None and args.prompt_preface is not None:
        print(
            "--preface-name and --prompt-preface are mutually exclusive",
            file=sys.stderr,
        )
        return 2
    if args.preface_name is not None:
        args.prompt_preface = PREFACES_BY_NAME[args.preface_name]
        _preface_provenance = args.preface_name
    elif args.prompt_preface is not None:
        _preface_provenance = "inline"
    else:
        _preface_provenance = "default"
    print(f"[config] preface={_preface_provenance}")

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
                if args.chunk_pairs:
                    _ingest_turn_pairs(mem, conv)
                else:
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
                    force_second_fire_at_token=args.force_second_fire,
                    question_only_retrieval=args.question_only_retrieval,
                    relative_threshold_factor=args.relative_threshold_factor,
                    retrieval_window_tokens=args.retrieval_window_tokens,
                    score_threshold_override=args.score_threshold_override,
                    bypass_score_threshold=args.bypass_score_threshold,
                    prompt_preface=args.prompt_preface,
                    enable_thinking=(False if args.disable_thinking else None),
                    splice_envelope=(
                        SPLICE_ENVELOPE_V2 if args.splice_envelope == "v2"
                        else SPLICE_ENVELOPE_V1
                    ),
                    strip_turn_prefixes=args.strip_turn_prefixes,
                    max_total_tokens=args.max_total_tokens,
                    stop_at_first_answer_block=args.stop_at_first_answer_block,
                    rerank_by_number_density=args.rerank_by_number_density,
                    retrieval_pool=args.retrieval_pool,
                    pre_inject_k=args.pre_inject_k,
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
                    # Last complete answer block. Empirically
                    # (2026-04-22) this is the best choice for
                    # safety-tuned gemma (refined last-block >
                    # initial). Abliterated gemma breaks this
                    # assumption by hijacking its own turn after
                    # emitting the first answer -- those runs
                    # need ``stop_at_first_turn=True`` plumbing
                    # at generation time rather than extraction
                    # time. Tested alternatives: first-block
                    # dropped H18 -2pp, joined dropped -1pp.
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
