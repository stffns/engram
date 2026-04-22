"""Mode C end-to-end demo -- milestone 4 of the Mode C plan.

Wires every piece the earlier milestones built into a single
runnable script:

- gemma-4-E2B-it-MLX-4bit (proven viable in phase0b/c/d).
- vstash.Memory as the retrieval substrate.
- StreamingDecider (``streaming_claim_detector.py``) as the
  continuous-cadence gate.
- KV-splice mechanism (proven in phase0c/d) as the injection
  path. Same one-call ``generate_step`` pattern, no token
  duplication.

Execution flow per question:

  1. Apply Gemma chat template. Create empty KV cache.
  2. Start a streaming generate_step with the chatted prompt.
  3. For each sampled token:
     a. Append to the running output.
     b. Feed the decoded token text to StreamingDecider.
     c. If the decider returns ``fire=True``, break out of the
        streamer.
  4. On break:
     a. Call vstash.search with the decider's window text
        (implicitly captures what the Builder just said).
     b. Take the top chunk, wrap with a light ``[Source: X]``
        header.
     c. Re-enter generate_step with the splice as the new
        "prompt" (prefills the splice K/V into the cache, then
        continues sampling from the enriched cache).
     d. Log ``[retrieving: source_id | N tokens]`` to stdout.
  5. Continue until EOS, the token budget runs out, or the
     decider's per-generation firing cap is reached.

Output: the final answer with ``[retrieving: ...]`` markers
showing every splice the decider triggered, plus the full audit
row to stdout as JSON for later analysis.

This demo is intentionally small-corpus (medlocal_concept.db)
and single-model. N=50 grid-style comparison against RAG-k3 /
Mode A lives in a follow-up, once we trust the mechanics.

Run:
  python experiments/midloop_concept/medlocal/mode_c_demo.py \\
    --question "A patient has HIV with CD4 180. Should I start co-trimoxazole?"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path.home()
LMSTUDIO_DIR = HOME / ".lmstudio/models/lmstudio-community"
DEFAULT_MODEL = LMSTUDIO_DIR / "gemma-4-E2B-it-MLX-4bit"
DEFAULT_DB = HOME / ".merken" / "medlocal_concept.db"
DEFAULT_PROJECT = "medlocal_concept"

# Token / firing budgets. The demo is small enough that these
# are hardcoded; a production runtime would expose them as
# policy knobs.
MAX_TOTAL_TOKENS = 800  # gemma-4-E2B-it burns 300-400 tokens on the
                         # <|channel>thought preamble before entering
                         # its final answer; 800 gives budget for the
                         # preamble + answer body where claim-shaped
                         # tokens surface.
CADENCE = 20
WINDOW_SIZE = 40
COOLDOWN = 40
MAX_SPLICES = 3
# Retrieval pool size. We ask for more than one hit so the
# ``fresh`` filter (dedup against already-spliced sources) still
# has candidates to draw from on subsequent fires.
RETRIEVAL_POOL = 5
# H12 (2026-04-21). Splice MULTIPLE top chunks per firing if
# their score passes the confidence threshold. Up to 3 chunks
# per firing, each must clear ``MULTI_SPLICE_SCORE_THRESHOLD``.
# Fewer than 3 qualifying -> splice however many (1 or 2). Zero
# qualifying -> skip the firing. The BBQ fail case showed the
# correct chunk at rank 3; top-1-per-firing guaranteed we
# spliced the wrong chunk first and never reached the right one.
#
# Budget cap on the total splice payload: 2000 tokens worth of
# concatenated chunk text per firing. Individual chunks that
# push past the budget are truncated.
MULTI_SPLICE_MAX_CHUNKS = 3
MULTI_SPLICE_SCORE_THRESHOLD = 0.0161
MULTI_SPLICE_BUDGET_TOKENS = 2000

# Splice wrapper. Keeping the injected K/V as natural English
# (not raw markdown) helps the model integrate the content with
# its own turn. Empirically in phase0c/d gemma handled
# WHO-style markdown fine, but the envelope makes the intent
# explicit in case we experiment with other Builders.
#
# Variants:
# - SPLICE_ENVELOPE_V1 (original, gemma-friendly): bare [Source:]
#   header + chunk text. Works for gemma which has strong safety
#   tuning that re-anchors to the user question. Fails on Qwen3+
#   which reads the leading "user: ..." / "assistant: ..." chunk
#   prefixes as ChatML-style turn markers and hallucinates
#   additional [Source:] blocks of its own to continue the
#   pattern (observed 2026-04-22 on Qwen3.5-4B-OptiQ smoke --
#   splices=1 but the model emitted 6 forged copies of the same
#   chunk with incremented source indices).
# - SPLICE_ENVELOPE_V2 (Builder-agnostic): fenced excerpt block
#   that is harder for the model to treat as a conversational
#   turn. Use together with ``strip_turn_prefixes=True`` which
#   removes literal "user: " / "assistant: " prefixes from the
#   chunk text before splicing, so the Builder sees context, not
#   a transcript.
SPLICE_ENVELOPE_V1 = "\n\n[Source: {source_id}]\n{text}\n\n"
SPLICE_ENVELOPE_V2 = (
    "\n\n<<<MEMORY_EXCERPT source={source_id}>>>\n"
    "{text}\n"
    "<<<END_MEMORY_EXCERPT>>>\n\n"
)
SPLICE_ENVELOPE = SPLICE_ENVELOPE_V1

# Confident-mode system-prompt equivalent for gemma-4-E2B-it.
# Gemma does not support a dedicated system role, so the
# instruction is prefixed to the user message. Without this
# preface the model often refuses personal-info questions with
# a long "I am an AI, cannot give medical advice" meta-reasoning
# preamble (observed on LongMemEval N=2 smoke); the mode_a
# baselines already use a "confident" system prompt, so this
# keeps Mode C on the same footing for head-to-head comparison.
PROMPT_PREFACE = (
    "Answer the question directly using the context available. "
    "Be specific, quote numbers and names verbatim. Do not hedge, "
    "do not refuse. If the answer is not in memory, say 'not in "
    "memory' rather than guessing.\n\n"
)


@dataclass
class SpliceEvent:
    """One retrieval+splice event inside a single generation."""

    trigger_token_index: int
    reason: str
    window_text: str
    source_id: str
    source_score: float | None
    splice_tokens: int
    pre_decision_text: str  # what the Builder had said so far


@dataclass
class ModeCResult:
    """Audit row for one end-to-end Mode C run."""

    question: str
    answer_tokens: list[int] = field(default_factory=list)
    answer_text: str = ""
    splices: list[SpliceEvent] = field(default_factory=list)
    total_tokens_sampled: int = 0
    wall_s: float = 0.0
    terminated_reason: str = ""


def _apply_chat(
    tokenizer,
    user_text: str,
    preface: str | None = None,
    enable_thinking: bool | None = None,
) -> str:
    # Prefix the confident-mode instruction so gemma does not
    # drop into its refusal preamble on personal-info questions.
    # ``preface`` overrides the module-level PROMPT_PREFACE when
    # supplied -- used by the CLI to swap in variants without
    # editing source (H6 variants A/B).
    #
    # ``enable_thinking`` (Qwen3+ family): when False the chat
    # template inserts an EMPTY ``<think></think>`` pair so the
    # model skips its thinking preamble. For Mode C this saves
    # ~300-400 tokens of budget that otherwise disappear into
    # meta-reasoning before the answer body surfaces. Ignored by
    # tokenizers that don't accept the kwarg (gemma, llama).
    prefaced = (preface if preface is not None else PROMPT_PREFACE) + user_text
    messages = [{"role": "user", "content": prefaced}]
    try:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Tokenizer does not accept enable_thinking; retry without it.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:  # pragma: no cover
        return prefaced


def _encode(tokenizer, text: str, *, add_special: bool) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=add_special)
    except TypeError:
        ids = tokenizer.encode(text)
        if not add_special and ids and ids[0] == getattr(tokenizer, "bos_token_id", -1):
            ids = ids[1:]
        return ids


def _stream_until_fire_or_eos(
    model,
    tokenizer,
    cache,
    input_ids,
    max_new_tokens: int,
    decider,
) -> tuple[str, list[int], object]:
    """Run generate_step until (a) the decider fires, (b) EOS
    surfaces, or (c) the batch budget is exhausted. Returns
    ``(reason, tokens_emitted, last_decision)``.

    ``reason`` in {"fire", "eos", "budget"}.

    Decoding uses a cumulative tail-decode pattern to handle
    BPE UTF-8 fragmentation correctly: a multi-byte character
    can span several tokens, and decoding each token alone
    emits replacement characters. We decode the tail of the
    running token list on every step and feed only the delta
    to the decider so the decider's window text is always
    valid UTF-8.
    """
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    eos_id = getattr(tokenizer, "eos_token_id", None)
    tokens: list[int] = []
    last_decision = None
    last_decoded_prefix = ""

    for tok, _lp in generate_step(
        prompt=mx.array(input_ids),
        model=model,
        max_tokens=max_new_tokens,
        prompt_cache=cache,
    ):
        tok_int = int(tok)
        tokens.append(tok_int)
        if eos_id is not None and tok_int == eos_id:
            return "eos", tokens, last_decision
        # Full decode each step so multi-byte UTF-8 characters
        # that span token boundaries resolve correctly. O(N^2)
        # in tokens but each decode is cheap; for N<=800 the
        # total cost is <1% of generation wall time.
        decoded = tokenizer.decode(tokens)
        if decoded.startswith(last_decoded_prefix):
            delta = decoded[len(last_decoded_prefix):]
        else:
            # Rare: BPE "undo" where a later token rewrites an
            # earlier partial. Fall back to the whole string so
            # the decider sees a self-consistent window.
            delta = decoded
        last_decoded_prefix = decoded
        last_decision = decider.on_token(delta)
        if last_decision is not None and last_decision.fire:
            return "fire", tokens, last_decision
        if len(tokens) >= max_new_tokens:
            return "budget", tokens, last_decision
    return "budget", tokens, last_decision


def load_model(model_path: Path):
    """Load model + tokenizer once. Callers that run Mode C on
    many questions should reuse the return value rather than
    reloading the model per question (gemma-4-E2B-it takes 2-3s
    to load -- amortising this across N calls is a real win for
    benchmarks).
    """
    from mlx_lm import load

    t0 = time.perf_counter()
    model, tokenizer = load(str(model_path))
    print(f"[model] loaded in {time.perf_counter()-t0:.1f}s")
    return model, tokenizer


def run_mode_c(
    model_path: Path,
    db_path: Path,
    project: str,
    question: str,
    *,
    model=None,
    tokenizer=None,
    force_first_fire_at_token: int | None = None,
    question_only_retrieval: bool = False,
    relative_threshold_factor: float | None = None,
    retrieval_window_tokens: int | None = None,
    score_threshold_override: float | None = None,
    prompt_preface: str | None = None,
    enable_thinking: bool | None = None,
    splice_envelope: str | None = None,
    strip_turn_prefixes: bool = False,
) -> ModeCResult:
    """Run one Mode C generation. Pass ``model`` + ``tokenizer``
    (from ``load_model``) to skip the per-call model load; omit
    them to load from ``model_path`` inline (suitable for the
    ``__main__`` demo entry).
    """
    import vstash
    from mlx_lm.models.cache import make_prompt_cache

    from experiments.midloop_concept.medlocal.cerebras_midloop import (
        retrieve as cerebras_retrieve,
    )
    from experiments.midloop_concept.medlocal.streaming_claim_detector import (
        StreamingDecider,
    )
    from merken.policies.claim_detector import HeuristicClaimDetector

    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_path)

    t0 = time.perf_counter()
    # Explicit collection="default" so the Memory opened here
    # hits the same collection our ingestion writes to (both
    # benchmark and demo callers default to "default").
    mem = vstash.Memory(db=str(db_path), project=project, collection="default")
    decider = StreamingDecider(
        detector=HeuristicClaimDetector(),
        cadence=CADENCE,
        window_size=WINDOW_SIZE,
        cooldown=COOLDOWN,
        max_firings_per_generation=MAX_SPLICES,
        force_first_fire_at_token=force_first_fire_at_token,
    )

    result = ModeCResult(question=question)
    try:
        chatted = _apply_chat(
            tokenizer, question,
            preface=prompt_preface,
            enable_thinking=enable_thinking,
        )
        prompt_ids = _encode(tokenizer, chatted, add_special=False)
        cache = make_prompt_cache(model)

        current_input = prompt_ids
        budget = MAX_TOTAL_TOKENS
        # Sources already spliced in this generation -- skip them
        # on subsequent fires so a single near-miss doesn't
        # consume the entire splice budget with the same chunk.
        spliced_sources: set[str] = set()

        while budget > 0:
            print(
                f"[stream] continuing from input of {len(current_input)} "
                f"tokens, budget {budget}"
            )
            reason, tokens, decision = _stream_until_fire_or_eos(
                model, tokenizer, cache, current_input, budget, decider,
            )
            result.answer_tokens.extend(tokens)
            budget -= len(tokens)

            if reason in ("eos", "budget"):
                result.terminated_reason = reason
                break

            if reason == "fire":
                assert decision is not None
                window_text = decision.window_text
                pre_text = tokenizer.decode(result.answer_tokens)
                print(
                    f"[fire] t={decision.token_index} "
                    f"reason={decision.reason}  "
                    f"window tail: {window_text[-80:]!r}"
                )
                # 3-way dual retrieval via the same helper the
                # mode_a baselines use, so Mode C and RAG / Mode
                # A hit the same retrieval substrate (hybrid +
                # fts(q+draft) + fts(q-only) interleaved). Without
                # this Mode C was on vstash's default search while
                # the baselines were on dual; the comparison was
                # biased.
                # H2: when ``question_only_retrieval`` is set, drop
                # the window_text from the query. Motivated by
                # observing that window_text captures the model's
                # refusal / meta-reasoning text which drifts the
                # retrieval away from the user's actual ask.
                # H15: when ``retrieval_window_tokens`` is set, only
                # send the last N tokens of the window (not the full
                # 40-token window). Rationale: the Emily probe
                # showed a long window pulls the target score from
                # 0.0167 down to ~0.0069 for the target chunk while
                # inflating unrelated chunks into the 0.016x range
                # where the absolute threshold cannot distinguish
                # them from signal.
                if question_only_retrieval:
                    ret_query = question
                elif retrieval_window_tokens is not None:
                    ret_query = f"{question}\n{window_text[-retrieval_window_tokens:]}"
                else:
                    ret_query = f"{question}\n{window_text}"
                excerpts = cerebras_retrieve(
                    mem,
                    ret_query,
                    top_k=RETRIEVAL_POOL,
                    retrieval_mode="dual",
                )
                # Filter to fresh (not-yet-spliced) candidates, then
                # keep those whose score clears the confidence
                # threshold, keeping at most MULTI_SPLICE_MAX_CHUNKS.
                # Splice all qualifying chunks in a single firing
                # so the model sees top-1/2/3 in its KV cache at
                # the same position -- addresses the BBQ failure
                # mode where the correct chunk sat at rank 3 but
                # top-1-per-firing could not reach it in time.
                #
                # H14: relative threshold. When
                # ``relative_threshold_factor`` is set, the cutoff
                # becomes ``top1_score * factor`` instead of a fixed
                # absolute. Adapts to query-noise regime: long
                # windowed queries produce inflated scores across the
                # board, short ones have wider gaps between signal
                # and noise. The Emily probe showed the target chunk
                # kept score 0.0167 across all query variants while
                # the rank=2 noise ranged from 0.0036 to 0.0164 --
                # a relative cutoff separates signal from noise in
                # both regimes.
                #
                # H16: ``score_threshold_override`` lets the caller
                # substitute a different absolute threshold for the
                # question-only regime where the default 0.0161 is
                # too aggressive (the absence of window_text
                # collapses the noise floor well below it).
                fresh = [
                    e for e in excerpts
                    if e.get("source_id", "memory") not in spliced_sources
                ]
                if fresh and relative_threshold_factor is not None:
                    top_score = fresh[0].get("score")
                    if isinstance(top_score, (int, float)):
                        effective_threshold = top_score * relative_threshold_factor
                    else:
                        effective_threshold = MULTI_SPLICE_SCORE_THRESHOLD
                elif score_threshold_override is not None:
                    effective_threshold = score_threshold_override
                else:
                    effective_threshold = MULTI_SPLICE_SCORE_THRESHOLD
                qualifying = [
                    e for e in fresh
                    if isinstance(e.get("score"), (int, float))
                    and e["score"] >= effective_threshold
                ][:MULTI_SPLICE_MAX_CHUNKS]

                if not qualifying:
                    print(
                        "[fire] no fresh vstash hits above threshold "
                        f"{effective_threshold:.4f} "
                        f"(already spliced {len(spliced_sources)} sources)"
                        " -- skipping this splice, continuing generation"
                    )
                    current_input = [result.answer_tokens[-1]]
                    continue

                # Build a combined splice from up to 3 chunks, cap
                # the total payload at MULTI_SPLICE_BUDGET_TOKENS so
                # a very long chunk doesn't blow the context.
                # ``splice_envelope`` overrides the module default
                # (V1 bare [Source:] header) so Builder-specific
                # formatting can avoid the "model treats splice
                # as a continued transcript" failure seen with Qwen.
                # ``strip_turn_prefixes`` removes leading "user:" /
                # "assistant:" markers from the chunk text before
                # splicing -- LongMemEval chunks come with those
                # prefixes and Qwen+ reads them as ChatML role
                # markers and hallucinates additional turn blocks.
                active_envelope = (
                    splice_envelope if splice_envelope is not None
                    else SPLICE_ENVELOPE
                )

                def _prepare_chunk_text(t: str) -> str:
                    if not strip_turn_prefixes or not t:
                        return t
                    import re as _re_strip
                    # Strip a leading "user:" or "assistant:" token,
                    # optionally preceded by whitespace. Only the
                    # very first one -- interior occurrences stay
                    # intact so multi-turn chunks retain their
                    # conversational shape minus the outermost
                    # role marker.
                    return _re_strip.sub(
                        r"^\s*(user|assistant)\s*:\s*", "",
                        t, count=1, flags=_re_strip.IGNORECASE
                    )

                combined_parts: list[str] = []
                combined_ids: list[int] = []
                picked: list[dict] = []
                for c in qualifying:
                    src = c.get("source_id", "memory")
                    part = active_envelope.format(
                        source_id=src,
                        text=_prepare_chunk_text(c.get("text", "")),
                    )
                    part_ids = _encode(tokenizer, part, add_special=False)
                    if len(combined_ids) + len(part_ids) > MULTI_SPLICE_BUDGET_TOKENS:
                        # Truncate this chunk to fit the remaining
                        # budget. Cut on the token side to avoid
                        # slicing mid-character.
                        remaining = MULTI_SPLICE_BUDGET_TOKENS - len(combined_ids)
                        if remaining <= 50:
                            # Not enough room to be useful -- stop.
                            break
                        part_ids = part_ids[:remaining]
                    combined_ids.extend(part_ids)
                    combined_parts.append(part)
                    spliced_sources.add(src)
                    picked.append(c)

                splice_ids = combined_ids
                source_ids = [p.get("source_id", "memory") for p in picked]
                score_strs = [
                    f"{p.get('score', 0):.4f}" if isinstance(p.get("score"), (int, float))
                    else "?"
                    for p in picked
                ]
                print(
                    f"[retrieving: {len(picked)} chunk(s) | "
                    f"{len(splice_ids)} tok splice | "
                    f"sources={source_ids} scores={score_strs}]"
                )

                # Record one SpliceEvent per chunk in the combined
                # splice so the audit trail captures each source
                # that entered the cache at this firing.
                for p in picked:
                    part_text = active_envelope.format(
                        source_id=p.get("source_id", "memory"),
                        text=_prepare_chunk_text(p.get("text", "")),
                    )
                    part_ids = _encode(tokenizer, part_text, add_special=False)
                    result.splices.append(SpliceEvent(
                        trigger_token_index=decision.token_index,
                        reason=decision.reason,
                        window_text=window_text,
                        source_id=str(p.get("source_id", "memory")),
                        source_score=p.get("score"),
                        splice_tokens=len(part_ids),
                        pre_decision_text=pre_text,
                    ))

                # We do NOT add splice_ids to answer_tokens.
                # The earlier iteration tried that ("fill the
                # transcript gap") but the splice content is
                # NOT what the model output -- it is what entered
                # the model's KV cache. Including it in
                # answer_text bleeds the retrieved chunk verbatim
                # into the user-facing string, which confuses
                # downstream oracle scoring (oracle ends up
                # grading the retrieved chunk instead of the
                # model's answer). The splice payloads live in
                # SpliceEvent.splice_tokens for audit-trail
                # completeness; answer_tokens stays faithful to
                # what the model actually sampled.

                current_input = splice_ids

        result.answer_text = tokenizer.decode(result.answer_tokens)
        # answer_tokens now contains ONLY sampled tokens (no
        # splice payloads), so total_tokens_sampled is just the
        # length. Matches how the Cerebras-side baselines report
        # token counts.
        result.total_tokens_sampled = len(result.answer_tokens)
        if not result.terminated_reason:
            # Natural exit from ``while budget > 0`` without an
            # inner break. Record it explicitly so audit rows do
            # not silently land with an empty reason.
            result.terminated_reason = "budget_exhausted"
    finally:
        mem.close()
    result.wall_s = time.perf_counter() - t0
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        default=(
            "A patient with HIV has a CD4 count of 180. At what CD4 "
            "threshold does the WHO guideline recommend starting "
            "co-trimoxazole prophylaxis?"
        ),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    args = parser.parse_args()

    if not args.model.exists():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2
    if not args.db.exists():
        print(f"db not found: {args.db}", file=sys.stderr)
        return 2

    try:
        result = run_mode_c(args.model, args.db, args.project, args.question)
    except Exception as exc:  # noqa: BLE001
        print(f"RED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print("\n" + "=" * 72)
    print("FINAL ANSWER")
    print("=" * 72)
    print(result.answer_text)
    print("\n" + "=" * 72)
    print(
        f"SUMMARY: {result.total_tokens_sampled} tokens in "
        f"{result.wall_s:.1f}s, {len(result.splices)} splice(s), "
        f"terminated={result.terminated_reason}"
    )
    for i, s in enumerate(result.splices):
        print(
            f"  splice {i}: t={s.trigger_token_index} "
            f"src={s.source_id} ({s.splice_tokens} tok)"
        )

    out = Path(__file__).parent / "mode_c_demo_last.json"
    out.write_text(json.dumps({
        "question": result.question,
        "answer_text": result.answer_text,
        "total_tokens_sampled": result.total_tokens_sampled,
        "wall_s": result.wall_s,
        "terminated_reason": result.terminated_reason,
        "splices": [
            {
                "trigger_token_index": s.trigger_token_index,
                "reason": s.reason,
                "source_id": s.source_id,
                "source_score": s.source_score,
                "splice_tokens": s.splice_tokens,
                "window_text_tail": s.window_text[-200:],
            }
            for s in result.splices
        ],
    }, indent=2) + "\n")
    print(f"\naudit: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
