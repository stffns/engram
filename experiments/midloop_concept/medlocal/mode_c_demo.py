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
TOP_K = 1  # one chunk per splice for the demo -- keeps the
           # injected payload focused

# Splice wrapper. Keeping the injected K/V as natural English
# (not raw markdown) helps the model integrate the content with
# its own turn. Empirically in phase0c/d gemma handled
# WHO-style markdown fine, but the envelope makes the intent
# explicit in case we experiment with other Builders.
SPLICE_ENVELOPE = "\n\n[Source: {source_id}]\n{text}\n\n"

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


def _apply_chat(tokenizer, user_text: str) -> str:
    # Prefix the confident-mode instruction so gemma does not
    # drop into its refusal preamble on personal-info questions.
    prefaced = PROMPT_PREFACE + user_text
    messages = [{"role": "user", "content": prefaced}]
    try:
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
    """
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    eos_id = getattr(tokenizer, "eos_token_id", None)
    tokens: list[int] = []
    last_decision = None

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
        tok_text = tokenizer.decode([tok_int])
        last_decision = decider.on_token(tok_text)
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
) -> ModeCResult:
    """Run one Mode C generation. Pass ``model`` + ``tokenizer``
    (from ``load_model``) to skip the per-call model load; omit
    them to load from ``model_path`` inline (suitable for the
    ``__main__`` demo entry).
    """
    import vstash
    from mlx_lm.models.cache import make_prompt_cache

    from experiments.midloop_concept.medlocal.streaming_claim_detector import (
        StreamingDecider,
    )
    from merken.policies.claim_detector import HeuristicClaimDetector

    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_path)

    t0 = time.perf_counter()
    mem = vstash.Memory(db=str(db_path), project=project)
    decider = StreamingDecider(
        detector=HeuristicClaimDetector(),
        cadence=CADENCE,
        window_size=WINDOW_SIZE,
        cooldown=COOLDOWN,
        max_firings_per_generation=MAX_SPLICES,
    )

    result = ModeCResult(question=question)
    try:
        chatted = _apply_chat(tokenizer, question)
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
                # Search memory using the recent output window --
                # in production we would also fuse the question
                # and a learned query extractor, but the window
                # alone is good enough for the demo.
                # Pull up to 5 hits so we can dedup against
                # already-spliced sources and still return the
                # next-best chunk. Without this the decider often
                # re-fires on near-duplicate content and the same
                # source gets spliced multiple times.
                hits = mem.search(f"{question}\n{window_text}", top_k=5)
                fresh = [
                    h for h in hits
                    if str(getattr(h, "title", "memory")) not in spliced_sources
                ]
                if not fresh:
                    print(
                        "[fire] no fresh vstash hits "
                        f"(already spliced {len(spliced_sources)} sources) "
                        "-- skipping splice"
                    )
                    result.terminated_reason = "no_fresh_retrieval"
                    break
                top = fresh[0]
                spliced_sources.add(str(getattr(top, "title", "memory")))
                splice_text = SPLICE_ENVELOPE.format(
                    source_id=getattr(top, "title", "memory"),
                    text=getattr(top, "text", ""),
                )
                splice_ids = _encode(tokenizer, splice_text, add_special=False)
                print(
                    f"[retrieving: {getattr(top, 'title', '?')} | "
                    f"{len(splice_ids)} tok splice]"
                )

                result.splices.append(SpliceEvent(
                    trigger_token_index=decision.token_index,
                    reason=decision.reason,
                    window_text=window_text,
                    source_id=str(getattr(top, "title", "memory")),
                    source_score=getattr(top, "score", None),
                    splice_tokens=len(splice_ids),
                    pre_decision_text=pre_text,
                ))

                current_input = splice_ids

        result.answer_text = tokenizer.decode(result.answer_tokens)
        result.total_tokens_sampled = len(result.answer_tokens)
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
