"""Phase 0b -- KV-cache splice feasibility.

Phase 0 (phase0_feasibility.py) proved we can observe + stop +
restart a generation stream. That's Mode A/B mechanics.

Phase 0b asks the harder question: can we APPEND context to the
running K/V cache mid-stream and have the continuation reflect
it WITHOUT a restart? That's Mode C -- the production shape Jay
wants. If this is green, Mode C is unlocked; if red, we ship
Mode B (streaming + abort + regenerate) and document why.

Mechanism
---------

mlx_lm.models.cache.KVCache stores one (keys, values) tensor
per layer. Each forward pass through the model with a given
cache appends the new K/V at each layer. generate_step with a
pre-existing prompt_cache continues from where the cache left
off -- processing the new "prompt" through the model prefill
extends the cache, then generation resumes.

So the splice pattern is:

  1. Run generate_step(initial_prompt, cache) for N tokens.
     Cache now holds K/V for: initial_prompt + N generated.
  2. Run generate_step(splice_text, cache) for M tokens.
     Prefill appends splice_text K/V to cache, then sampling
     continues from the end of splice_text.

If generation in step 2 reflects the splice content, splicing
works. If step 2's output looks identical to what step 1 would
have kept producing, the splice was ignored.

Probe
-----

Baseline: "Count to 10: " -> generate 16 tokens.
Spliced:   "Count to 10: " -> generate 5 tokens ->
            splice "\nActually use capital letters: " ->
            generate 16 tokens.

If the post-splice continuation shifts from digits to letters,
the splice took effect. If it keeps producing digits, the
splice was ignored.

Success criterion: the spliced continuation contains at least
one capital letter (A-Z) in the first 16 post-splice tokens
AND the baseline continuation contains only digits / commas /
spaces. That's a clear behavioral delta with no ambiguity.

Run
---

  python experiments/midloop_concept/medlocal/phase0b_kv_splice.py

Output: a side-by-side report + a machine-readable
phase0b_report.json next to this script, so a future session
can grep the verdict without re-running.
"""

from __future__ import annotations

import json
import re
import sys
import time
import traceback
from pathlib import Path

HOME = Path.home()
LMSTUDIO_DIR = HOME / ".lmstudio/models/lmstudio-community"
# Phase 0 proved both of these stream + break + relaunch cleanly.
# E2B is the smaller one, preferred for iteration speed.
DEFAULT_MODEL = LMSTUDIO_DIR / "gemma-4-E2B-it-MLX-4bit"

# The probe prompt. Open-ended enough that the model naturally
# keeps producing numeric content if left alone, so a letter-
# shift after splice is a clear behavioral signal.
PROMPT = "Count to 10: "
# Pre-splice generation length. Short enough that the splice
# arrives while the model is still in "numeric counting" mode.
PRE_SPLICE_TOKENS = 5
# The splice. An explicit instruction plus an exemplar (A, B, C)
# so the model can't plausibly ignore it as noise.
SPLICE_TEXT = "\nActually, use capital letters instead: A, B, C,"
# Post-splice / post-baseline generation length. 16 tokens is
# enough to make a letters-vs-digits signal unambiguous.
POST_SPLICE_TOKENS = 16

CAP_LETTER = re.compile(r"[A-Z]")
DIGIT = re.compile(r"[0-9]")


def _gen(model, tokenizer, ids, cache, max_new: int) -> list[int]:
    """Run generate_step with an existing prompt_cache. Returns
    the newly-generated token ids. If ``ids`` is non-empty the
    prefill extends the cache with their K/V first.
    """
    from mlx_lm.generate import generate_step
    import mlx.core as mx

    out: list[int] = []
    # Empty prompt is not valid for generate_step; the caller
    # always passes at least one token.
    arr = mx.array(ids)
    for i, (tok, _lp) in enumerate(generate_step(
        prompt=arr,
        model=model,
        max_tokens=max_new,
        prompt_cache=cache,
    )):
        out.append(int(tok))
        if len(out) >= max_new:
            break
    return out


def _encode(tokenizer, text: str, *, add_special: bool) -> list[int]:
    # Tokenizers expose different kwargs depending on backend
    # (sentencepiece vs tiktoken vs hf fast). Handle the common
    # cases without trying to be too clever.
    try:
        return tokenizer.encode(text, add_special_tokens=add_special)
    except TypeError:
        ids = tokenizer.encode(text)
        # If add_special is False and we cannot request it, slice
        # the likely BOS off so the splice does not reset context.
        if not add_special and ids and ids[0] == getattr(tokenizer, "bos_token_id", -1):
            ids = ids[1:]
        return ids


def run_probe(model_path: Path) -> dict:
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    report: dict = {
        "model": model_path.name,
        "model_path": str(model_path),
        "ok": False,
        "facts": [],
    }

    def fact(s: str) -> None:
        report["facts"].append(s)
        print(f"  . {s}")

    t0 = time.perf_counter()
    model, tokenizer = load(str(model_path))
    fact(f"loaded in {time.perf_counter()-t0:.2f}s")

    prompt_ids = _encode(tokenizer, PROMPT, add_special=True)

    # --- baseline ---------------------------------------------
    # No splice. Generate PRE_SPLICE_TOKENS + POST_SPLICE_TOKENS
    # tokens straight. This is what the model would produce
    # without intervention; the spliced run is compared against
    # this.
    baseline_cache = make_prompt_cache(model)
    baseline_tokens = _gen(
        model, tokenizer, prompt_ids, baseline_cache,
        PRE_SPLICE_TOKENS + POST_SPLICE_TOKENS,
    )
    baseline_text = tokenizer.decode(baseline_tokens)
    fact(f"baseline tokens: {baseline_text!r}")

    # --- spliced ---------------------------------------------
    spliced_cache = make_prompt_cache(model)
    pre_tokens = _gen(
        model, tokenizer, prompt_ids, spliced_cache, PRE_SPLICE_TOKENS,
    )
    pre_text = tokenizer.decode(pre_tokens)
    fact(f"pre-splice tokens: {pre_text!r}")

    # Splice. add_special=False so we do NOT inject a fresh BOS
    # into the middle of the stream -- that would reset the
    # model's notion of sequence start and defeat the splice.
    splice_ids = _encode(tokenizer, SPLICE_TEXT, add_special=False)
    fact(f"splice payload: {len(splice_ids)} tokens, text={SPLICE_TEXT!r}")

    post_tokens = _gen(
        model, tokenizer, splice_ids, spliced_cache, POST_SPLICE_TOKENS,
    )
    post_text = tokenizer.decode(post_tokens)
    fact(f"post-splice tokens: {post_text!r}")

    # --- verdict ---------------------------------------------
    # Success: post-splice shifted to letters AND baseline tail
    # kept producing digits. This is the "behavioral delta"
    # check -- both conditions together avoid false positives
    # from either stream randomly producing letters.
    baseline_tail = tokenizer.decode(baseline_tokens[PRE_SPLICE_TOKENS:])
    post_has_letter = bool(CAP_LETTER.search(post_text))
    baseline_tail_has_letter = bool(CAP_LETTER.search(baseline_tail))
    post_has_digit = bool(DIGIT.search(post_text))
    baseline_tail_has_digit = bool(DIGIT.search(baseline_tail))

    fact(f"baseline tail (same position as post-splice): {baseline_tail!r}")
    fact(
        "letter/digit signals: "
        f"post-splice letter={post_has_letter} digit={post_has_digit} | "
        f"baseline-tail letter={baseline_tail_has_letter} digit={baseline_tail_has_digit}"
    )

    # Strict verdict: the splice demonstrably changed the
    # continuation. Letter appears in the spliced continuation
    # that did NOT appear in the baseline at the same token
    # offset.
    splice_worked = post_has_letter and not baseline_tail_has_letter
    report["baseline_text"] = baseline_text
    report["spliced_text"] = pre_text + SPLICE_TEXT + post_text
    report["ok"] = splice_worked
    report["verdict"] = (
        "KV-SPLICE WORKS -- Mode C feasible: splice altered the "
        "continuation without restart."
        if splice_worked
        else "KV-SPLICE UNCLEAR -- baseline tail and spliced "
             "continuation do not show a clear letter-shift "
             "signal. Inspect the text fields before concluding."
    )
    fact(report["verdict"])
    return report


def main() -> int:
    model_path = DEFAULT_MODEL
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1]).expanduser()
    if not model_path.exists():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 2
    try:
        report = run_probe(model_path)
    except Exception as exc:
        print("RED:", exc, file=sys.stderr)
        traceback.print_exc()
        report = {
            "model": model_path.name,
            "model_path": str(model_path),
            "ok": False,
            "verdict": f"RED -- {exc.__class__.__name__}: {exc}",
            "facts": [],
            "error_traceback": traceback.format_exc(),
        }
    out = Path(__file__).parent / "phase0b_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nreport: {out}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
