"""Phase 0d -- mid-turn splice position sweep.

Phase 0c proved long-splice mechanically works when the splice
arrives 5 tokens into the assistant turn (gemma-4-E2B-it is still
emitting its <|channel>thought preamble at that point, so the
splice effectively displaces the thinking). Production Mode C
fires the decider at arbitrary positions -- some early (during
thinking), some late (mid-answer body). This probe asks: **does
splicing still take effect when the model is deep into generating
its own answer content?**

Setup:

- Use one probe that passed at Phase 0c (``cotrimoxazole_cd4`` at
  800-token payload). We are NOT retesting whether the splice
  mechanism works -- Phase 0c already proved that. We are testing
  whether the splice POSITION (how many tokens the model has
  already sampled before the splice) affects fact uptake.
- Sweep ``pre_splice_tokens`` in {5, 30, 60, 100, 150}.
  - 5 -- splice during early thinking (same as Phase 0c).
  - 30 -- splice mid-thinking.
  - 60 -- likely around the thinking->final channel transition.
  - 100 -- splice after the model has started its final answer.
  - 150 -- splice deep into the final answer (late intervention).
- Post-splice budget is 150 tokens so the model has room to
  integrate + produce a recognizable continuation.

Success signal: keyword "350" appears in the post-splice body
(pre-EOT portion) for most / all sweep positions. A clean pass
across positions proves the continuous streaming decider is
architecturally sound -- the decider can fire at any token index
and still redirect the generation via splice.

If the keyword appears at early positions but vanishes at late
positions, the decider must fire early; late firing is useless.
Document the threshold if that is the case.

Output: stdout table + phase0d_report.json next to this file.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

HOME = Path.home()
LMSTUDIO_DIR = HOME / ".lmstudio/models/lmstudio-community"
DEFAULT_MODEL = LMSTUDIO_DIR / "gemma-4-E2B-it-MLX-4bit"

PROBE = {
    "name": "cotrimoxazole_cd4",
    "needle": "all HIV+ patients with CD4 <350",  # to find the chunk
    "prompt": (
        "Q: A patient with HIV has a CD4 count of 180. "
        "At what CD4 threshold does the WHO guideline "
        "recommend starting co-trimoxazole prophylaxis?\n"
        "A:"
    ),
    "expected_keyword": "350",
}

PRE_SPLICE_SWEEP = [5, 30, 60, 100, 150]
POST_SPLICE_TOKENS = 150
SPLICE_LENGTH = 800  # tokens; the length that passed Phase 0c

_EOT_MARKERS = (
    "<end_of_turn>",
    "<|end_of_turn|>",
    "<turn|>",
    "<|eot|>",
)


def _apply_chat(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:  # pragma: no cover
        return user_text


def _encode(tokenizer, text: str, *, add_special: bool) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=add_special)
    except TypeError:
        ids = tokenizer.encode(text)
        if not add_special and ids and ids[0] == getattr(tokenizer, "bos_token_id", -1):
            ids = ids[1:]
        return ids


def _gen(model, tokenizer, ids, cache, max_new: int) -> list[int]:
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    out: list[int] = []
    for tok, _lp in generate_step(
        prompt=mx.array(ids),
        model=model,
        max_tokens=max_new,
        prompt_cache=cache,
    ):
        out.append(int(tok))
        if len(out) >= max_new:
            break
    return out


def _clip_before_eot(text: str) -> str:
    cut = len(text)
    for marker in _EOT_MARKERS:
        i = text.find(marker)
        if i != -1 and i < cut:
            cut = i
    return text[:cut].rstrip()


def _load_splice_text() -> str:
    """Pull the hiv-who chunk (the one that passed Phase 0c) from
    medlocal_concept.db. Raises if the corpus is not seeded.
    """
    from vstash import Memory

    mem = Memory(
        db=str(HOME / ".merken" / "medlocal_concept.db"),
        project="medlocal_concept",
    )
    try:
        hits = mem.search(PROBE["needle"], top_k=10, fts_only=True)
        for h in hits:
            text = getattr(h, "text", "") or ""
            if PROBE["needle"].lower() in text.lower():
                return text
        raise RuntimeError(
            f"could not find hiv-who chunk via needle {PROBE['needle']!r}"
        )
    finally:
        mem.close()


def run(model_path: Path) -> dict:
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    splice_text = _load_splice_text()
    print(f"[load] splice_text len={len(splice_text)} chars")

    t0 = time.perf_counter()
    model, tokenizer = load(str(model_path))
    print(f"[model] loaded in {time.perf_counter()-t0:.1f}s")

    chatted = _apply_chat(tokenizer, PROBE["prompt"])
    prompt_ids = _encode(tokenizer, chatted, add_special=False)
    splice_ids_full = _encode(tokenizer, splice_text, add_special=False)
    splice_ids = splice_ids_full[:SPLICE_LENGTH]
    print(
        f"[encoding] prompt={len(prompt_ids)} tok  splice={len(splice_ids)} "
        f"tok (from {len(splice_ids_full)} available)"
    )

    report: dict = {
        "model": model_path.name,
        "probe": PROBE["name"],
        "keyword": PROBE["expected_keyword"],
        "splice_length": len(splice_ids),
        "runs": {},
    }

    for pre_n in PRE_SPLICE_SWEEP:
        cache = make_prompt_cache(model)
        t_probe = time.perf_counter()

        # Generate pre_n tokens of the model's own continuation.
        pre_tokens = _gen(model, tokenizer, prompt_ids, cache, pre_n)
        pre_text = tokenizer.decode(pre_tokens)

        # Splice + continue in ONE generate_step call. Passing
        # ``splice_ids`` as the prompt prefills the entire splice
        # through the model (populating K/V at each layer) and
        # then samples POST_SPLICE_TOKENS continuation tokens from
        # the enriched cache. Single call avoids the token-
        # duplication issue a two-call pattern would introduce
        # (``max_tokens=1`` + ``[splice_ids[-1]]`` re-prefills the
        # tail and keeps a spurious sampled token in the cache).
        post_tokens = _gen(model, tokenizer, splice_ids, cache, POST_SPLICE_TOKENS)
        post_text = tokenizer.decode(post_tokens)
        body = _clip_before_eot(post_text)
        kw_present = PROBE["expected_keyword"].lower() in body.lower()
        wall_s = time.perf_counter() - t_probe

        print(f"\n--- pre_splice_tokens={pre_n} ({wall_s:.1f}s) ---")
        print(f"pre:   {pre_text!r}")
        print(f"post:  {post_text[:220]!r}")
        print(f"body (pre-EOT): {body[:200]!r}")
        print(f"keyword_present={kw_present}")

        report["runs"][pre_n] = {
            "pre_text": pre_text,
            "post_text": post_text,
            "body_pre_eot": body,
            "keyword_present": kw_present,
            "wall_s": wall_s,
        }

    hits = [n for n, r in report["runs"].items() if r["keyword_present"]]
    report["pre_positions_with_keyword"] = hits
    report["any_position_passes"] = bool(hits)
    report["all_positions_pass"] = len(hits) == len(PRE_SPLICE_SWEEP)
    print("\n" + "=" * 72)
    print("MID-TURN SPLICE SWEEP VERDICT")
    print("=" * 72)
    print(f"  positions tested: {PRE_SPLICE_SWEEP}")
    print(f"  positions with keyword: {hits}")
    print(f"  any passes: {report['any_position_passes']}")
    print(f"  all pass:  {report['all_positions_pass']}")
    return report


def main() -> int:
    model_path = DEFAULT_MODEL
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1]).expanduser()
    if not model_path.exists():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 2
    try:
        report = run(model_path)
    except Exception as exc:  # noqa: BLE001
        print(f"RED: {exc}", file=sys.stderr)
        traceback.print_exc()
        report = {"error": repr(exc), "traceback": traceback.format_exc()}

    out = Path(__file__).parent / "phase0d_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nreport: {out}")
    return 0 if report.get("any_position_passes") else 1


if __name__ == "__main__":
    sys.exit(main())
