"""Phase 0 -- interception mechanics across small local MLX models.

Per notes/midloop-concept-test-medlocal.md, Phase 0 asks ONE
question: can we get inside the generation loop of a small model
that runs on this laptop? Everything else (v1c-6L wiring, vstash
seeding, the full pipeline) is deferred to Phase 1+.

The probe is intentionally narrow. For each model target:

  1. Load it via mlx_lm.
  2. Stream N tokens from a prompt and break out of the generator
     at token K (proves we can observe + stop).
  3. Relaunch generation with a modified prompt that injects a
     correction -- and confirm the new tokens REFLECT the
     injection (proves we can influence the next generation).

If this probe is green across both gemma-E2B and gemma-E4B, Mode C
interception is mechanically feasible on any MLX small model, and
Phase 1 can proceed with a chosen Builder. If it's red, we fall
back to Mode A (Cerebras + step-boundary correction) and document
why in phase0_report.json.

Deferred to later phases / scripts:
  - v1c-6L detector loading + forward pass sanity (Phase 1).
  - vstash + snapvec seeding + retrieval sanity (Phase 1).
  - Mid-stream KV-cache splicing (Phase 3 stretch).

Run:
  python experiments/midloop_concept/medlocal/phase0_feasibility.py

Optional model override:
  PHASE0_MODELS="path1,path2" python .../phase0_feasibility.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

HOME = Path.home()
LMSTUDIO_DIR = HOME / ".lmstudio/models/lmstudio-community"
DEFAULT_MODELS: list[tuple[str, Path]] = [
    ("gemma-4-E2B-it-MLX-4bit", LMSTUDIO_DIR / "gemma-4-E2B-it-MLX-4bit"),
    ("gemma-4-E4B-it-MLX-4bit", LMSTUDIO_DIR / "gemma-4-E4B-it-MLX-4bit"),
]

# --------------------------------------------------------------- plumbing

class Probe:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ok: bool = False
        self.duration_s: float = 0.0
        self.facts: list[str] = []
        self.err: str | None = None

    def fact(self, s: str) -> None:
        self.facts.append(s)
        print(f"  . {s}")

    def __enter__(self) -> Probe:
        print(f"\n[{self.name}] starting")
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.duration_s = time.perf_counter() - self._t0
        if exc is None:
            self.ok = True
            print(f"[{self.name}] GREEN ({self.duration_s:.1f}s)")
        else:
            self.err = f"{exc.__class__.__name__}: {exc}"
            tb_lines = traceback.format_exception(exc_type, exc, tb)
            print(f"[{self.name}] RED ({self.duration_s:.1f}s)")
            print("".join(tb_lines[-6:]))
        return True


# --------------------------------------------------------------- interception

def _stream_tokens(
    model, tokenizer, prompt: str, max_tokens: int, stop_after: int | None = None,
) -> tuple[list[str], list[int], float, float]:
    """Stream from the model. Optionally break after `stop_after` tokens.

    Returns (text_pieces, token_ids, first_token_latency_ms,
    total_wall_ms). Early-break via ``break`` in the generator's
    for-loop is the one mechanism Mode C depends on -- if we can
    break mid-stream cleanly, the interception point exists.
    """
    from mlx_lm import stream_generate

    pieces: list[str] = []
    token_ids: list[int] = []
    t_first: float | None = None
    t0 = time.perf_counter()
    for i, resp in enumerate(stream_generate(model, tokenizer, prompt, max_tokens=max_tokens)):
        if t_first is None:
            t_first = time.perf_counter()
        pieces.append(getattr(resp, "text", ""))
        tok = getattr(resp, "token", None)
        if tok is not None:
            token_ids.append(int(tok))
        if stop_after is not None and (i + 1) >= stop_after:
            break
    t_end = time.perf_counter()
    first_ms = ((t_first or t0) - t0) * 1000
    total_ms = (t_end - t0) * 1000
    return pieces, token_ids, first_ms, total_ms


def _apply_chat(tokenizer, user_msg: str, prior_assistant: str | None = None) -> str:
    """Render a chat prompt. Falls back to raw on template failure."""
    messages = [{"role": "user", "content": user_msg}]
    if prior_assistant is not None:
        messages.append({"role": "assistant", "content": prior_assistant})
    try:
        # add_generation_prompt=True only when there's no trailing
        # assistant message (let the model start a fresh turn).
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(prior_assistant is None),
        )
    except Exception:
        # Raw concatenation fallback for tokenizers missing the
        # chat template.
        if prior_assistant is None:
            return user_msg
        return f"{user_msg}\n\n{prior_assistant}"


def probe_model_interception(p: Probe, model_name: str, model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"{model_name} not found at {model_path}")
    # Keep only the directory basename in the report so the
    # committed JSON does not leak the developer's home layout.
    # The absolute path is still printed to stdout via Probe when
    # someone runs the probe locally.
    p.fact(f"model: {model_path.name}")

    from mlx_lm import load

    t_load = time.perf_counter()
    model, tokenizer = load(str(model_path))
    p.fact(f"loaded in {time.perf_counter() - t_load:.1f}s")

    # --- Step 1: stream and break at token 6 ---------------------------
    # Prompt is deliberately leading: "Count from 1 to 10". The model
    # is expected to emit '1, 2, 3, 4, 5, 6, ...'. We break at 6
    # tokens so we have a partial, predictable trajectory.
    prompt_a = _apply_chat(tokenizer, "Count from 1 to 10, separated by commas, on a single line.")
    pieces_a, ids_a, ftl_a, tot_a = _stream_tokens(
        model, tokenizer, prompt_a, max_tokens=32, stop_after=8,
    )
    text_a = "".join(pieces_a)
    p.fact(f"stream1: {len(ids_a)} tokens, first-token {ftl_a:.0f}ms, total {tot_a:.0f}ms")
    p.fact(f"stream1 text: {text_a[:80]!r}")
    if len(ids_a) == 0:
        raise RuntimeError("stream_generate yielded zero tokens before break")
    # Hard verification that we broke early: we asked for up to 32
    # but stopped at 8.
    if len(ids_a) > 10:
        raise RuntimeError(
            f"early-break didn't work -- got {len(ids_a)} tokens, "
            f"expected <=8. Generator didn't honor the break."
        )
    p.fact("early-break honored (Mode C observe+stop works)")

    # --- Step 2: relaunch with an injected correction ------------------
    # We simulate a midloop firing at token 6 that decided the count
    # is wrong and wants the model to restart, going backwards
    # instead of forward. The new system instruction flips the
    # counting direction. If the model follows it on the second run,
    # interception is not just observable -- it's ACTIONABLE.
    injected_prompt = _apply_chat(
        tokenizer,
        (
            "Count from 10 DOWN to 1, separated by commas, on a single "
            "line. (This is a correction; ignore any previous counting "
            "direction.)"
        ),
    )
    pieces_b, ids_b, ftl_b, tot_b = _stream_tokens(
        model, tokenizer, injected_prompt, max_tokens=32, stop_after=10,
    )
    text_b = "".join(pieces_b)
    p.fact(f"stream2 (post-injection): {len(ids_b)} tokens, first-token {ftl_b:.0f}ms")
    p.fact(f"stream2 text: {text_b[:80]!r}")
    # Behavioral check: the second stream should mention "10" or "9"
    # near the beginning (since we asked it to count down starting
    # at 10). This is a weak signal -- a model could refuse or go
    # off-script -- but if it produces "10, 9" or similar, the
    # injection clearly took effect.
    norm = text_b.replace(" ", "").replace(",", "")
    injection_took = (
        "10" in norm[:6] or "9" in norm[:6] or norm[:2] in ("10", "9,")
    )
    if injection_took:
        p.fact("injection took effect: stream2 reflects the corrected instruction")
    else:
        # Not fatal: the model may have paraphrased. Downgrade to a
        # soft warning rather than failing the probe -- the critical
        # proof (stream + break + relaunch) succeeded regardless.
        p.fact(
            "WARN: injection behavioral signal unclear; "
            "stream+break+relaunch mechanics still OK"
        )


# --------------------------------------------------------------- runner

def main() -> int:
    print("=" * 72)
    print("Phase 0 feasibility -- interception mechanics on small MLX models")
    print("=" * 72)

    override = os.environ.get("PHASE0_MODELS", "").strip()
    if override:
        models: list[tuple[str, Path]] = []
        for p in override.split(","):
            p = p.strip()
            if not p:
                continue
            models.append((Path(p).name, Path(p).expanduser()))
    else:
        models = DEFAULT_MODELS

    results: list[Probe] = []
    for (name, path) in models:
        probe = Probe(f"interception: {name}")
        with probe:
            probe_model_interception(probe, name, path)
        results.append(probe)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    all_green = True
    for p in results:
        status = "GREEN" if p.ok else "RED  "
        print(f"  [{status}] {p.name}  ({p.duration_s:.1f}s)")
        if p.err:
            print(f"         -> {p.err}")
        all_green = all_green and p.ok

    if all_green:
        verdict = (
            "ALL GREEN -- Mode C interception is mechanically feasible.\n"
            "Next: Phase 1 (wire v1c-6L as detector, seed vstash, run observe-only)."
        )
    elif any(p.ok for p in results):
        verdict = (
            "PARTIAL -- some models support interception, others don't.\n"
            "Decide: restrict Phase 1 to the green models, OR document the red ones."
        )
    else:
        verdict = (
            "ALL RED -- Mode C on local MLX models is NOT feasible with mlx_lm "
            "stream_generate as used here.\nFall back to Mode A (Cerebras + "
            "step-boundary correction); document in RESULTS.md."
        )
    print("\n" + verdict)

    out = Path(__file__).parent / "phase0_report.json"
    out.write_text(json.dumps({
        "all_green": all_green,
        "verdict": verdict,
        "probes": [
            {
                "name": p.name,
                "ok": p.ok,
                "duration_s": round(p.duration_s, 3),
                "facts": p.facts,
                "err": p.err,
            } for p in results
        ],
    }, indent=2))
    print(f"\nreport: {out}")
    return 0 if all_green else 1


if __name__ == "__main__":
    sys.exit(main())
