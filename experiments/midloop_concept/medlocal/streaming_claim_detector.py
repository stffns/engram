"""Streaming wrapper over ``merken.policies.claim_detector``.

Milestone 3 of the Mode C plan (notes/mode-c-continuous-decider.md).
The claim_detector scaffolding is one-shot: given a block of text,
return ``ClaimDetection`` with a list of ``ClaimSpan`` objects.
Production Mode C needs the decider to fire PERIODICALLY while the
Builder is streaming tokens -- the gate is stateful, not one-shot.

This wrapper provides:

- A ``StreamingDecider`` that maintains a rolling window of recent
  tokens, calls the underlying ``ClaimDetector`` on every
  ``cadence``-th token, and emits a ``FiringDecision`` each time.
- Overlap (``window_size`` > ``cadence``) or disjoint windows
  (``window_size == cadence``) are both supported -- the concept
  memo explicitly flagged this as a knob.
- A cooldown (``cooldown`` tokens after a positive firing before
  the next fire is allowed) so one confirmed claim does not trigger
  a retrieval storm.
- Budget cap (``max_firings_per_generation``) -- simple responses
  may never fire; complex ones fire up to this cap. The memo's
  "simple responses may not use it, complex ones will consult
  vstash more frequently" invariant.

The wrapper is a pure Python class, no LLM calls of its own. It
delegates detection to whatever ``ClaimDetector`` the caller passes
in (``HeuristicClaimDetector`` for the cheap production default,
``LLMClaimDetector`` for high-fidelity labeling runs, future
nanoGPT-distilled detector for speed).

Probe usage:

    python experiments/midloop_concept/medlocal/streaming_claim_detector.py

This runs the wrapper against two hand-crafted token streams -- a
factual answer that should trigger firings and a conversational
greeting that should trigger zero firings -- and prints the
firing trace. No model load, no API calls.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from merken.policies.claim_detector import (
    ClaimDetectionInput,
    ClaimSpan,
    HeuristicClaimDetector,
)

if TYPE_CHECKING:  # pragma: no cover -- protocol import only for typing
    from merken.policies.claim_detector import ClaimDetector


@dataclass(frozen=True)
class FiringDecision:
    """One decision emitted by the streaming decider at a token
    boundary. A decision with ``fire=True`` corresponds to a
    production-time "call vstash.search on this window" event.
    """

    token_index: int            # global position of this firing
    fire: bool                  # True => decider says "retrieve now"
    reason: str                 # human-readable one-liner
    window_text: str            # the sliding window at firing time
    claims: tuple[ClaimSpan, ...] = ()
    signals: dict = field(default_factory=dict)


@dataclass
class StreamingDecider:
    """Wraps a one-shot ``ClaimDetector`` to fire on a token
    stream. Call ``on_token(text)`` for each token as it is
    sampled; the wrapper updates its rolling window, evaluates
    cadence/cooldown/budget, and returns ``None`` for "do not
    fire" or a ``FiringDecision`` for "fire now".

    All the Mode C knobs the memo listed live here:
    - ``cadence`` -- fire every N tokens (default 20).
    - ``window_size`` -- how many recent tokens the detector
      sees (default 40). ``window_size > cadence`` means sliding
      overlap.
    - ``cooldown`` -- after a positive firing, wait this many
      tokens before allowing another firing.
    - ``max_firings_per_generation`` -- hard cap per stream.
    """

    detector: ClaimDetector
    cadence: int = 20
    window_size: int = 40
    cooldown: int = 30
    max_firings_per_generation: int = 5
    # H1 2026-04-21: force the first firing at a fixed early
    # position regardless of claim patterns. Motivated by the
    # N=30 Mode C observation that the heuristic detector rarely
    # fires before t=700, by which point gemma-4-E2B-it has
    # committed to a refusal. Setting e.g.
    # ``force_first_fire_at_token=30`` guarantees the model gets
    # memory context BEFORE the thinking preamble hardens into
    # a committed answer. ``None`` disables (honours cadence-only
    # firing, the pre-H1 default).
    force_first_fire_at_token: int | None = None
    # internal state. ``_last_fire_at`` is None until the first
    # firing so cooldown does not suppress the first candidate --
    # an earlier implementation initialised this to 0 and reported
    # "cooldown(10 left)" on the first boundary, which was
    # misleading.
    _buffer: list[str] = field(default_factory=list)
    _token_count: int = 0
    _last_fire_at: int | None = None
    _firings: int = 0

    def reset(self) -> None:
        self._buffer.clear()
        self._token_count = 0
        self._last_fire_at = None
        self._firings = 0

    def on_token(self, token_text: str, *, task_description: str = "") -> FiringDecision | None:
        """Consume one token of generated text. Returns a firing
        decision at every cadence-aligned boundary, and ``None``
        at non-boundary tokens. A firing decision with
        ``fire=True`` is the signal to pause generation and call
        vstash; ``fire=False`` boundaries still emit a
        ``FiringDecision`` (with reason=cadence_missed or
        cooldown) so the caller can log / audit.
        """
        self._token_count += 1
        self._buffer.append(token_text)
        if len(self._buffer) > self.window_size:
            # Drop from the head so the tail reflects the most
            # recent ``window_size`` tokens.
            overflow = len(self._buffer) - self.window_size
            del self._buffer[:overflow]

        # H1 forced-first-fire. When the knob is set and we have
        # NOT fired yet, unconditionally emit a fire decision at
        # the configured token index. Subsequent firings honour
        # the normal cadence / cooldown / budget path.
        if (
            self.force_first_fire_at_token is not None
            and self._firings == 0
            and self._token_count >= self.force_first_fire_at_token
        ):
            self._firings += 1
            self._last_fire_at = self._token_count
            return FiringDecision(
                token_index=self._token_count,
                fire=True,
                reason="forced_first_fire",
                window_text="".join(self._buffer),
                claims=(),
                signals={"forced": 1.0},
            )

        # Only emit a decision on cadence-aligned boundaries --
        # avoids calling the underlying detector every token
        # (expensive for LLMClaimDetector, cheap for
        # HeuristicClaimDetector but still wasted work).
        if self._token_count % self.cadence != 0:
            return None

        # Budget cap -- cheap guard before firing.
        if self._firings >= self.max_firings_per_generation:
            return FiringDecision(
                token_index=self._token_count,
                fire=False,
                reason="budget_exhausted",
                window_text="".join(self._buffer),
            )

        # Cooldown since the last POSITIVE firing. If no firing
        # has happened yet, do not apply cooldown.
        if self._last_fire_at is not None:
            tokens_since = self._token_count - self._last_fire_at
            if tokens_since < self.cooldown:
                return FiringDecision(
                    token_index=self._token_count,
                    fire=False,
                    reason=f"cooldown({self.cooldown - tokens_since} left)",
                    window_text="".join(self._buffer),
                )

        window_text = "".join(self._buffer)
        detection = self.detector.detect(
            ClaimDetectionInput(
                step_text=window_text,
                prev_steps=[],
                task_description=task_description,
            )
        )
        if not detection.has_claim:
            return FiringDecision(
                token_index=self._token_count,
                fire=False,
                reason="no_claim",
                window_text=window_text,
                signals=dict(detection.signals or {}),
            )

        self._firings += 1
        self._last_fire_at = self._token_count
        return FiringDecision(
            token_index=self._token_count,
            fire=True,
            reason=detection.reason,
            window_text=window_text,
            claims=detection.claims,
            signals=dict(detection.signals or {}),
        )

    @property
    def firings_count(self) -> int:
        return self._firings


# --------------------------------------------------------------- probe


def _tokenize_for_probe(text: str) -> list[str]:
    """Cheap whitespace tokenization for the no-model probe. In
    production this would be driven by the real tokenizer's token
    stream; here we just want to feed discrete units to
    ``on_token``.
    """
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in " \n\t.,!?;:":
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return out


PROBES = [
    {
        "name": "factual_answer",
        "text": (
            "The recommended CD4 threshold for starting "
            "co-trimoxazole prophylaxis is 350 cells per mm^3. "
            "Patients should be started on therapy within one "
            "week. The dose is one tablet daily, taken with "
            "food. Monitor for rash and adjust if needed."
        ),
        "expect_firings": True,
    },
    {
        "name": "conversational_greeting",
        "text": (
            "Hi, how are you today? I hope you are doing well. "
            "I just wanted to say thanks for the help earlier. "
            "Let me know if there is anything else you need."
        ),
        "expect_firings": False,
    },
    {
        "name": "refusal",
        "text": (
            "I am an AI assistant and I am not able to provide "
            "medical advice. Please consult with a licensed "
            "healthcare provider for specific dosing "
            "recommendations. I can share general education "
            "material if that helps."
        ),
        "expect_firings": False,
    },
]


def main() -> int:
    detector = HeuristicClaimDetector()
    print(f"[detector] {detector.name}")
    print(
        "[decider] cadence=20 window=40 cooldown=30 "
        "budget=5 (HeuristicClaimDetector)"
    )

    results: dict[str, list[FiringDecision]] = {}
    for probe in PROBES:
        print(f"\n{'='*72}\n[{probe['name']}] expect_firings={probe['expect_firings']}\n{'='*72}")
        print(f"text: {probe['text']!r}")
        wrapper = StreamingDecider(
            detector=detector,
            cadence=20,
            window_size=40,
            cooldown=30,
            max_firings_per_generation=5,
        )
        firings: list[FiringDecision] = []
        tokens = _tokenize_for_probe(probe["text"])
        t0 = time.perf_counter()
        for tok in tokens:
            decision = wrapper.on_token(tok)
            if decision is None:
                continue
            firings.append(decision)
            flag = "FIRE" if decision.fire else "skip"
            print(
                f"  t={decision.token_index:3d} {flag:4s} "
                f"reason={decision.reason} "
                f"claims={len(decision.claims)}"
            )
            if decision.fire:
                print(f"       window: {decision.window_text[:100]!r}")
        dt = time.perf_counter() - t0
        n_fires = wrapper.firings_count
        print(
            f"\n  tokens={len(tokens)} boundaries={len(firings)} "
            f"firings={n_fires} wall={dt*1000:.1f}ms"
        )
        results[probe["name"]] = firings

    # Sanity: the factual probe should fire; the greeting should not.
    factual_fired = any(f.fire for f in results.get("factual_answer", []))
    greeting_fired = any(f.fire for f in results.get("conversational_greeting", []))
    refusal_fired = any(f.fire for f in results.get("refusal", []))
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  factual_answer fired:          {factual_fired}  (expect True)")
    print(f"  conversational_greeting fired: {greeting_fired}  (expect False)")
    print(f"  refusal fired:                 {refusal_fired}  (expect False)")
    passes = factual_fired and (not greeting_fired) and (not refusal_fired)
    print(f"\n  streaming decider separates factual vs filler: {passes}")
    return 0 if passes else 1


if __name__ == "__main__":
    sys.exit(main())
