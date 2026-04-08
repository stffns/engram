"""``should_remember`` policies — the first decision primitive.

Two implementations land in Phase 1:

- ``AlwaysWrite`` — the mempalace baseline ("store everything"). Useful as a
  benchmark control in ``experiments/longmemeval/``: it tells us whether
  engram's filtering is helping or hurting.
- ``HeuristicWriteDecider`` — engram's default. No LLM. Skips empty,
  too-short, too-long, and exact-duplicate writes. The minimum policy that
  is honestly better than ``AlwaysWrite`` for a real loop.

LLM-backed deciders (novelty scoring, intent classification) live in a later
phase, gated on a benchmark per CONSTITUTION §9.
"""

from __future__ import annotations

from engram.policies.types import Decision, Event, WriteContext


def _normalize(text: str) -> str:
    """Whitespace-collapsed lower-bound for exact-dedup comparison."""
    return " ".join(text.split())


class AlwaysWrite:
    """Baseline: every event gets written. Mirrors the mempalace philosophy.

    Exists so that experiments can isolate the cost/benefit of filtering.
    Not the engram default.
    """

    name = "AlwaysWrite"

    def decide(self, event: Event, ctx: WriteContext) -> Decision:  # noqa: ARG002
        return Decision(
            write=True,
            reason="always_write",
            confidence=1.0,
            policy=self.name,
        )


class HeuristicWriteDecider:
    """Default Phase 1 policy.

    Rules, in order:

    1. Empty / whitespace-only → skip (``empty``).
    2. Length below ``min_chars`` → skip (``too_short:<N``).
    3. Length above ``max_chars`` → skip (``too_long:>N``). Not a hard
       failure; the user can override the decider if they want truly large
       writes.
    4. Exact-text duplicate already in the same layer → skip
       (``dup_exact``). Comparison is whitespace-normalized.
    5. Otherwise → write (``novel``).

    No LLM, no embedding-similarity threshold. Embedding-based novelty
    detection is intentionally deferred until a benchmark says it earns its
    cost.
    """

    name = "HeuristicWriteDecider"

    def __init__(
        self,
        *,
        min_chars: int = 8,
        max_chars: int = 100_000,
        dedup_top_k: int = 3,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.dedup_top_k = dedup_top_k

    def decide(self, event: Event, ctx: WriteContext) -> Decision:
        text = event.text.strip()

        if not text:
            return Decision(False, "empty", 1.0, self.name)

        if len(text) < self.min_chars:
            return Decision(False, f"too_short:<{self.min_chars}", 1.0, self.name)

        if len(text) > self.max_chars:
            return Decision(False, f"too_long:>{self.max_chars}", 1.0, self.name)

        norm = _normalize(text)
        try:
            hits = ctx.recall(text, self.dedup_top_k, event.layer)
        except Exception:
            # Recall failures must never block writes. The audit log will
            # show ``novel`` with no dedup check, which is the correct
            # signal that something is wrong upstream.
            hits = []

        for hit in hits:
            hit_text = getattr(hit, "text", "") or ""
            if _normalize(hit_text) == norm:
                return Decision(False, "dup_exact", 1.0, self.name)

        return Decision(True, "novel", 0.8, self.name)
