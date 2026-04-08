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
    4. Exact-text duplicate already seen by this decider instance → skip
       (``dup_exact``). Comparison is whitespace-normalized and uses an
       in-process set, NOT a recall against vstash. Recall-based dedup
       was the original design but a real-data run on LongMemEval showed
       it dominates ingest cost (each write triggered a 700ms+ hybrid
       search against an expanding index — O(N²) per haystack). Exact
       dedup is what this rule actually wants, and a hash lookup is the
       right tool. The ``ctx.recall`` callable stays in the WriteContext
       for *future* similarity-based deciders that genuinely need it.
    5. Otherwise → write (``novel``) and remember the normalized text.

    No LLM, no embedding-similarity threshold. Embedding-based novelty
    detection is intentionally deferred until a benchmark says it earns
    its cost.

    Per-instance state: ``HeuristicWriteDecider`` is **stateful**. The
    ``seen`` set lives for the life of the decider, which mirrors the life
    of the engram ``Memory`` it's attached to. A fresh ``Memory`` gets a
    fresh decider with an empty set — no cross-conversation contamination.
    """

    name = "HeuristicWriteDecider"

    def __init__(
        self,
        *,
        min_chars: int = 8,
        max_chars: int = 100_000,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._seen: set[str] = set()

    def decide(self, event: Event, ctx: WriteContext) -> Decision:  # noqa: ARG002
        text = event.text.strip()

        if not text:
            return Decision(False, "empty", 1.0, self.name)

        if len(text) < self.min_chars:
            return Decision(False, f"too_short:<{self.min_chars}", 1.0, self.name)

        if len(text) > self.max_chars:
            return Decision(False, f"too_long:>{self.max_chars}", 1.0, self.name)

        norm = _normalize(text)
        if norm in self._seen:
            return Decision(False, "dup_exact", 1.0, self.name)

        self._seen.add(norm)
        return Decision(True, "novel", 0.8, self.name)
