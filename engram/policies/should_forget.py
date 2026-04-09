"""``should_forget`` policies — decision primitive #4, the last one.

Decides *whether* an episodic event should be tombstoned. The
companion to ``should_consolidate``: once an event's content is
preserved in a semantic fact, the raw episodic copy becomes
redundant for recall purposes. Forgetting it reclaims vstash
space and reduces noise in recall.

**Tombstone, not delete (CONSTITUTION §5.1).** Engram's forget
operation is reversible by design:

1. Before removing from the main collection, the full text of the
   event is written to ``engram_tombstones``, a dedicated
   collection that mirrors ``engram_audit`` in philosophy — it is
   the authoritative record of what was forgotten and why.
2. Then the event is ``vstash.remove()``'d from its original
   collection, so it no longer surfaces in recall.
3. The semantic fact's ``derived_from`` still points at the
   original event path. The provenance chain is unbroken.
4. To unforget, search ``engram_tombstones``, find the entry, and
   re-``remember`` with the preserved text.

This is deliberately lower-tech than a "soft-delete flag on the
original row" because vstash doesn't support in-place metadata
mutation. The two-collection split (``engram_audit`` for every
decision, ``engram_tombstones`` for the forgotten material)
works within the vstash public API with no new verbs.

Two deciders ship in v1:

- ``NeverForget`` — the safe default. Engram never tombstones
  automatically. Forgetting is always an explicit user action
  (``mem.forget(force=True)``).
- ``ForgetConsolidated`` — the "cheap compression" policy. An
  episodic event whose path appears in the ``derived_from`` of
  at least ``min_facts`` semantic facts is tombstoned. One fact
  is enough by default because consolidation already required
  ``min_cluster=2`` corroborating events to fire.

LLM-aware forgetting (e.g. "don't forget if the event is still
being cited by recent queries") lives in a later phase, gated on
a scenario that exposes the value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ForgetContext:
    """Context passed to a ``should_forget`` decider.

    Intentionally small. Later phases can add fields like
    ``last_recall_at`` or ``access_count`` without churning the
    Protocol.
    """

    project: str
    derived_in_facts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ForgetDecision:
    """The output of a ``should_forget`` policy."""

    tombstone: bool
    reason: str
    confidence: float
    policy: str


class ForgetDecider(Protocol):
    """Protocol for ``should_forget`` policies."""

    name: str

    def decide(
        self,
        event_path: str,
        event_text: str,
        ctx: ForgetContext,
    ) -> ForgetDecision: ...


class NeverForget:
    """Baseline: never tombstone automatically.

    This is the safe default because an accidentally-forgotten
    event is an information loss even with the tombstone backup.
    Users who want auto-forget opt in explicitly with
    ``ForgetConsolidated`` or their own decider.
    """

    name = "NeverForget"

    def decide(
        self,
        event_path: str,  # noqa: ARG002
        event_text: str,  # noqa: ARG002
        ctx: ForgetContext,  # noqa: ARG002
    ) -> ForgetDecision:
        return ForgetDecision(
            tombstone=False,
            reason="never_auto",
            confidence=1.0,
            policy=self.name,
        )


class ForgetConsolidated:
    """Tombstone episodic events already preserved in semantic facts.

    An event's path is tombstoneable iff it appears in the
    ``derived_from`` of at least ``min_facts`` semantic facts.
    ``min_facts=1`` is the default because consolidation's own
    ``min_cluster=2`` requirement already enforces "at least two
    events had to corroborate this" before a fact is written —
    adding another round of N-of-M here would be double-counting.

    Parameters
    ----------
    min_facts:
        Minimum number of semantic facts that must cite this
        event in their ``derived_from`` before it becomes
        forgettable. Default ``1``.
    """

    name = "ForgetConsolidated"

    def __init__(self, *, min_facts: int = 1) -> None:
        self.min_facts = min_facts

    def decide(
        self,
        event_path: str,  # noqa: ARG002
        event_text: str,  # noqa: ARG002
        ctx: ForgetContext,
    ) -> ForgetDecision:
        n = len(ctx.derived_in_facts)
        if n < self.min_facts:
            return ForgetDecision(
                tombstone=False,
                reason=f"not_consolidated:{n}<{self.min_facts}",
                confidence=1.0,
                policy=self.name,
            )
        return ForgetDecision(
            tombstone=True,
            reason=f"consolidated_in_{n}_facts",
            confidence=1.0,
            policy=self.name,
        )
