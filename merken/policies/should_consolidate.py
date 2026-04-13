"""``should_consolidate`` policies — decision primitive #2.

Decides *when* consolidation runs. Separate from ``consolidate.py``
which decides *how*. Same split as the write path:
``should_remember`` says yes/no, the actual write is a different
concern.

Two implementations land in Phase 2:

- ``NeverConsolidate`` — consolidation only runs when the user passes
  ``force=True``. Useful as the default for code paths that want
  explicit control.
- ``PeriodicConsolidator`` — runs when there are at least ``min_events``
  episodic events to look at. No wall-clock trigger; consolidation is
  a pull operation in v1.

LLM-aware deciders (e.g. "consolidate when the episodic layer contains
unresolved contradictions") live in a later phase, gated on a scenario
benchmark in ``experiments/loop_quality/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ConsolidateContext:
    """Context passed to a ``should_consolidate`` decider.

    Kept minimal on purpose. Later phases can add an ``unresolved_count``
    or ``last_consolidated_at`` field without churning the Protocol.
    """

    project: str


@dataclass(frozen=True)
class ConsolidationDecision:
    """The output of a ``should_consolidate`` policy."""

    proceed: bool
    reason: str
    policy: str


class ConsolidateDecider(Protocol):
    """Protocol for ``should_consolidate`` policies."""

    name: str

    def decide(
        self,
        n_events: int,
        ctx: ConsolidateContext,
    ) -> ConsolidationDecision: ...


class NeverConsolidate:
    """Baseline: never run on its own, only when forced.

    This is the safe default — makes consolidation an explicit user
    action, never a surprise. Matches the CONSTITUTION §4.2 glass-box
    principle: the user's audit log shows "consolidate requested"
    every time the pipeline runs, because the user asked for it.
    """

    name = "NeverConsolidate"

    def decide(
        self,
        n_events: int,  # noqa: ARG002
        ctx: ConsolidateContext,  # noqa: ARG002
    ) -> ConsolidationDecision:
        return ConsolidationDecision(
            proceed=False,
            reason="never_triggers_auto",
            policy=self.name,
        )


class PeriodicConsolidator:
    """Run consolidation when enough episodic events have accumulated.

    Parameters
    ----------
    min_events:
        Minimum episodic count before the decider says yes. Below this
        the decision is ``too_few_events:<N<min>``. Above or equal,
        ``enough_events:<N>``.
    """

    name = "PeriodicConsolidator"

    def __init__(self, *, min_events: int = 10) -> None:
        self.min_events = min_events

    def decide(
        self,
        n_events: int,
        ctx: ConsolidateContext,  # noqa: ARG002
    ) -> ConsolidationDecision:
        if n_events < self.min_events:
            return ConsolidationDecision(
                proceed=False,
                reason=f"too_few_events:{n_events}<{self.min_events}",
                policy=self.name,
            )
        return ConsolidationDecision(
            proceed=True,
            reason=f"enough_events:{n_events}",
            policy=self.name,
        )
