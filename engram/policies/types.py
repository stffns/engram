"""Shared types for engram decision policies.

Plain dataclasses + a ``Protocol`` for the decider interface. No Pydantic at
this layer (CONSTITUTION §10 #6 — plain Python functions until a real reuse
case appears).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Event:
    """A candidate write to memory.

    Built by ``Memory.remember()`` from the user's call and handed to the
    write decider. Decision policies are pure functions over ``(Event,
    WriteContext)`` so they can be tested in isolation.
    """

    text: str
    layer: str = "episodic"
    title: str | None = None
    tags: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """The output of a decision policy.

    ``write`` is the only field engram acts on. The other fields exist for
    the audit log: future-you needs to know *why* an event was kept or
    dropped, not just whether it was. CONSTITUTION §4.2 — glass box.
    """

    write: bool
    reason: str
    confidence: float
    policy: str


@dataclass
class WriteContext:
    """Context passed to ``WriteDecider.decide``.

    Kept deliberately small. A write decider gets the project name and
    nothing else; if it needs similarity information, it should use
    hydration (see ``HeuristicWriteDecider.set_hydrate_fn``) to
    pre-populate state at ``Memory`` construction time, NOT live queries
    during ``decide()``.

    **Historical note (2026-04-09, Jay's extending.md review):** an earlier
    version exposed a ``recall`` callable here, backed by
    ``vstash.Memory.search``. That was dead weight: the default decider
    never used it (it had been removed during the O(N²) dedup fix), and
    its presence contradicted the extending.md rule "never call vstash
    from decide()". Removed entirely. If a future similarity-based
    decider needs read access to vstash at decision time, the right move
    is to re-add a clearly-named callable with an explicit docstring
    warning about cost — not to leave a vague hook around "for later."
    """

    project: str


class WriteDecider(Protocol):
    """Protocol for ``should_remember`` policies."""

    name: str

    def decide(self, event: Event, ctx: WriteContext) -> Decision: ...
