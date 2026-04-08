"""Shared types for engram decision policies.

Plain dataclasses + a ``Protocol`` for the decider interface. No Pydantic at
this layer (CONSTITUTION §10 #6 — plain Python functions until a real reuse
case appears).
"""

from __future__ import annotations

from collections.abc import Callable
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

    Kept deliberately small. The ``recall`` callable is the policy's only
    way to peek at existing memory — it never touches the vstash instance
    directly, which keeps the policies pure and the boundary clean.
    """

    project: str
    recall: Callable[[str, int, str | None], list[Any]]


class WriteDecider(Protocol):
    """Protocol for ``should_remember`` policies."""

    name: str

    def decide(self, event: Event, ctx: WriteContext) -> Decision: ...
