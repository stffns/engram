"""``should_recall`` policies — decision primitive #3.

Decides *how* to route a recall across merken's memory layers. Until
this primitive exists, ``Memory.recall(query)`` is a pass-through to
``vstash.Memory.search`` with whatever layer the caller happened to
pass, and the semantic facts produced by ``consolidate()`` are
dead code unless the caller explicitly asks for them.

The shape is symmetric with ``should_remember`` and
``should_consolidate``: a Protocol, a default implementation, and a
baseline useful for experiments. No LLM in v1; LLM-aware variants
(e.g. "route technical queries to procedural") can land later under
their own scenarios in ``experiments/loop_quality/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class LayerRequest:
    """A single layer read within a ``RecallPlan``.

    ``top_k`` is the per-layer fetch budget, not the final result
    size. ``Memory.recall`` concatenates hits across requests and
    truncates to the user's requested top_k after deduplication.
    """

    layer: str
    top_k: int


@dataclass(frozen=True)
class RecallPlan:
    """The output of a ``should_recall`` policy.

    ``layers`` is a priority-ordered list: ``Memory.recall`` visits
    each request in turn and stops once the caller's top_k is
    satisfied. ``reason`` and ``policy`` exist for the audit log.
    """

    layers: list[LayerRequest] = field(default_factory=list)
    reason: str = ""
    policy: str = ""


@dataclass
class RecallContext:
    """Context passed to a ``should_recall`` decider.

    Kept small on purpose. A later phase can add ``caller_intent`` or
    ``recent_layer_hits`` without churning the Protocol.
    """

    project: str
    top_k: int


class RecallDecider(Protocol):
    """Protocol for ``should_recall`` policies."""

    name: str

    def decide(self, query: str, ctx: RecallContext) -> RecallPlan: ...


class SemanticOnlyRecaller:
    """Baseline: semantic layer only, no fallback.

    Useful as a control in ``experiments/loop_quality/`` to isolate
    the cost/benefit of fallback routing. If this decider and the
    default give the same query pass rate on a scenario, the
    fallback layer is not contributing anything for that content and
    is a waste of vstash queries.
    """

    name = "SemanticOnlyRecaller"

    def decide(self, query: str, ctx: RecallContext) -> RecallPlan:  # noqa: ARG002
        return RecallPlan(
            layers=[LayerRequest(layer="semantic", top_k=ctx.top_k)],
            reason="semantic_only",
            policy=self.name,
        )


class LayeredRecaller:
    """Default v1 policy: semantic first, episodic fallback.

    Semantic holds consolidated facts — the answers the loop has
    already corroborated. Episodic holds raw events — the original
    material the agent lived through. When a query hits semantic
    cleanly, that is the right answer. When semantic is silent or
    ambiguous, falling back to episodic lets recall use the raw
    stream as evidence even if consolidation missed a cluster.

    Both layers are queried unconditionally in v1. ``Memory.recall``
    dedupes by vstash path and truncates to the user's top_k, so
    querying the fallback when semantic is already full is cheap
    (the semantic hits dominate the final list). A smarter variant
    could skip episodic when semantic returns ≥ ``ctx.top_k`` hits —
    deferred until a loop_quality scenario shows the cost matters.

    Parameters
    ----------
    top_k_semantic, top_k_episodic:
        Per-layer fetch budgets. The defaults (5 and 3) oversample
        semantic relative to the typical user top_k of 5, giving the
        semantic hits a head start in the final ranking.
    """

    name = "LayeredRecaller"

    def __init__(
        self,
        *,
        top_k_semantic: int = 5,
        top_k_episodic: int = 3,
    ) -> None:
        self.top_k_semantic = top_k_semantic
        self.top_k_episodic = top_k_episodic

    def decide(self, query: str, ctx: RecallContext) -> RecallPlan:  # noqa: ARG002
        return RecallPlan(
            layers=[
                LayerRequest(layer="semantic", top_k=self.top_k_semantic),
                LayerRequest(layer="episodic", top_k=self.top_k_episodic),
            ],
            reason=(
                f"layered_sem={self.top_k_semantic}_epi={self.top_k_episodic}"
            ),
            policy=self.name,
        )
