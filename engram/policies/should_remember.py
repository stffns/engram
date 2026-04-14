"""``should_remember`` policies — the first decision primitive.

Two implementations land in Phase 1:

- ``AlwaysWrite`` — the "store-everything" baseline. Useful as a control
  in ``experiments/`` runs: it tells us whether engram's filtering is
  helping or hurting on a given dataset.
- ``HeuristicWriteDecider`` — engram's default. No LLM. Skips empty,
  too-short, too-long, and exact-duplicate writes. The minimum policy that
  is honestly better than ``AlwaysWrite`` for a real loop.

LLM-backed deciders (novelty scoring, intent classification) live in a later
phase, gated on a benchmark per CONSTITUTION §9.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from engram.policies.types import Decision, Event, WriteContext


def _normalize(text: str) -> str:
    """Whitespace-collapsed lower-bound for exact-dedup comparison."""
    return " ".join(text.split())


class AlwaysWrite:
    """Baseline: every event gets written.

    Exists so that experiments can isolate the cost/benefit of engram's
    filtering against a "store everything" control. Not the engram default.
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
    4. Exact-text duplicate already seen → skip (``dup_exact``).
       Comparison is whitespace-normalized and uses an in-process set,
       NOT a recall against vstash (that dominates ingest cost at
       real-data scale, O(N²) per haystack).
    5. Otherwise → write (``novel``) and remember the normalized text.

    **State hydration (2026-04-09):** originally the decider's ``_seen``
    set was fresh on every instantiation, which meant cross-invocation
    duplicates weren't caught — a CLI user who re-piped the same chat
    log twice would get every line written twice because each CLI
    invocation constructed a new decider. Fix: ``Memory`` now hands the
    decider a ``hydrate_fn`` callable that returns the texts of every
    existing episodic doc in the collection. On the first ``decide()``
    call, the decider lazily pulls those texts into ``_seen``. Cross-
    invocation dedup now works for any caller who keeps the same DB
    between runs. If ``hydrate_fn`` is None (the default for tests and
    standalone usage), the decider is stateless across instances — old
    behavior preserved.

    The hydration is lazy (deferred to the first ``decide()``) because
    it's expensive: one ``list()`` plus N ``get_document_chunks()``
    calls against vstash. For a caller that never writes, we never
    pay the cost.
    """

    name = "HeuristicWriteDecider"

    def __init__(
        self,
        *,
        min_chars: int = 8,
        max_chars: int = 100_000,
        hydrate_fn: Callable[[], Iterable[str]] | None = None,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._seen: set[str] = set()
        self._hydrate_fn = hydrate_fn
        # ``_hydrated`` is strictly "has the hydration run yet?" — it
        # does NOT short-circuit construction with no hydrate_fn, so
        # a later ``set_hydrate_fn`` call (from Memory.__init__) can
        # still wire the decider before the first ``decide()``.
        self._hydrated = False

    def set_hydrate_fn(
        self,
        fn: Callable[[], Iterable[str]],
    ) -> None:
        """Attach a hydration callable after construction.

        Used by ``Memory.__init__`` to wire the decider to its
        surrounding vstash without requiring the caller to know about
        the dedup set. If the decider has already hydrated (the first
        ``decide()`` ran), this is a no-op — we don't retro-fill the
        set, the caller should provide hydrate_fn at construction
        time for that.
        """
        if self._hydrated:
            return
        self._hydrate_fn = fn

    def _hydrate_if_needed(self) -> None:
        if self._hydrated:
            return
        self._hydrated = True
        if self._hydrate_fn is None:
            return
        try:
            for text in self._hydrate_fn():
                if text:
                    self._seen.add(_normalize(text))
        except Exception:
            # Hydration failures must not block writes. If vstash is
            # unreachable at hydration time, the decider behaves as if
            # ``_seen`` is empty and future dup_exact checks are lost
            # — acceptable degradation.
            pass

    def decide(self, event: Event, ctx: WriteContext) -> Decision:  # noqa: ARG002
        text = event.text.strip()

        if not text:
            return Decision(False, "empty", 1.0, self.name)

        if len(text) < self.min_chars:
            return Decision(False, f"too_short:<{self.min_chars}", 1.0, self.name)

        if len(text) > self.max_chars:
            return Decision(False, f"too_long:>{self.max_chars}", 1.0, self.name)

        self._hydrate_if_needed()

        norm = _normalize(text)
        if norm in self._seen:
            return Decision(False, "dup_exact", 1.0, self.name)

        self._seen.add(norm)
        return Decision(True, "novel", 0.8, self.name)


class ContentTypePriorDecider:
    """Extends HeuristicWriteDecider with an A-MAC-style content-type prior.

    Gates acceptance on a tag-based prior before falling through to the
    heuristic rules. Events tagged with low-prior types (< 0.25) are
    rejected immediately; the rest delegate to ``HeuristicWriteDecider``
    with the prior folded into the confidence score.

    **Type resolution order:**

    1. Explicit ``type:<kind>`` tag on the event — always wins.
    2. If ``auto_classify=True`` (default) and no explicit tag,
       ``classify_content_type()`` infers the type from the text using
       regex/keyword heuristics. No LLM.
    3. If both fail, the type is ``"unknown"`` and the prior is 1.0
       (pass through unchanged).

    See ``notes/research-2026-04-09.md`` for the A-MAC paper analysis
    and ``docs/extending.md`` for usage examples.
    """

    name = "ContentTypePriorDecider"

    _DEFAULT_PRIORS: dict[str, float] = {
        "decision": 1.0,
        "fact": 0.9,
        "observation": 0.7,
        "question": 0.6,
        "status": 0.3,
        "ambient_chat": 0.2,
        "ack": 0.15,
        "tool_echo": 0.1,
    }

    def __init__(
        self,
        *,
        priors: dict[str, float] | None = None,
        low_prior_threshold: float = 0.25,
        auto_classify: bool = True,
        **heuristic_kwargs,
    ) -> None:
        self._priors = priors or dict(self._DEFAULT_PRIORS)
        self._low_prior_threshold = low_prior_threshold
        self._auto_classify = auto_classify
        self._heuristic = HeuristicWriteDecider(**heuristic_kwargs)

    def set_hydrate_fn(
        self,
        fn: Callable[[], Iterable[str]],
    ) -> None:
        """Delegate hydration to the inner heuristic decider."""
        self._heuristic.set_hydrate_fn(fn)

    def decide(self, event: Event, ctx: WriteContext) -> Decision:
        content_type = self._resolve_type(event)
        prior = self._priors.get(content_type, 1.0)

        if prior < self._low_prior_threshold:
            return Decision(
                write=False,
                reason=f"low_prior:{content_type}:{prior:.2f}",
                confidence=1.0 - prior,
                policy=self.name,
            )

        base = self._heuristic.decide(event, ctx)

        if base.write:
            return Decision(
                write=True,
                reason=f"novel+prior:{content_type}:{prior:.2f}",
                confidence=base.confidence * prior,
                policy=self.name,
            )
        return base

    def _resolve_type(self, event: Event) -> str:
        """Resolve content type: explicit tag first, then auto-classify."""
        explicit = self._extract_type(event.tags)
        if explicit != "unknown":
            return explicit
        if self._auto_classify:
            from engram.classification import classify_content_type

            return classify_content_type(event.text)
        return "unknown"

    @staticmethod
    def _extract_type(tags: str | None) -> str:
        if not tags:
            return "unknown"
        for tag in tags.split(","):
            tag = tag.strip()
            if tag.startswith("type:"):
                return tag.split(":", 1)[1]
        return "unknown"
