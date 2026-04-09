# Extending engram — write your own decider

Every decision primitive in engram is a **Protocol**. Extending
engram means implementing the Protocol and passing your
instance via the `Memory` constructor. You don't subclass,
you don't touch the existing deciders, you don't modify
`Memory.py`.

This doc walks through writing a custom decider for each of
the four primitives, with working code examples and test
patterns.

## The general pattern

1. **Import the Protocol and the Decision type** for the
   primitive you're extending.
2. **Write a class** with a `name` attribute and a `decide()`
   method that matches the Protocol.
3. **Construct an instance** and pass it to `Memory` via the
   matching `*_decider` kwarg.
4. **Write unit tests** using fake contexts so the decider
   can be exercised without vstash.
5. **Validate on a loop-quality scenario** before committing
   it as engram's default.

If your decider needs state (a cache, a counter, a model), the
state lives on the instance. Each `Memory` gets its own
decider instance, so state is per-Memory-lifetime.

## 1. Write a custom `should_remember` decider

### The Protocol

```python
# engram/policies/types.py
class WriteDecider(Protocol):
    name: str
    def decide(self, event: Event, ctx: WriteContext) -> Decision: ...
```

Where:

```python
@dataclass
class Event:
    text: str
    layer: str = "episodic"
    title: str | None = None
    tags: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class WriteContext:
    project: str
    recall: Callable[[str, int, str | None], list[Any]]

@dataclass(frozen=True)
class Decision:
    write: bool
    reason: str
    confidence: float
    policy: str
```

### Example: content-type prior decider

Suppose you tag events with a `type:<kind>` tag, and you want
to accept "decision"-type events aggressively but reject
"ambient_chat" events unless they're novel.

```python
# my_deciders.py
from engram.policies.types import Decision, Event, WriteContext
from engram.policies.should_remember import HeuristicWriteDecider


class ContentTypePriorDecider:
    """Extends HeuristicWriteDecider with an A-MAC-style content
    type prior. Gates acceptance on a tag-based prior before
    falling through to the heuristic rules.

    Expected tag format: `type:<kind>`, e.g. `type:decision`.
    Events without a type tag get the default prior (1.0 — pass
    through to the heuristic rules unchanged).
    """

    name = "ContentTypePriorDecider"

    _DEFAULT_PRIORS = {
        "decision": 1.0,       # always pass through
        "fact": 0.9,
        "observation": 0.7,
        "question": 0.6,
        "ambient_chat": 0.2,
        "tool_echo": 0.1,
    }

    def __init__(self, *, priors: dict[str, float] | None = None):
        self._priors = priors or dict(self._DEFAULT_PRIORS)
        self._heuristic = HeuristicWriteDecider()

    def decide(self, event: Event, ctx: WriteContext) -> Decision:
        # Extract type tag
        content_type = self._extract_type(event.tags)
        prior = self._priors.get(content_type, 1.0)

        # Very low prior: skip without running the heuristic
        if prior < 0.25:
            return Decision(
                write=False,
                reason=f"low_prior:{content_type}:{prior:.2f}",
                confidence=1.0 - prior,
                policy=self.name,
            )

        # Otherwise delegate to the heuristic decider
        base = self._heuristic.decide(event, ctx)

        # If the heuristic decided to write, fold the prior into confidence
        if base.write:
            return Decision(
                write=True,
                reason=f"novel+prior:{content_type}:{prior:.2f}",
                confidence=base.confidence * prior,
                policy=self.name,
            )
        return base

    def _extract_type(self, tags: str | None) -> str:
        if not tags:
            return "unknown"
        for tag in tags.split(","):
            tag = tag.strip()
            if tag.startswith("type:"):
                return tag.split(":", 1)[1]
        return "unknown"
```

### Using it

```python
from engram import Memory
from my_deciders import ContentTypePriorDecider

with Memory(
    project="my_agent",
    write_decider=ContentTypePriorDecider(),
) as mem:
    # High-prior: always written
    mem.remember("We decided to use Postgres 16", tags="type:decision")

    # Low-prior: skipped silently
    mem.remember("ok thanks", tags="type:ambient_chat")

    # No type tag: falls through to heuristic (default prior = 1.0)
    mem.remember("a long enough observation about the codebase")
```

### Custom priors per project

```python
mem = Memory(
    project="medlocal",
    write_decider=ContentTypePriorDecider(priors={
        "diagnosis": 1.0,       # keep everything medical
        "demo_case": 1.0,
        "ambient_chat": 0.1,    # even more aggressive skip
        "tool_echo": 0.05,
    }),
)
```

### Testing your custom write decider

```python
# tests/test_my_deciders.py
from engram.policies.types import Event, WriteContext
from my_deciders import ContentTypePriorDecider


def _ctx() -> WriteContext:
    return WriteContext(project="unit", recall=lambda q, k, l: [])


def test_decision_tag_writes_novel() -> None:
    d = ContentTypePriorDecider()
    result = d.decide(
        Event(
            text="we chose Postgres 16 for the analytics warehouse",
            tags="type:decision",
        ),
        _ctx(),
    )
    assert result.write
    assert "novel+prior" in result.reason
    assert "decision" in result.reason


def test_ambient_tag_skipped_without_heuristic() -> None:
    d = ContentTypePriorDecider()
    result = d.decide(
        Event(
            text="ok thanks for the explanation about postgres",
            tags="type:ambient_chat",
        ),
        _ctx(),
    )
    assert not result.write
    assert "low_prior" in result.reason
    assert "ambient_chat" in result.reason


def test_no_tag_falls_through_to_heuristic() -> None:
    """An event without a type tag should hit the heuristic
    rules unchanged — no prior is applied."""
    d = ContentTypePriorDecider()
    result = d.decide(
        Event(text="a sufficiently long event with no type tag"),
        _ctx(),
    )
    assert result.write
    assert "novel" in result.reason


def test_custom_priors_override_defaults() -> None:
    d = ContentTypePriorDecider(priors={"banter": 0.05})
    result = d.decide(
        Event(text="something said in passing", tags="type:banter"),
        _ctx(),
    )
    assert not result.write
    assert "low_prior:banter" in result.reason
```

Run with:

```bash
pytest tests/test_my_deciders.py -v
```

### Validating on a loop-quality scenario

Before committing your decider as the default for engram,
verify it doesn't regress the scenarios:

```bash
# Point the runner at a specific scenario
python -m experiments.loop_quality.runner \
    --scenario experiments/loop_quality/scenarios/jay_vstash_2026_04_09_snapshot.json
```

If the scenario's pass_rate or purity drops, the decider isn't
strictly better — it's a trade-off, and the trade-off has to
be documented in the commit message. See
[`primitives.md`](primitives.md) for the rules engram's own
deciders were held to.

## 2. Custom `should_recall`

### The Protocol

```python
class RecallDecider(Protocol):
    name: str
    def decide(self, query: str, ctx: RecallContext) -> RecallPlan: ...
```

Where:

```python
@dataclass
class RecallContext:
    project: str
    top_k: int

@dataclass(frozen=True)
class LayerRequest:
    layer: str
    top_k: int

@dataclass(frozen=True)
class RecallPlan:
    layers: list[LayerRequest]
    reason: str
    policy: str
```

### Example: query-type-aware router

```python
from engram.policies.should_recall import (
    LayerRequest,
    RecallContext,
    RecallPlan,
)


class QueryTypeRouter:
    """Routes entity/specific queries to episodic first, theme
    queries to semantic first. A tiny regex-based classifier,
    no LLM.

    'What did X say about Y?' — entity query, episodic first
    'Summarize our decisions' — theme query, semantic first
    """

    name = "QueryTypeRouter"

    # Naive classifier: if the query mentions proper nouns or
    # asks about a specific time / person, it's an entity
    # query. Otherwise theme.
    _ENTITY_SIGNALS = (
        "who ", "when ", "where ", "last ", "yesterday", "earlier",
        "on 2026", "said", "decided", "meeting",
    )

    def __init__(self, *, top_k_per_layer: int = 5):
        self.top_k_per_layer = top_k_per_layer

    def decide(self, query: str, ctx: RecallContext) -> RecallPlan:
        is_entity = self._classify(query)

        if is_entity:
            order = [("episodic", self.top_k_per_layer),
                     ("semantic", self.top_k_per_layer)]
            reason = "entity_first_episodic"
        else:
            order = [("semantic", self.top_k_per_layer),
                     ("episodic", self.top_k_per_layer)]
            reason = "theme_first_semantic"

        return RecallPlan(
            layers=[LayerRequest(layer=l, top_k=k) for l, k in order],
            reason=reason,
            policy=self.name,
        )

    def _classify(self, query: str) -> bool:
        q = query.lower()
        return any(sig in q for sig in self._ENTITY_SIGNALS)
```

### Using it

```python
from engram import Memory
from my_deciders import QueryTypeRouter

with Memory(
    project="my_agent",
    recall_decider=QueryTypeRouter(top_k_per_layer=4),
) as mem:
    # Classifies as entity query → episodic first
    mem.recall("what did the user say on Monday?")

    # Classifies as theme query → semantic first
    mem.recall("summarize the architecture decisions")
```

### Testing

```python
def test_entity_query_routes_episodic_first() -> None:
    d = QueryTypeRouter()
    plan = d.decide(
        "what did we decide yesterday?",
        RecallContext(project="unit", top_k=5),
    )
    assert plan.layers[0].layer == "episodic"
    assert plan.reason == "entity_first_episodic"


def test_theme_query_routes_semantic_first() -> None:
    d = QueryTypeRouter()
    plan = d.decide(
        "summarize everything about databases",
        RecallContext(project="unit", top_k=5),
    )
    assert plan.layers[0].layer == "semantic"
    assert plan.reason == "theme_first_semantic"
```

## 3. Custom `should_consolidate`

### The Protocol

```python
class ConsolidateDecider(Protocol):
    name: str
    def decide(
        self,
        n_events: int,
        ctx: ConsolidateContext,
    ) -> ConsolidationDecision: ...
```

### Example: time-based consolidator

```python
import time
from engram.policies.should_consolidate import (
    ConsolidateContext,
    ConsolidationDecision,
)


class TimeBasedConsolidator:
    """Fire consolidation when either N events accumulate OR
    T seconds have passed since the last run."""

    name = "TimeBasedConsolidator"

    def __init__(
        self,
        *,
        min_events: int = 10,
        min_seconds_since_last: int = 3600,
    ):
        self.min_events = min_events
        self.min_seconds = min_seconds_since_last
        self._last_run_at = 0.0

    def decide(
        self,
        n_events: int,
        ctx: ConsolidateContext,
    ) -> ConsolidationDecision:
        now = time.time()
        elapsed = now - self._last_run_at

        if n_events >= self.min_events:
            self._last_run_at = now
            return ConsolidationDecision(
                proceed=True,
                reason=f"enough_events:{n_events}",
                policy=self.name,
            )

        if elapsed >= self.min_seconds and n_events > 0:
            self._last_run_at = now
            return ConsolidationDecision(
                proceed=True,
                reason=f"time_elapsed:{elapsed:.0f}s_n={n_events}",
                policy=self.name,
            )

        return ConsolidationDecision(
            proceed=False,
            reason=f"skip:n={n_events}_elapsed={elapsed:.0f}s",
            policy=self.name,
        )
```

**Gotcha:** the decider's state (`_last_run_at`) is per-instance
and reset on `Memory` construction. If you want
cross-invocation persistence (like `HeuristicWriteDecider`'s
hydration), you need to pull state from somewhere vstash-side
yourself. There's no built-in hydration hook for
`ConsolidateDecider` or `ForgetDecider` in v1 — only
`should_remember` has `set_hydrate_fn`.

## 4. Custom `should_forget`

### The Protocol

```python
class ForgetDecider(Protocol):
    name: str
    def decide(
        self,
        event_path: str,
        event_text: str,
        ctx: ForgetContext,
    ) -> ForgetDecision: ...
```

### Example: age-based forgetter

```python
from datetime import datetime, timezone
from engram.policies.should_forget import ForgetContext, ForgetDecision


class AgeBasedForget:
    """Tombstone events older than max_age_days that have also
    been consolidated into at least one fact.

    Combines temporal decay with coverage — strictly more
    conservative than ForgetConsolidated alone.
    """

    name = "AgeBasedForget"

    def __init__(
        self,
        *,
        max_age_days: int = 30,
        min_facts: int = 1,
    ):
        self.max_age_days = max_age_days
        self.min_facts = min_facts

    def decide(
        self,
        event_path: str,
        event_text: str,
        ctx: ForgetContext,
    ) -> ForgetDecision:
        # Gate 1: must be consolidated
        if len(ctx.derived_in_facts) < self.min_facts:
            return ForgetDecision(
                tombstone=False,
                reason=f"not_consolidated:{len(ctx.derived_in_facts)}<{self.min_facts}",
                confidence=1.0,
                policy=self.name,
            )

        # Gate 2: must be old enough
        # (engram doesn't pass added_at to ForgetContext in v1,
        #  so we'd need to extract from the path or pass it
        #  via ctx in a future version)
        # For now, this gate is a placeholder demonstrating the
        # extension pattern.

        return ForgetDecision(
            tombstone=True,
            reason=f"old_and_consolidated:n_facts={len(ctx.derived_in_facts)}",
            confidence=1.0,
            policy=self.name,
        )
```

**Note on `ForgetContext`.** As of v1, the context only exposes
`project` and `derived_in_facts`. If your decider needs
additional signal like `added_at` or `access_count`, that's a
case for extending `ForgetContext` (a contribution to engram
itself, not just your custom decider).

## Per-decider state management

Different primitives have different state needs. Here's the
current state management model:

| Primitive | State lives where | Cross-instance? |
|---|---|---|
| `should_remember` | `self._seen` on the decider instance, hydrated lazily from vstash via `hydrate_fn` on first `decide()` | Yes — every new Memory rehydrates |
| `should_recall` | Stateless by default. A custom decider can hold state on the instance. | No — per-Memory-lifetime |
| `should_consolidate` | Stateless by default. | No |
| `should_forget` | Stateless by default. | No |

If you need cross-invocation state for a non-remember decider,
the options are:

1. **Store it in vstash yourself** under a dedicated collection
   (e.g. `my_decider_state`). Look it up at the start of
   `decide()`.
2. **Use the audit log** — your previous decisions are already
   persistent there. You can query them via `Memory.audit()`
   from inside your decider if you hold a reference to the
   Memory.
3. **Hold Memory via context** — extend the context dataclass
   (in your own code, not in engram core) to include a
   callable that opens the parent Memory, and use it to query.

For v1 all four primitives in engram core are either stateless
or use `should_remember`'s `hydrate_fn` pattern. Custom
extensions are welcome to do more.

## Testing your custom deciders

The test patterns are:

### Pure unit tests (no vstash)

Fake the context object (pass a lambda for `recall`, an empty
list for `derived_in_facts`, etc.) and assert on the returned
decision. Fastest feedback loop.

### Integration tests (with vstash, tmp_path DB)

Construct `Memory` with your decider, call the real methods,
and assert on behavior. Slower but catches wiring bugs.

```python
from pathlib import Path
from engram import Memory
from my_deciders import ContentTypePriorDecider


def test_content_type_prior_integrated(tmp_path: Path) -> None:
    with Memory(
        project="integration_test",
        db=tmp_path / "e.db",
        write_decider=ContentTypePriorDecider(),
    ) as mem:
        r1 = mem.remember(
            "we decided postgres 16 for the analytics warehouse",
            tags="type:decision",
        )
        r2 = mem.remember(
            "ok thanks for the explanation about postgres",
            tags="type:ambient_chat",
        )

    assert r1.written
    assert not r2.written
    assert "low_prior" in r2.decision.reason
```

### Loop-quality scenario runs

The strictest bar. Run your decider against a real-content
scenario and compare pass_rate / purity / coverage to the
baseline (engram's default decider).

```bash
# Baseline: default decider
python -m experiments.loop_quality.runner

# With your decider: (requires modifying the runner to
# accept --write-decider dotted.path.to.ContentTypePriorDecider,
# which is a future addition)
```

For now, the runner uses the default deciders hardcoded. To
test a custom decider at scenario level, write a small driver
script that loads a scenario, constructs a `Memory` with your
decider, and reports the same metrics. See
`experiments/loop_quality/runner.py:run_scenario` for the
reference implementation.

## Design principles for custom deciders

Adapted from the patterns the four default deciders follow:

### Keep decide() pure

`decide()` should be a function of `(input, ctx)` with minimal
side effects. If you need to update state (like
`HeuristicWriteDecider` updating `_seen`), do it explicitly
and document it in the docstring.

### Never call vstash from decide()

Decision primitives should not trigger vstash searches at
decision time. If you need similarity information, pre-compute
it (via `hydrate_fn` for writes) or accept the limitation. A
decider that does O(1) in the hot path is almost always
better than one that does O(log N) by consulting vstash.

### Reason strings should be grep-able

Engram's audit log is queryable via vstash hybrid search. A
good `reason` string is:

- **Machine-parseable** — e.g. `too_short:<8`, not "text is
  too short"
- **Uniquely grep-able** — e.g. `low_prior:ambient_chat:0.20`,
  not just "skipped"
- **Informative without the source** — a reader of
  `engram audit low_prior` should understand what happened
  without reading the decider code

### Fail open, unless it's a safety-critical write

`HeuristicWriteDecider` swallows hydration failures — if vstash
is temporarily unreachable, dedup degrades but writes keep
happening. This is fail-open and it's correct for decision
primitives.

The one exception is `Memory.forget()` — it raises if the
tombstone write fails, because we must not remove the original
without a backup. That's fail-closed and also correct.

For your custom decider, default to fail-open. Make
exceptions explicit.

### Version your `name`

If you iterate on a decider's logic, append a version
suffix to the `name` attribute:

```python
class ContentTypePriorDecider:
    name = "ContentTypePriorDecider_v2"
```

The audit log preserves the `policy` field — in a year's time,
you'll be able to see which version of your decider made which
decisions and correlate with behavior changes.

## Extending engram core vs extending in your own code

**Extend in your own code** if:

- The custom decider is specific to your agent / project
- It depends on state or context outside of engram's general
  model
- You want to experiment before upstreaming

**Contribute to engram core** if:

- The decider represents a general-interest improvement (e.g.
  ContentTypePrior, TemporalRecaller, EbbinghausDecayForget
  — all discussed in [`../notes/research-2026-04-09.md`](../notes/research-2026-04-09.md))
- It comes with a loop-quality scenario that demonstrates the
  value
- It doesn't regress any existing scenario
- The `decide()` function is pure and testable

The general-interest bar is high on purpose — engram aims for
a small, stable default set of deciders, with extensions living
in user code until empirically justified.

## Further reading

- [`primitives.md`](primitives.md) — the four default deciders
  in depth, with audit formats and trade-offs
- [`architecture.md`](architecture.md) — the memory model your
  custom decider will run against
- [`../notes/research-2026-04-09.md`](../notes/research-2026-04-09.md)
  — 6 papers with concrete extension ideas (ContentTypePrior
  from A-MAC, TemporalRecaller from CMA, EbbinghausDecay from
  SuperLocalMemory)
- `engram/policies/*.py` — the Protocol definitions and
  reference implementations
- `tests/test_should_*.py` — reference test patterns
