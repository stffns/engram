# The four decision primitives

engram's central idea is that every memory operation is a
**structured decision** with explicit inputs, explicit outputs,
and an audit row. This doc covers the four primitives in depth:
what they decide, how the default implementations work, when
they fire, and how to read their audit rows.

## Shared shape

Every primitive follows the same pattern:

- A **Protocol** that defines `decide(input, ctx) -> Decision`
- At least two **concrete implementations**: a baseline that
  always says the same thing (useful as a control) and a
  default with real logic
- A **Decision** dataclass with at minimum `reason`, `policy`,
  and `confidence` fields (plus primitive-specific fields)
- An **audit row** written for every call, whether the action
  fires or not

All primitives live in `engram/policies/`. Types are in
`engram/policies/types.py`.

## 1. `should_remember`

**Protocol:** `WriteDecider` in `engram/policies/types.py`.

**Job:** Decide whether an incoming event merits a vstash write.

### Implementations

#### `AlwaysWrite` (baseline)

```python
class AlwaysWrite:
    name = "AlwaysWrite"

    def decide(self, event, ctx):
        return Decision(
            write=True,
            reason="always_write",
            confidence=1.0,
            policy=self.name,
        )
```

Every event gets written. Useful as a control in experiments:
does engram's filtering actually help, or is it costing us
information we'd rather keep?

#### `HeuristicWriteDecider` (default)

Rules, in order:

1. **Empty** — text is whitespace-only → skip with
   `reason="empty"`
2. **Too short** — text < `min_chars` (default 8) → skip with
   `reason="too_short:<8"`
3. **Too long** — text > `max_chars` (default 100,000) → skip
   with `reason="too_long:>100000"`
4. **Exact duplicate** — whitespace-normalized text already in
   the in-process `_seen` set → skip with `reason="dup_exact"`
5. **Novel** — passes all checks → write with `reason="novel"`

**Why exact-match dedup and not embedding similarity?** Recall-
based dedup at write time turned out to be O(N²) per haystack on
real data — each write triggered a vstash.search against an
expanding index. Exact hash lookup in a Python set is O(1) and
is what the rule actually wants. Embedding-similarity novelty
lives in consolidation, where it belongs.

**Cross-invocation dedup.** The `_seen` set is per-instance and
starts empty on Memory construction. To make dedup work across
CLI invocations, `Memory` hands the decider a `hydrate_fn` that
lazily pulls every existing episodic doc's text into `_seen` on
the first `decide()` call. So if you run `engram remember "..."`
twice with the same text, the second call sees it as a
duplicate.

You can override either param at construction:

```python
from engram import HeuristicWriteDecider, Memory

mem = Memory(
    project="my_agent",
    write_decider=HeuristicWriteDecider(min_chars=20, max_chars=5000),
)
```

### Post-decision: vstash rejection

engram's decision is not the final word. vstash has its own
guardrail — it silently rejects text shorter than ~20 chars
with `IngestResult(status="empty", chunks=0, chars=0)`, no
exception, no error field.

`Memory.remember()` catches this and surfaces it as a
corrective decision:

```python
override = Decision(
    write=False,
    reason=f"vstash_rejected:{ingest.status}",
    confidence=1.0,
    policy=decision.policy,
)
```

Two audit rows get written in this case: the original
`novel` decision from the heuristic decider, plus the
`vstash_rejected:empty` override. `RememberResult.written`
reflects the final state — if it's `True`, an actual write
happened.

### Audit format

```
timestamp: 2026-04-09T11:04:33+00:00
decision: should_remember
write: True
reason: novel
policy: HeuristicWriteDecider
confidence: 0.8
event_layer: episodic
event_title: 
event_tags: 
event_text_preview: the user switched to Postgres on 2026-04-08
```

Query via `engram audit should_remember` or
`mem.audit(query="should_remember")`.

## 2. `should_recall`

**Protocol:** `RecallDecider` in `engram/policies/should_recall.py`.

**Job:** Decide which layers to query and with what top_k budget.

### Types

```python
@dataclass(frozen=True)
class LayerRequest:
    layer: str
    top_k: int

@dataclass(frozen=True)
class RecallPlan:
    layers: list[LayerRequest]
    reason: str
    policy: str

@dataclass
class RecallContext:
    project: str
    top_k: int  # the caller's requested top_k
```

### Implementations

#### `SemanticOnlyRecaller` (baseline)

```python
class SemanticOnlyRecaller:
    name = "SemanticOnlyRecaller"

    def decide(self, query, ctx):
        return RecallPlan(
            layers=[LayerRequest(layer="semantic", top_k=ctx.top_k)],
            reason="semantic_only",
            policy=self.name,
        )
```

Only queries the semantic layer. Useful as a control: how much
does the episodic fallback contribute to recall success?

#### `LayeredRecaller` (default)

```python
class LayeredRecaller:
    name = "LayeredRecaller"

    def __init__(self, *, top_k_semantic=5, top_k_episodic=3):
        self.top_k_semantic = top_k_semantic
        self.top_k_episodic = top_k_episodic

    def decide(self, query, ctx):
        return RecallPlan(
            layers=[
                LayerRequest(layer="semantic", top_k=self.top_k_semantic),
                LayerRequest(layer="episodic", top_k=self.top_k_episodic),
            ],
            reason=f"layered_sem={self.top_k_semantic}_epi={self.top_k_episodic}",
            policy=self.name,
        )
```

Queries semantic first, then episodic as a fallback. Both layers
always run — `Memory.recall` does the actual interleave (see
below).

### Post-decision: fetch-then-interleave

`Memory.recall` does **not** drain layers sequentially. Earlier
it did, and the result was that when semantic returned ≥ top_k
hits, episodic was never consulted. A real-content smoke test
caught the bug — the query "what happened in the Kafka merchant
pipeline meeting?" never returned the Kafka singleton because
semantic facts filled the budget first.

The fix is round-robin interleave:

```python
# Fetch all layers first
per_layer_hits = [
    self._vstash.search(query, top_k=req.top_k, layer=req.layer, ...)
    for req in plan.layers
]

# Interleave round-robin, dedup by path
seen_paths = set()
merged = []
max_len = max(len(hs) for hs in per_layer_hits)
for i in range(max_len):
    for layer_hits in per_layer_hits:
        if i >= len(layer_hits):
            continue
        h = layer_hits[i]
        path = getattr(h, "path", None)
        if path in seen_paths:
            continue
        seen_paths.add(path)
        merged.append(h)
        if len(merged) >= top_k:
            return merged

return merged[:top_k]
```

Every layer in the plan is guaranteed at least one slot in the
final list (until the user's `top_k` runs out), which is what
"layered recall" was supposed to mean from the start.

See [`../notes/silt.md`](../notes/silt.md) for the commit chain
behind this fix.

### Explicit layer bypass

If the caller passes `layer=` explicitly, the decider is
**bypassed** and the call is a direct `vstash.search`. This is
the escape hatch for benchmarks and for callers who already know
which layer they want. Explicit-layer calls do NOT write a
`should_recall` audit row — they're raw passthrough.

```python
# Goes through the decider, writes audit:
mem.recall("kafka meeting")

# Bypasses decider, no audit row:
mem.recall("kafka meeting", layer="episodic")
```

### Audit format

```
timestamp: 2026-04-09T11:05:12+00:00
decision: should_recall
query_preview: what happened in the Kafka merchant pipeline meeting?
plan_layers: semantic:5,episodic:3
reason: layered_sem=5_epi=3
policy: LayeredRecaller
```

## 3. `should_consolidate`

**Protocol:** `ConsolidateDecider` in
`engram/policies/should_consolidate.py`.

**Job:** Decide whether it's time to distill episodic events
into semantic facts.

### Types

```python
@dataclass(frozen=True)
class ConsolidationDecision:
    proceed: bool
    reason: str
    policy: str

@dataclass
class ConsolidateContext:
    project: str
```

### Implementations

#### `NeverConsolidate` (baseline)

Always says no. Consolidation only runs on `mem.consolidate(force=True)`.

#### `PeriodicConsolidator` (default)

```python
class PeriodicConsolidator:
    name = "PeriodicConsolidator"

    def __init__(self, *, min_events=10):
        self.min_events = min_events

    def decide(self, n_events, ctx):
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
```

Fires when at least `min_events` episodic events exist in the
store. Simple count-based trigger. No time-based trigger in v1.

### Post-decision: the consolidation pipeline

If the decider says proceed (or the caller passed `force=True`),
`Memory.consolidate` runs this:

1. **List episodic docs** from the engram collection.
2. **Reassemble text** from each doc's chunks.
3. **Resolve the embedder model** from vstash's `store_meta`
   (falling back to config default).
4. **Cluster** using `cluster_by_embedding`:
   - Raw cosine matrix, O(N² × dim)
   - Complete linkage agglomerative: merge two clusters only
     when every cross-pair is above threshold
   - Default threshold: **0.70** (picked via grid search)
5. For each cluster ≥ `min_cluster` (default 2):
   - **Materialize** into a `Fact` via `materialize_fact`
   - **Fingerprint** via `fact_fingerprint` (stable SHA-1 of
     sorted `derived_from`)
   - **Write** to semantic layer with
     `title=f"fact_{fingerprint}"` — identity is stable so
     re-running consolidation is idempotent
   - **Provenance** via `tags=f"derived_from:<paths>"`

### Why complete linkage, why 0.70

**Why complete linkage (not single):** single-link cascaded two
genuine-but-cross-topic edges into one impure cluster on the
`session_2026_04_09` scenario. Complete-link refuses to merge
two clusters unless every cross-pair is above threshold, which
prevents the cascade.

**Why threshold 0.70 (not 0.65):** grid search across three
scenarios, documented in
[`../experiments/loop_quality/RESULTS.md`](../experiments/loop_quality/RESULTS.md):

```
thresh   min_pass  min_purity
0.60     75%       33%
0.63     75%       75%
0.65     75%       75%   ← old default
0.68     75%       75%
0.70     100%      100%  ← new default
0.72     100%      100%
```

0.70 is the lowest threshold that maximizes both the minimum
pass rate and the minimum purity across all three scenarios.

### Alternative clustering methods

`Memory.consolidate(method=...)` accepts three values:

- `"embedding_v1"` (default) — raw cosine + complete linkage
- `"jaccard_v1"` — token-overlap Jaccard, useful only for
  near-duplicate text
- `"recall_v1"` — vstash hybrid search as neighbor hint,
  brittle on small corpora

See `engram/consolidation.py` docstrings for the full trade-offs.

### Audit format

```
timestamp: 2026-04-09T11:06:44+00:00
decision: should_consolidate
proceed: True
reason: enough_events:12
policy: PeriodicConsolidator
n_events: 12
```

## 4. `should_forget`

**Protocol:** `ForgetDecider` in `engram/policies/should_forget.py`.

**Job:** Decide whether an episodic event is safe to tombstone.

### Types

```python
@dataclass(frozen=True)
class ForgetDecision:
    tombstone: bool
    reason: str
    confidence: float
    policy: str

@dataclass
class ForgetContext:
    project: str
    derived_in_facts: list[str]
```

### Implementations

#### `NeverForget` (default — safe)

```python
class NeverForget:
    name = "NeverForget"

    def decide(self, event_path, event_text, ctx):
        return ForgetDecision(
            tombstone=False,
            reason="never_auto",
            confidence=1.0,
            policy=self.name,
        )
```

**This is the default** because an accidentally-forgotten event
is information loss even with the tombstone backup. Users who
want auto-forget opt in explicitly with `ForgetConsolidated` or
their own decider.

#### `ForgetConsolidated` (opt-in)

```python
class ForgetConsolidated:
    name = "ForgetConsolidated"

    def __init__(self, *, min_facts=1):
        self.min_facts = min_facts

    def decide(self, event_path, event_text, ctx):
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
```

Tombstones an episodic event if its path appears in at least
`min_facts` semantic facts' `derived_from` tag. `min_facts=1`
is the default because consolidation already required
`min_cluster=2` corroborating events, so one fact transitively
means two-or-more events agreed.

### Post-decision: the tombstone pipeline

For each event the decider says to forget (or if the caller
passed `force=True`):

1. **Write to `engram_tombstones`** with:
   - Full original text
   - Original title, layer, tags
   - `derived_in_facts` provenance pointer list
   - Reason, policy, timestamp
2. **Call `vstash.remove(event.path)`** to delete the original
   from the `default` collection

The tombstone is written **first**. If it fails, the function
raises — we must not remove the original without a backup.
Every other audit write is fail-open; this one is fail-closed.

**Semantic facts are not touched.** Their `derived_from` tags
still point at the tombstoned path, which is the provenance
record by design. A human can always trace a fact back to its
original event via the `engram_tombstones` collection.

### Forcing a wipe

`mem.forget(force=True)` bypasses the decider entirely and
tombstones every episodic event. Useful as a "wipe the episodic
layer after a known-good consolidation pass" operation. Every
event still gets a tombstone — nothing is destroyed, only
moved.

### Querying tombstones

```python
rows = mem.tombstones(query="kafka")
# or
```

```bash
engram tombstones kafka
```

The tombstone body starts with metadata then has a `---\n`
separator then the full original text. Human and JSON formats
both expose this.

### Audit format

The forget decider writes a *decision* audit row (short,
metadata only), and the tombstone itself is the *text-preserving*
record in a separate collection.

Decision row:

```
timestamp: 2026-04-09T11:08:19+00:00
decision: should_forget
event_path: text://the-team-chose-postgres-20260408-145533
tombstone: True
reason: consolidated_in_1_facts
policy: ForgetConsolidated
derived_in_facts: text://fact_a1b2c3d4e5f6
```

Tombstone row (in `engram_tombstones`):

```
tombstone_of: text://the-team-chose-postgres-20260408-145533
tombstoned_at: 2026-04-09T11:08:19+00:00
reason: consolidated_in_1_facts
policy: ForgetConsolidated
original_title: the-team-chose-postgres-20260408-145533
original_layer: episodic
original_tags: 
derived_in_facts: text://fact_a1b2c3d4e5f6
---
The team chose Postgres 16 for the new analytics warehouse because
of write concurrency. Decision made on 2026-04-08 during the planning
call.
```

## Cross-primitive interactions

The primitives are independent in the sense that each has its
own Protocol and its own state, but they interact through the
memory model:

- **remember → consolidate.** Every remember that lands creates
  an episodic event. Consolidate reads the episodic layer and
  writes the semantic layer. A broken `should_remember` (e.g.
  accepts everything) makes consolidate work harder.
- **consolidate → forget.** `ForgetConsolidated` reads the
  `derived_from` tags that consolidate writes. Without
  consolidation, `ForgetConsolidated` never finds a reason to
  tombstone anything.
- **consolidate → recall.** Recall's `LayeredRecaller` queries
  semantic first. Without consolidate producing facts, recall
  collapses to episodic-only (which is fine on small stores,
  noisy on big ones).
- **recall → debug audit.** The `audit` method is itself a
  recall on the `engram_audit` collection. Every time you want
  to know *why* a decision happened, you're using recall to
  find it.

**None of the primitives currently have cross-primitive
awareness** — `should_remember` does not know whether the event
will eventually be consolidated, `should_forget` does not know
whether the caller is about to recall. A v2 that gives
primitives access to each other's history (via the audit log
or a shared context object) is a possible direction but is
gated on a scenario that shows the current independence causes
a real problem.

## Writing your own decider

Every primitive is a Protocol, so extending engram means
implementing the Protocol and passing your instance via the
`Memory` constructor. See [`extending.md`](extending.md) for
the full guide.

## Further reading

- [`architecture.md`](architecture.md) — how the primitives fit
  into the full memory model
- [`../notes/research-2026-04-09.md`](../notes/research-2026-04-09.md)
  — 6 papers surveyed, candidate extensions (ContentTypePrior,
  TemporalRecaller, EbbinghausDecayForget)
- [`../notes/silt.md`](../notes/silt.md) — the four design-rule
  interventions that shaped the defaults you see here
- `engram/policies/*.py` — the source of truth
