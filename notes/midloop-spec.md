# Midloop spec -- validation against merken codebase (2026-04-19)

This note consolidates Jay's design synthesis for extending merken
with a `NanoGPTMidloopDecider` primitive (third loop, distinct from
Builder/Judge), validated against the actual merken code. It is the
canonical reference for the midloop design until implementation
starts; once code lands the spec moves to docstrings + the relevant
HYPOTHESES.md sections.

The full design doc Jay wrote is preserved in conversation history
(`merken_recall("midloop")` post-2026-04-19). What follows is the
validation findings + the implementation plan that survives them.

## Architectural foundation: hybrid memory, not partitioned

Two memory ontologies coexist in the same storage:

- **Type A (derived):** episodic events, briefs, traces, outcomes.
  Mutable, temporal, decays, consolidatable, tombstoneable. No
  stable ground truth.
- **Type B (authoritative):** WHO/MSF protocols, fixed corpora,
  laws, technical docs. Immutable, atemporal, no decay, no
  consolidation, no arbitrary tombstone -- only versioned
  replacement.

**Critical decision (Jay corrected explicitly):** Type A and B do
NOT separate into silos. They coexist in the same storage because
real decisions integrate both ("given past experience grounded in
an immutable fact"). The distinction is metadata per event, not a
separate layer.

Operation-level differential treatment:

| operation | Type A | Type B |
|---|---|---|
| decay | yes | NO |
| consolidate -> semantic | yes | NO |
| forget/tombstone | yes | NO (only versioned replacement) |
| remember by default | yes (with decider) | only with explicit flag |
| recall (candidates) | yes | yes -- mixed |
| use in synthesis | yes | yes -- both |
| citation preference | lower | higher |

## Validation -- 6 questions answered against the code

### Q1. Schema supports adding `immutable` / `source_type` without disruptive migration?

**YES, no migration.** Three paths, recommended one in bold:

- **Tag-based** (cheapest, RECOMMENDED): vstash sqlite already has
  a `tags TEXT` column. `ContentTypePriorDecider`
  (`merken/policies/should_remember.py:368-387`) already extracts
  tags by `type:<value>` convention. Same pattern: tag events with
  `source:authoritative` (or `source:protocol_who`). Zero migration,
  all FTS queries find it.
- Metadata dict: `Event.metadata: dict[str, Any]` exists
  (`merken/policies/types.py:26-27`) but does NOT persist to vstash
  automatically -- only `tags` does. Would need serialization to
  tags anyway.
- New schema column: overkill for a boolean.

Decision: convention is `source:authoritative` tag activates Type B
behavior in mutative operations.

### Q2. WriteDecider Protocol extensible to MidloopDecider without combinator refactor?

**YES, with honest fork.** Each existing primitive has its own
Protocol + Decision shape: `WriteDecider`, `ConsolidateDecider`,
`RecallDecider`, `ForgetDecider`. Cero acoplamiento entre ellos.
`ChainedWriteDecider` and `ShadowWriteDecider`
(`should_remember.py:156-289`) hardcode the WriteDecider types --
they do NOT generalize freely.

For midloop: define `MidloopDecider` Protocol + dataclasses
(`MidloopObservation`, `MidloopDecision`, `MidloopContext`) in a new
`merken/policies/midloop.py`. Fork `ShadowMidloopDecider` from
`ShadowWriteDecider` (~30 lines). Do NOT generalize to
`Shadow[Generic[T]]` until 3+ use cases demand it.

### Q3. Audit log supports `task_id + step_index` embedded in body for FTS?

**YES, no format extension needed.** `format_audit_row`
(`audit.py:137-168`) writes plain-text key:value lines. Adding
`task_id:` and `step_index:` lines is trivial. vstash FTS over body
finds `task_id:abc123`. Confirmed limit is n<10k; for higher n you
want indexed columns. Midloop early traffic (~200 task_ids/day)
fits comfortably.

### Q4. Decay/consolidate/forget operations have central place to check `immutable`?

**MOSTLY central, ~3 lines per operation:**

- `Memory.consolidate()` `memory.py:540-543` -- single
  `for doc in docs:` loop. Filter at append.
- `Memory.forget()` `memory.py:728-836` -- multiple `for event in
  episodic` loops but all in one method. One check at the top.
- `decay` -- DOES NOT EXIST in merken today. If added, design with
  the check from day one.

Note on existing partial pattern: `Memory.consolidate()` already
reads only `layer="episodic"`. Briefs in `layer="semantic"` are
incidentally untouchable by `forget()` because forget filters to
episodic. This is a layer-as-Type-B-proxy that does NOT serve the
midloop use case (where Type A and B must mix in episodic for
synthesis). Use the orthogonal `source:authoritative` tag.

### Q5. snapvec Python API direct for embed + cosine_sim?

**YES, both via libraries.** snapvec 0.7.1 installed. vstash 0.32.0
installed.

- Embed: `from vstash.embed import embed_query, embed_texts`. Local
  fallback works without daemon.
- Cosine_sim: snapvec exports `SnapIndex`, `IVFPQSnapIndex`,
  `PQSnapIndex`, `ResidualSnapIndex`, `padded_dim`, `get_codebook`,
  `rht`. For standalone semantic-similarity in the data pipeline:
  `numpy.dot(a, b) / (norm(a) * norm(b))` or `SnapIndex.search()`.
- Pure library, no subprocess. Pipeline can be a Python module.

### Q6. TrajectoryWindow session-scoped lives as instance state of Memory.observe_step?

**YES.** `Memory` is instantiated per-project (`memory.py:218-249`)
and already maintains live session state (`self._write_decider._seen`
is session-scoped). Adding `self._trajectory: deque[StepObservation]`
matches the existing pattern. Persist only relevant steps at end-
of-task via `is_persistable_step()` filter.

Caveat: if the app creates multiple `Memory(project=X)` instances
per session (e.g. each subprocess), the trajectory is lost. Mitigate
by documenting "reuse the same Memory instance across a session."
Do NOT promote to a class-level singleton -- introduces coupling.

## Implementation plan -- ordered by dependencies

### Phase 0 -- start this week (~1 day total)

1. **Tag convention `source:authoritative`** (~30 min). Document in
   `docs/extending.md`. Add the filter in `Memory.consolidate()`
   and `Memory.forget()` (single-line check each). No new decider.
   Permits ingesting WHO protocols pre-marked.
   **Fail-closed semantics (Jay 2026-04-19):** the predicate is
   `source_is_authoritative_or_unknown(doc)` -- if the tag does not
   parse, the format changed, or the value is unrecognized, the
   default is NO mutative action. Safety-critical principle: when
   in doubt, do not act. Concretely the filter shape is
   `if not source_is_safely_derived(doc): continue`, never
   `if source == "authoritative": continue`. The asymmetry costs us
   a small number of legitimate Type-A consolidations / forgets
   when tags are temporarily malformed; the alternative loses
   protocols permanently.
2. **CLI: `merken remember --immutable`** (~30 min). Maps to tag
   `source:authoritative`. User-friendly flow for protocol ingest.
3. **Tests for staleness fix already shipped** (~1h). Smoke tests
   for `_supersede_brief()` + verify `**As of:**` appears in
   generated briefs.

### Phase 1 -- midloop data pipeline (~1-2 weeks, Jay's priority)

4. Module `merken/training/midloop_dataset.py`. Ingests Type B
   protocols, runs large-model + small-model side by side, aligns
   token-level with edit distance, filters with snapvec semantic
   similarity (threshold ~0.90 calibrated empirically). Output:
   JSONL with `(prompt, model_response, intervene_at_token_idx)`
   triples.
   **Sizing note (Jay 2026-04-19):** with ~20-50 divergent regions
   per case and 1k cases, expect 20k-50k subsequence embeddings.
   `embed_texts` is much faster in batch than `embed_query` per
   subsequence. Architecture: accumulate ALL divergent
   subsequences for a case, then a single `embed_texts(batch)`
   call at end-of-case. Prefer one batch per case (clear lifecycle)
   over one global batch (memory pressure).
5. Pipeline standalone CLI: `merken-midloop-dataset --protocol-dir
   ... --large-model claude-sonnet --small-model gemma-4e4b --out
   dataset.jsonl`.
6. Pilot 1k cases over 3 WHO protocols (Phase 0 of Jay's spec).
   Dataset only, no model yet.

### Phase 2 -- midloop primitive (~3-5 days, after pilot signal)

7. `merken/policies/midloop.py`: Protocol + dataclasses +
   `HeuristicMidloopDecider` + `NoopMidloopDecider` +
   `ShadowMidloopDecider`. ~250 lines.
8. `Memory.observe_step()` + `format_midloop_audit_row()`.
9. Tests.

### Phase 3 -- NanoGPTMidloopDecider (training cycle, weeks)

10. Train on Phase 1 dataset. Same loop as NanoGPTWriteDecider.
11. ShadowMidloopDecider with NanoGPT shadow + Heuristic primary.
    Accumulate ~200 (decision, outcome) pairs for graduation.

**Token-level inference loop architecture (Jay 2026-04-19):**
deliberately deferred until the model works in shadow mode. Cannot
decide loop architecture without knowing what streaming APIs the
large model exposes. Keeping this open prevents premature
commitment. The decision arrives when we know which large model
runs in production and what its streaming semantics are.

## What NOT to build (anti-overengineering)

- Base class abstracta para los N primitivos (cada Decision shape
  distinto es saludable).
- Generalize ShadowWriteDecider to `Shadow[Generic[T]]` (premature
  hasta 3+ casos).
- Persistencia de TrajectoryWindow step-a-step (session-scoped
  state, no memory).
- ClaimExtractor with LLM (Jay's spec discards -- midloop learns
  end-to-end, no parses claims).
- ClaimVerifier with entailment (same reason).
- ResponseCorrector post-hoc (correction is inline via stream
  injection).
- Token-level inference loop ANTES de Phase 3 (depends on how the
  large model exposes streaming -- design with the model in hand).

## MedLocal as the target case -- bounded reliability contract

Approach justified because the use case is clinical: hallucinating
costs lives. Three non-negotiable decisions:

1. **Domain-bounded assistant**: only answers within memory; refuses
   out-of-domain. Not a "general trustworthy assistant."
2. **Auto-correction with citation**: not just flag; rewrite with
   traceable citation to the specific protocol clause.
3. **Robustness**: must generalize without overfit; not publishable
   nor useful if domain-bound.

Asymmetric metrics (clinical stakes):

- **Hallucination detection rate**: of incorrect baseline responses,
  % detected.
- **False alarm rate**: of correct responses, % marked incorrectly.
- **Precision under intervention** (target ~100%): of
  approved/corrected responses, % correct.
- **Citation accuracy**: of corrections, does the cited clause
  actually support the correction?

Optimization principle: fix "approved-without-intervention >=99%
correct" as hard constraint, maximize detection rate given that
constraint. Do NOT optimize F1.

## Probe A/B findings that affect this design

Probes (engram, 2026-04-19) confirmed:

- Recall bottleneck is not ranking -- it is brief volume and
  freshness (Type A property).
- 30% hallucination with stale context is NOT "LLM prefers briefs
  over episodic" -- it is total absence of fresh information. 0/3
  WRONG cases had a contradicting episodic; 3/3 had stale or
  absent context entirely.
- Ingestion gap identified: SessionEnd has one-session delay; git
  commits are not auto-ingested.

Implication for midloop: staleness does NOT apply to Type B
(protocols). Training over synthetic data derived from protocols is
sound. But it reinforces that Type A and B require differential
treatment in inference -- exactly what the tag-based convention
above enables.

## Open uncertainties (not resolved in design)

1. **Input encoding of midloop**: (a) last N tokens only, (b) last
   N + prompt embedding, (c) last N + large-model hidden state.
   Tentative preference (b). (c) requires local model, not portable.
2. **Parameter count of midloop**: start same as
   NanoGPTWriteDecider (~800K), adjust on under/overfitting.
3. **Quality of automatic labeling**: probably sufficient on factual
   domain (e.g. LOTR), ambiguous in clinical where multiple phrasings
   are correct. May require semantic labeling with snapvec as
   proposed.
4. **Semantic threshold for snapvec filter at labeling**: 0.90 is
   intuition; calibrate empirically.

## Decision: data pipeline first, NOT input encoding first

Reasons:
- snapvec exists and works (validated benchmarks).
- Pipeline with snapvec as semantic filter produces day-one
  superior dataset.
- Midloop input encoding is designed better with real data in hand.

## What this spec deliberately leaves out

- Execution order beyond "data pipeline first" -- the rest depends
  on validations against running code.
- Concrete threshold values (emerge from shadow mode, not design).
- Specific semantic-similarity algorithm at runtime (depends on
  snapvec config).
- Whisper mode / stream-injection format (depends on how the large
  model exposes streaming).
- MedLocal as a product (CHW interface, deployment, real clinical
  validation).
- Rollback/abort policy if midloop is used in autonomous-agent
  context (Reforge explicitly out of scope).
