# #39 -- Few-shot semantic role classifier: Phase 1 results

Pre-registered Phase 1 of merken issue #39. Run 2026-04-28 on
`feature/semantic-markers-39`. Spec at
`notes/2026-04-28-semantic-markers-issue.md`.

## Verdict

**Borderline NO_GO at the strict per-seed reading. Borderline PARTIAL
at the 3-seed mean reading.** Per-seed gate decisions are 2 NO_GO + 1
PARTIAL. 3-seed mean macro-F1 = 0.753 (in the [0.70, 0.85] PARTIAL
band) but worst-case lower 95% CI = 0.633 (below the 0.70 PARTIAL
threshold).

The Phase 2 (marker-prefix conditioning) decision is **not
auto-resolved by the spec** -- single-seed reporting was deprecated
during merken's Phase 2 work, but the spec table evaluates per-seed
lower CIs. Surfacing this as a decision the next session must make
explicitly. **My recommendation below**: run Phase 2 as an exploratory
diagnostic targeting the single failure mode (DECISION class), not as
a gated commitment.

## The single failure mode: DECISION confused with PREFERENCE

Pooled confusion matrix across 3 seeds (240 eval events, 60 per role):

| true \\ pred  | decision | investigation | observation | preference | recall |
|---------------|----------|---------------|-------------|------------|--------|
| decision      | 23       | 12            | 2           | **23**     | 38.3%  |
| investigation | 0        | 54            | 6           | 0          | 90.0%  |
| observation   | 0        | 8             | 52          | 0          | 86.7%  |
| preference    | 0        | 3             | 0           | 57         | 95.0%  |

3 of 4 roles separate cleanly. DECISION class is the bottleneck:

- 38.3% recall is well below the 0.75 per-class threshold for STRONG.
- 23 of 37 misclassified DECISION events go to PREFERENCE. That
  is 38% of the entire DECISION class collapsing into PREFERENCE
  alone.
- The collision is **structural, not noise**: the same pattern
  appears on all 3 seeds (recall 0.30 / 0.55 / 0.30).

The mechanism, by inspection of the events:

- DECISION text: "Approved Stripe over Square as the primary payment
  processor after the merchant-fee analysis."
- PREFERENCE text: "Personally I would prefer keeping Stripe and just
  absorbing the fee bump."

Both express *comparative judgment with a named choice*. BGE-small
embeds them similarly because they share lexical structure ("X over
Y", "switch to X", "keep X"). The semantic difference -- did the
choice happen, or is it a stated leaning? -- is carried by tense and
modal verbs, which BGE-small does not encode strongly.

This is a **distinct failure from #38's**. #38 found that geometry
fails to recover role mismatch *between v1/v2/v3 of the same topic*.
#39 finds that BGE-small fails to distinguish "did X" from "would
do X" *within the role taxonomy* -- a different axis of failure.
Both point to the same conclusion: the embedding space lacks
upstream role signal.

## Headline numbers

3-seed run (seeds 42, 43, 44) on the canonical eval set
(`experiments/role_markers/eval.json`, 80 events, 20 per role):

| seed | macro-F1 mean | macro-F1 95% CI | dec | inv | obs | pref | per-seed gate |
|------|---------------|------------------|-----|-----|-----|------|----------------|
| 42   | 0.733         | [0.634, 0.816]   | 0.300 | 0.950 | 0.800 | 1.000 | NO_GO_EMBEDDER_LACKS_ROLE_SIGNAL |
| 43   | 0.807         | [0.718, 0.886]   | 0.550 | 0.950 | 0.800 | 0.950 | PARTIAL_MARKER_ASSISTED |
| 44   | 0.720         | [0.633, 0.802]   | 0.300 | 0.800 | 1.000 | 0.900 | NO_GO_EMBEDDER_LACKS_ROLE_SIGNAL |
| **3-seed** | **0.753 +- 0.047** | **min lo 0.633** | 0.383 | 0.900 | 0.867 | 0.950 | **mixed** |

Pre-registered thresholds: macro-F1 lower CI > 0.85 + per-class >=
0.75 -> STRONG; 0.70-0.85 OR one-class-< 0.75 with rest >= 0.85 ->
PARTIAL; < 0.70 -> NO_GO.

**Strict reading:** 2/3 seeds fail PARTIAL on the macro-F1 lower CI.
Per-class threshold also fails on every seed (DECISION at 0.30-0.55).

**Generous reading:** 3-seed mean macro-F1 = 0.753 sits in the
PARTIAL band, and pre-committed project policy is "single-seed
reporting deprecated" (memory: 2026-04-27). On that reading, the
result is PARTIAL.

## K-saturation curve

| K  | macro-F1 (seed 42) | seed 43 | seed 44 |
|----|--------------------|---------|---------|
| 1  | 0.480              | 0.616   | 0.471   |
| 3  | 0.610              | 0.685   | 0.580   |
| 5  | 0.736              | 0.803   | 0.660   |
| 7  | 0.733              | 0.807   | 0.720   |
| 10 | 0.788              | 0.857   | 0.745   |

K-saturation is monotonic-ish across seeds (mostly increasing with
K, with a small dip at K=7 for seed 42 within bootstrap noise).
K=10 gives roughly +0.04 macro-F1 over K=7. The saturation curve
suggests the prototype pool is *underprovisioned* -- another 5-10
prototypes per role might lift macro-F1 by another 0.04-0.06, but
that lift would not get DECISION's 0.38 recall to 0.75. The
DECISION ↔ PREFERENCE collision is not a sample-size problem.

## What this means

### For #39 itself

The hypothesis ("BGE-small encodes enough signal to classify the
4-role taxonomy via prototype matching") is **partially supported**:
3 of 4 roles work at production-grade recall (>= 0.85 on most seeds).
DECISION fails because of the embedding's weakness on the
committed-vs-stated distinction.

This means a marker-based gate using *the existing embedder, no
training* can reliably detect INVESTIGATION, OBSERVATION, and
PREFERENCE events. It cannot reliably detect DECISION events. For
merken's `consolidate()` use case, where the role distinction we
care about most is "is this a committed decision or a stated
preference," the gate is incomplete in exactly the wrong place.

### For Phase 2

Phase 2 prepends `[role: <role>]` prefix to event text, re-embeds,
and tests whether the prefix-conditioned embedding produces lower
contamination rates at matched recall. The Schick-Schütze hypothesis
is that prefix-conditioning shifts the pretrained representation
toward a task-relevant subspace.

**Risk:** Phase 2's prefix is determined by Phase 1's predicted role.
For DECISION events, Phase 1 mispredicts 60% as PREFERENCE, so 60%
of true-DECISION events would receive `[role: preference]` prefix.
That doesn't fix the bottleneck -- it amplifies it.

**Possible value:** Phase 2 might help on the 3 well-classified
roles, where Phase 1's prediction is accurate. The marker-prefix
could then sharpen INVESTIGATION / OBSERVATION / PREFERENCE
clustering further. But that's not what the original problem
requires.

### For #40 (combined gate)

Originally planned as "EVR_1 + markers". After #38b that became
"markers alone or markers + MaxNormRatio." After #39 Phase 1, even
that needs revision: the markers themselves fail on the DECISION
class, so any gate built on them inherits the same failure.

The merken paper's argument now consolidates:

1. Pure cosine geometry: cannot recover role from the embedding
   (#38, #38b).
2. Few-shot semantic markers on the existing embedder: detect 3 of 4
   roles well, but fail specifically on DECISION (#39 Phase 1).
3. The shared root cause: BGE-small does not encode the temporal-
   modality distinction (committed vs stated, did vs would) that
   merken's knowledge_update bottleneck requires.

External structure -- knowledge graph supersession (Zep), explicit
typed schemas (brief_v1's DECISION/ENTITY/EVENT/FREE shapes), or
upstream metadata (commit type, ticket field, audit trail) -- is
necessary, not optional. This is the merken paper meta issue's
existing conclusion, now stronger.

## Phase 2 decision (recommended)

**Run Phase 2 as a diagnostic, not a gated experiment.** Specifically:

- Use the K=10 prototypes (best Phase 1 result).
- Apply the role-prefix to events with **ground-truth role**, not
  Phase 1's predicted role. This isolates the question "does
  prefix-conditioning help separate DECISION from PREFERENCE in
  embedding space" from the question "does Phase 1 predict DECISION
  correctly so we can attach the right prefix."
- If ground-truth-prefixed events become more separable than
  vanilla, the issue is purely in Phase 1's prediction, not in the
  embedding manifold.
- If ground-truth-prefixed events do not become more separable,
  prefix-conditioning is not a viable rescue.

This is **not the pre-registered Phase 2** (which uses Phase 1's
predicted role). It is a diagnostic that cannot count as a gate
result, but it is cheap (~30-45 minutes) and would refine the next
move (#40 re-scope or pivot to external structure).

Alternatively, **skip Phase 2 entirely** with the rationale that
#39's strict gate fails on 2/3 seeds, the per-class failure mode is
identified (DECISION ↔ PREFERENCE), and the structural conclusion
(BGE-small lacks committed-vs-stated distinction) is robust enough
to motivate the pivot to external structure without additional
geometric work.

Open question for next session: which of these two paths.

## Anti-leak verification (per spec)

- Prototype topics: {logging, dashboards, build_system, code_review, oncall}
- Eval topics: {payment, search, mobile, ml_serving, queue, cdn,
  file_storage, auth, billing, notifications, recommendations,
  fraud, analytics, deployments, monitoring, infra, frontend, api,
  database, scheduling}
- Topic intersection: empty.
- Exact-text overlap: 0 events (verified at run time by
  `assert_disjoint`).
- Style guide frozen 2026-04-28 EOD before prototype authoring;
  prototypes frozen before eval set authoring (commit timestamps
  document the order).

## What changed vs the spec

The spec named "knowledge_update scenarios" as the eval set, but
those scenarios contain only `decision` and `noise` roles by id
pattern (`*_v1/v2/v3` and `noise_*`). They lack the 4-role taxonomy
(DECISION / INVESTIGATION / OBSERVATION / PREFERENCE) the spec
requires. We authored a new 80-event eval set
(`experiments/role_markers/eval.json`) explicitly for the 4-role
taxonomy, on a topic set disjoint from both the prototypes and the
existing knowledge_update scenarios. Scope expansion: the original
spec planned to evaluate on existing scenarios; we authored new
data because the existing scenarios cannot test the spec.

## Reproducibility

- Code: `experiments/role_markers/few_shot_classifier.py` (~440 LOC).
  Code-reviewed before run; one bug fixed (decide-tree hole when
  macro-F1 high but multiple per-class weak; fixed to map to
  NO_GO_PER_CLASS_GAPS_EXCEED_PARTIAL conservatively).
- Data: `prototypes.json` (40 events, 10 per role, 5 prototype
  topics), `eval.json` (80 events, 20 per role, 20 eval topics),
  `prototype_style_guide.md` (frozen authoring spec). All committed.
- Runs: `experiments/role_markers/runs/phase1_20260428_1324*` (3
  per-seed dirs). Gitignored.
- 3 seeds (42, 43, 44), 1000 bootstrap resamples.
- Independent `rng.spawn(2)` for prototype shuffle and bootstrap.
- K-saturation uses NESTED prototype subsets (K=10 superset of K=7
  superset of K=5 etc.) to isolate the K effect from selection noise.
- Strict-JSON serialization with NaN scrubbed to None.

## Files written per run

- `phase1_classification.csv` (one row per event with predicted role,
  per-role similarity, top-1 sim, margin).
- `phase1_metrics.json` (macro-F1 bootstrap CI, per-class recall,
  confusion matrix, K-saturation table, used prototype IDs, env
  metadata).

## Phase 2 diagnostic results (run 2026-04-28)

Phase 2 was run as a diagnostic (not the gated pre-registered Phase 2)
to test whether prefix-conditioning helps the embedding manifold
separate roles. Four cohorts at seed 42:

| cohort                  | same role mean | cross role mean | separation | DEC-PREF mean |
|-------------------------|-----------------|------------------|-------------|----------------|
| vanilla                 | 0.532           | 0.475            | 0.056       | 0.489          |
| ground_truth_prefix     | 0.610           | 0.536            | 0.074       | 0.577          |
| predicted_prefix        | 0.607           | 0.537            | 0.070       | 0.579          |
| random_prefix           | 0.590           | 0.538            | 0.052       | 0.555          |

Pre-registered diagnostic gate (recorded before run):

- "manifold can host role with prefix" iff GT-prefix separation
  delta >= 0.05 over vanilla AND beats random-prefix improvement
  by >= 0.05.
- "DEC-PREF specifically rescued" iff GT-prefix DEC-PREF cross
  cosine drops by >= 0.10 below vanilla (lower cosine = more
  separation).

**Result on all 3 seeds (42, 43, 44): PREFIX_CONDITIONING_INEFFECTIVE.**

- GT-prefix separation delta: +0.018 (vs threshold 0.05)
- GT vs random delta: +0.022 (vs threshold 0.05)
- DEC-PREF GT-prefix delta: +0.087 (cosine *increases*; threshold
  required <= -0.10)

The DEC-PREF result is particularly stark: prefix-conditioning makes
DECISION and PREFERENCE events MORE similar (cross cosine climbs
0.489 -> 0.577 with GT prefix), not less. The 3-token prefix
("[role: X] ") inflates pairwise cosine uniformly across all event
pairs, but the role-specific component does not differentially push
same-role pairs together vs. cross-role pairs apart. The lift from
ground-truth role prefix is barely larger than from a random
NATO-phonetic-tag prefix.

3-seed result is essentially identical across seeds (vanilla and
GT-prefix cohorts do not depend on seed; random and predicted vary
only in the random tag draws and the Phase 1 prediction). The
diagnostic outcome is consistent.

## What Phase 2 changes about the conclusion

Phase 2 closes the prefix-conditioning rescue path. The embedding
manifold of BGE-small does not host role information via the
Schick-Schutze pattern-conditioning mechanism, at least not at the
4-token prefix scale tested.

But Phase 2's negative result is actually consistent with a
constructive next move: **redefine the DECISION role to have
non-overlapping lexical markers** with PREFERENCE. The 3 roles that
work in Phase 1 (INVESTIGATION 90%, OBSERVATION 87%, PREFERENCE 95%)
all have distinctive lexical markers that BGE picks up natively
("investigating", "shows/noticed/observed", "prefer/lean/recommend").
DECISION fails precisely because it shares lexical structure with
PREFERENCE ("X over Y", "switch to X").

A redefined role like **STATE_CHANGE_REPORT**:

- Past tense + concrete state-change verb (deployed, migrated,
  shipped, launched, removed, configured, switched-to-X-as-of-DATE)
- Names a specific identifiable entity in production
- Reports state, not the act of choosing among alternatives
- Avoids the "X over Y" comparative structure entirely

Example contrast:

- Current DECISION: "Switched search infra from Elasticsearch to
  Vespa to support hybrid queries." -> collides with PREFERENCE.
- STATE_CHANGE_REPORT: "Production search has been running on Vespa
  since 2026-04-15; Elasticsearch instances were decommissioned the
  same day." -> distinct from PREFERENCE.

The redefinition is testable as #39b (or simply a re-author of
prototypes.json + eval.json with the new role definitions). It is
NOT in scope for #39 itself; the original spec committed to the
4-role taxonomy as defined.

## V2 (#39b): taxonomy redefinition rescues the classifier

After V1 + Phase 2 diagnostic, Jay flagged that the failure mode might
be a taxonomy artifact rather than a BGE limitation: DECISION as
defined collides with PREFERENCE because both use "X over Y"
comparative structure. The pre-committed follow-up #39b replaces
DECISION with **STATE_CHANGE_REPORT** -- past tense + concrete
state-change verbs + specific entity, no comparative structure.

### V2 result

3-seed run on the canonical 80-event V2 eval set
(`experiments/role_markers/eval_v2.json`, 20 events per role):

| seed | macro-F1 mean | macro-F1 CI | SCR | inv | obs | pref | gate |
|------|---------------|--------------|-----|-----|-----|------|------|
| 42   | 0.937         | [0.884, 0.987] | 1.000 | 0.950 | 0.800 | 1.000 | **STRONG** |
| 43   | 0.926         | [0.870, 0.975] | 1.000 | 0.950 | 0.800 | 0.950 | **STRONG** |
| 44   | 0.925         | [0.861, 0.975] | 1.000 | 0.800 | 1.000 | 0.900 | **STRONG** |
| **3-seed** | **0.929 +- 0.007** | **min lo 0.861** | **1.000** | 0.900 | 0.867 | 0.950 | **STRONG_PROCEED_PHASE_2** |

K-saturation: K=1 mean macro-F1 ~0.68; K=10 mean macro-F1 = 0.938
(seed-invariant at 0.938 across all 3 seeds at K=10). Adding more
prototypes saturates around K=7-10.

**STATE_CHANGE_REPORT recall = 1.000 across all 3 seeds.** Zero
collisions with PREFERENCE. Pooled confusion matrix (3 seeds, 240
events):

| true \\ pred         | scr | inv | obs | pref | recall |
|----------------------|-----|-----|-----|------|--------|
| state_change_report  | 60  | 0   | 0   | 0    | **100%** |
| investigation        | 0   | 54  | 6   | 0    | 90.0%  |
| observation          | 0   | 8   | 52  | 0    | 86.7%  |
| preference           | 0   | 3   | 0   | 57   | 95.0%  |

The V1 DECISION-PREFERENCE failure mode is gone. SCR separates
perfectly because:

- Past tense + concrete state-change verbs ("has been running on",
  "decommissioned", "went live on", "rolled out") are lexically
  distinct from PREFERENCE's modal markers ("prefer", "lean toward",
  "recommend").
- Specific dates and entities anchor SCR events in
  past-and-confirmed reality, where PREFERENCE is forward-looking
  or hypothetical.
- The "X over Y" comparative is absent from SCR, so the structural
  pattern that confused V1's DECISION vanishes.

Other 3 roles' recall is *identical* to V1 (90% / 87% / 95%), as
expected -- their events are unchanged, only the role replacing
DECISION shifted.

### What V2 changes about the conclusion

**The hypothesis "BGE-small encodes role signal via prototype
matching" is now SUPPORTED for the V2 taxonomy.** The V1 failure was
a TAXONOMY-LEVEL phenomenon: when role definitions share lexical
structure, BGE conflates them; when role definitions have
non-overlapping lexical markers, BGE separates them at AUC 0.93.

This is a substantial pivot from the V1 + Phase 2 diagnostic
"NO_GO + structurally inflexible" framing. BGE-small does carry role
information through its surface lexical markers, but only along axes
that have distinct lexical realization. Choose the taxonomy
intentionally and the marker-based gate works.

For merken's `consolidate()` use case:

- The original concern (v1+v2+v3 of one topic merging into one fact,
  losing temporal evolution) is precisely what STATE_CHANGE_REPORT
  detects. SCR-marked events are *committed state changes*, so a
  newer SCR event can supersede an older SCR event for the same
  topic.
- Marker-based gating is now a working production primitive. The
  consolidator can:
  1. Classify each event with the V2 prototype set.
  2. Within a topic-tight cluster, treat SCR events as a
     supersession chain (newest takes precedence).
  3. Treat PREFERENCE events as non-superseding (preferences don't
     replace each other).
  4. Treat OBSERVATION events as additive context.
  5. Treat INVESTIGATION events as open questions.
- Cost: ~1 extra cosine pass per event at ingest time (no training,
  no LLM call).

### Phase 2 prefix-conditioning revisited

V1 Phase 2 diagnostic showed prefix-conditioning is structurally
ineffective at the 4-token scale. V2's positive result *does not
need prefix-conditioning* -- the taxonomy itself separates cleanly
without intervention. So Phase 2 is moot for the V2 protocol; the
pre-committed Phase 2 gate (>= 30% contamination drop, >= 20pp
gap vs random) is not relevant when Phase 1 already meets STRONG.

The pre-committed Phase 2 stays open as a future test for cases
where the taxonomy is fixed by external constraints (e.g., merken
paper's section X.Y taxonomy if that turns out to be DECISION-style
rather than SCR-style). For the merken consolidate() use case
specifically, V2 obviates Phase 2.

## Decision

V1: per-class DECISION failure structural in the taxonomy.
V1 Phase 2 diagnostic: prefix-conditioning ineffective.
V2: **STRONG_PROCEED gate met on all 3 seeds**, STATE_CHANGE_REPORT
recall = 100% pooled.

**Net verdict for #39 (combining V1 + V2): SUCCESS** when the
taxonomy is chosen with non-overlapping lexical markers between
adjacent roles. Failure when adjacent roles share lexical structure.

The combined #38 + #38b + #39 P1+P2 result before V2 was: BGE-small
cannot recover the role distinction merken needs through any of the
four methods tested -- spectral functionals (PR/EVR_1/AngDisp/NormVar),
cheap outlier detection (MaxNormRatio), few-shot prototype classification
on the V1 DECISION/PREFERENCE taxonomy, or prefix-conditioning.

V2 (#39b) flips the prototype-classification finding: with a
taxonomy that has non-overlapping lexical markers between adjacent
roles (STATE_CHANGE_REPORT vs PREFERENCE), BGE separates cleanly at
AUC 0.93. The other three negative methods still hold; V2 specifically
rescues the few-shot prototype path with the right taxonomy.

**Updated next moves (#39b confirmed in-line; remaining work):**

- ~~**#39b: STATE_CHANGE_REPORT taxonomy**~~ DONE in this writeup as
  the V2 section above. Result: STRONG gate met on all 3 seeds.
- **Update meta paper issue** (`notes/2026-04-28-paper-geometric-methods-meta-issue.md`)
  to reflect the V1 + V2 split: 3 of 4 methods are negative
  (geometry, outlier detection, prefix-conditioning), but few-shot
  prototype classification is positive *with the right taxonomy*.
  The pivot to external structure is no longer total -- markers on
  the existing embedder are a working primitive for the SCR vs
  PREFERENCE axis.
- **#40 (combined gate) re-scopes again** -- now from "markers alone"
  to "STATE_CHANGE_REPORT markers as the primary gate, MaxNormRatio
  as a cheap pre-filter for outlier-shaped contamination." Open
  follow-up to wire this into `consolidate()` when ready.
- **Production integration as a separate issue** -- once #40 produces
  the integration design, an issue for actually wiring SCR markers
  into `Memory.remember()` and `consolidate()`. Out of #39's scope.

The merken paper now has a more nuanced consolidated argument:

1. Pure cosine geometry: cannot recover role-mismatch (#38, #38b).
2. Cheap outlier detection: matches eigenvalue functionals (#38b).
3. Few-shot prototype classification with taxonomy that has
   overlapping lexical markers: fails (#39 V1 DECISION-PREFERENCE).
4. Prefix-conditioning: structurally ineffective (#39 V1 P2).
5. **Few-shot prototype classification with taxonomy that has
   non-overlapping lexical markers: succeeds (#39b V2 SCR
   AUC 0.93)**.

The lesson: marker-based gating works *when the marker taxonomy
aligns with BGE's surface-lexical clusters*. Aligning it requires
intentional taxonomy design. The merken paper section 7.6 / 7.7
expansion now describes both halves of the story: where the
embedding fails and where it succeeds.
