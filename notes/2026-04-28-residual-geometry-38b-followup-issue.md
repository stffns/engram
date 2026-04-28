## Motivation

#38 produced a partial-band signal on EVR_1 in the predicted direction
(3-seed AUC 0.760 at N-variable, 0.824 at N=4 controlled with lower CI
0.768) but failed the pre-registered primary gate on PR. EVR_1 was
named a "consistency check, not independent evidence" of PR by #38's
spec, so it cannot post-hoc lift #38's verdict. This issue
pre-registers a follow-up to validate EVR_1-as-gate on **data disjoint
from #38**, with explicit pre-registration of N-control and two new
baselines designed to rule out structural confounds.

The hypothesis being tested is the same one #38 tested at the
spectrum level (residual covariance separates role-mixed clusters
from same-role clusters), narrowed to the *specific functional*
that worked in #38 (EVR_1) and tested under conditions that protect
against the two known confounds.

## What is genuinely new in #38b vs #38

- **EVR_1 is the pre-registered primary metric.** PR is reported as
  a sanity check (we expect it to remain inverted) but does not gate
  the decision.
- **Disjoint data.** The 50topics and 20topics scenarios that produced
  the EVR_1 finding in #38 are **excluded** from the gate set. New
  scenarios are used.
- **N-mode is pre-registered.** Two AUCs reported separately: N=4
  fixed vs N variable. Decision rule applies to each independently
  (see Gate table).
- **Two new baselines** in addition to mean-cosine: cluster size N
  alone and an O(N) outlier-detection metric. EVR_1 must beat all
  three baselines by delta-AUC >= 0.10.

## Pre-registered hypothesis

EVR_1 (top eigenvalue ratio of the residual covariance) separates
role-clean clusters from role-mixed clusters with ROC-AUC > 0.80
lower 95% CI on data disjoint from #38, and beats all three
baselines (mean intra-cluster cosine, cluster size N alone, and
an O(N) concentration metric) by delta-AUC >= 0.10.

## Pre-registration

### Data (disjoint from #38)

**Primary gate set:**
- `experiments/loop_quality/scenarios/knowledge_update_hard.json`
  (108 events, 5 topics, harder synthetic variant of the
  knowledge_update family).

**Secondary transfer set (reported, but does not gate the decision
unless primary is borderline):**
- A real-content scenario chosen at random before run from
  {`jay_vstash_2026_04_09_snapshot.json`, `analytics_project.json`,
  `session_2026_04_09.json`}, deterministically by the run seed.
  Pre-commit: the choice is recorded in `auc_summary.json` and
  cannot be re-rolled if the result is unfavorable.

**Hard exclusion:** `knowledge_update.json`,
`knowledge_update_20topics.json`, and `knowledge_update_50topics.json`
must NOT appear in the gate computation. They are the discovery set
and reusing them would be silent overfitting.

### N control (explicit two-mode pre-registration)

Two distinct questions answered separately:

(a) **N=4 fixed.** Does EVR_1 separate clean from contaminated when
all clusters have N=4? Synthetic clusters constructed with K=4 events
each, exactly. AUC computed on this restricted set.

(b) **N variable (4-6).** Does EVR_1 separate clean from contaminated
when N is allowed to vary, matching the construction in #38? AUC
computed on the full set.

The two answer different production-relevant questions:

- If only (a) passes: EVR_1 is a fixed-N gate. To use in production,
  `consolidate()` would need to enforce N=K constraints on its
  emitted clusters (split-after-K). Restrictive but viable.
- If only (b) passes: N variation is doing the heavy lifting, not
  the eigenvalue structure. Likely confounded; do not ship.
- If both pass: EVR_1 is N-robust. Production use is straightforward.
- If neither passes: drop EVR_1 as a gate.

Pre-commit: report both AUCs side by side. Apply the gate rule below
to each. Final disposition is the joint of the two outcomes.

### Baselines (must beat all three by margin >= 0.10)

**Baseline 1: cluster size N alone.**
- Score = N (for the N-variable analysis only; degenerate at N=4
  fixed, where N is constant and not informative).
- Tests whether the EVR_1 signal is just N-driven on the
  N-variable side.

**Baseline 2: O(N) concentration metric (no SVD).**
- `MaxNormRatio = max_i(||residual_i||) / mean_i(||residual_i||)`
- Captures "is there a single event far from the cluster centroid?"
- Cheap (O(N) after centering, no SVD).
- Pre-commit: if `MaxNormRatio` AUC matches EVR_1 AUC within 0.05,
  EVR_1 is doing nothing eigenvalue-specific -- the spectrum
  interpretation collapses to "one event is far from the rest" and
  the cheaper metric is preferred.
- Reported under both N modes.

**Baseline 3: mean intra-cluster cosine similarity** (carry-over from
#38, the metric the original motivation said fails).
- Already implemented in the experiment script.
- Reported under both N modes.

EVR_1 must beat ALL THREE baselines by delta-AUC >= 0.10 (point
estimate, with 95% bootstrap CI for each baseline reported alongside)
to count as genuine eigenvalue signal.

### Gate table

Decision applied separately to (a) N=4-fixed and (b) N-variable:

| Outcome (per N-mode)                                   | Verdict                              |
|--------------------------------------------------------|--------------------------------------|
| EVR_1 lower CI > 0.80 AND beats all 3 baselines by 0.10 | STRONG: ship as gate (in this N mode) |
| EVR_1 lower CI 0.65-0.80 AND beats all baselines       | PARTIAL: input to combined gate (#40) |
| EVR_1 lower CI < 0.65 OR ties any baseline within 0.05  | NO_GO in this mode                   |

Joint dispositions:

- STRONG on both modes -> **GO**: integrate EVR_1 as a production
  gate without N constraint. Open #41 for `consolidate()` integration.
- STRONG on (a) only -> **GO_RESTRICTIVE**: ship behind a
  cluster-size-K constraint. Open #41 for size-constrained
  integration.
- STRONG on (b) only -> investigate whether the N=4 result was
  underpowered (more clusters? bootstrap?) before deciding. Default
  is **NO_GO** until that is resolved.
- PARTIAL on either mode -> **PARTIAL**: input to #40 only.
- NO_GO on both modes -> **NO_GO**: drop EVR_1 as a gate.

### Multiple-comparison protection

Pre-commit: the gate is on EVR_1 only. PR / NormVar / AngDisp /
MaxNormRatio / N / mean-cosine are all reported, but only EVR_1's CI
gates the decision. No Bonferroni correction needed because we are
not searching across metrics; we are validating one specific
post-hoc selection from #38.

### Reproducibility / artifact requirements

Mirroring #38:

- Code-review pass on the experiment script before run, per global
  CLAUDE.md (the 2026-04-20 hardcoded-path lesson).
- All AUCs with 95% bootstrap CIs (n_resamples >= 1000).
- Strict-JSON `auc_summary.json` with scenario sha256, vstash
  version, numpy version, seed, embedding model, embedding dim,
  bootstrap counts, AND the secondary scenario picked deterministically
  by seed.
- 3 seeds (42, 43, 44) for the gate computation. Single-seed
  reporting is deprecated per project memory.
- Independent `rng.spawn(N)` per stage (cluster construction,
  cluster-level bootstrap, pairwise CSV sampling, pairwise bootstrap)
  so adding a stage downstream cannot retroactively shift earlier
  CIs.

## Why this is not p-hacking

- The hypothesis ("residual covariance separates role-mixed clusters")
  was pre-registered in #38, before any data was seen.
- The functional choice (EVR_1 vs PR) is post-hoc relative to #38, so
  it cannot be validated on #38's data alone -- this issue exists
  precisely to validate it on disjoint data.
- The disjoint data is committed to before run.
- Baselines control for the two specific confounds we know about:
  cluster size and "outlier event" -- both of which co-occur with the
  positive class in #38's data, and could be doing the work EVR_1
  appears to do.
- The gate threshold (lower CI > 0.80, delta-AUC >= 0.10 over each
  baseline) is committed before the run. No re-rolling.

## Out of scope

- Production integration into `consolidate()` (separate issue if
  EVR_1 passes).
- Re-running #38 on more scenarios with a different aim (different
  question).
- Comparing different embedders (would change the residual geometry
  itself; out of scope for this validation).
- Combining with #39 (semantic markers) -- that is #40, gated on
  this issue.

## Deliverable

`experiments/role_geometry/evr_1_validation.py` plus a writeup at
`notes/role_geometry_38b.md` containing:

- AUC table for all metrics under each N mode (table layout
  pre-committed below).
- Bootstrap 95% CIs for each cell.
- Per-baseline delta-AUC and decision per the Gate table.
- Joint disposition.

Output files:

- `metrics_by_cluster.csv` (one row per cluster, all metrics +
  ground-truth label, per N mode).
- `auc_summary.json` (per N mode + joint disposition).
- `notes/role_geometry_38b.md` (writeup with go/no-go).

### Pre-committed table layout

| metric          | N=4 fixed AUC [CI] | N variable AUC [CI] |
|-----------------|---------------------|----------------------|
| EVR_1 (primary) |                     |                      |
| PR              | (sanity, expected inverted) |              |
| NormVar         |                     |                      |
| AngDisp         |                     |                      |
| MaxNormRatio    |                     |                      |
| N alone         | -- (degenerate)     |                      |
| mean cosine     |                     |                      |

| comparison                          | delta-AUC (N=4) | delta-AUC (Nvar) |
|-------------------------------------|------------------|--------------------|
| EVR_1 vs MaxNormRatio               |                  |                    |
| EVR_1 vs N alone                    | --              |                    |
| EVR_1 vs mean cosine                |                  |                    |

| N mode    | EVR_1 lower CI cleared 0.80? | Beats all baselines by 0.10? | Disposition |
|-----------|------------------------------|------------------------------|-------------|
| N=4 fixed |                              |                              |             |
| Nvariable |                              |                              |             |
| **Joint** |                              |                              |             |

## References

- #38 (parent): `notes/2026-04-28-directional-residual-geometry-issue.md`
- #38 results: `notes/directional_residual_geometry.md`
- #39 (sibling, runs in parallel): `notes/2026-04-28-semantic-markers-issue.md`
- #40 (depends on this and #39): `notes/2026-04-28-combined-gate-issue.md`
- merken paper section 7.3 (clustering bottleneck) and 7.6 (future work).
