# Directional residual geometry as a role-contamination signal -- results

Pre-registered experiment for merken issue #38. Run 2026-04-28 on
`feature/role-geometry-residuals`. Numbers below are post a labeling
fix found by a focused bug audit (see "Bug audit" section); the audit
confirmed no sign-flipping bug, the labeling fix changed AUC by < 1pp.

## Verdict (per pre-registered primary gate)

**Primary gate (PR) failed per strict pre-registration. Secondary
metric (EVR_1) shows partial-band signal in the predicted direction.
Pre-committed pivot to #39 stands; additionally, #38b opens as a
pre-registered follow-up to validate EVR_1-as-gate on disjoint data.**

Cluster-level PR ROC-AUC across 3 seeds on
`knowledge_update_50topics.json` (1100 events, 50 decision topics x
{initial, refinement, reversal}, 950 noise events), post-fix:

| seed | AUC mean | 95% CI lower | 95% CI upper |
|------|----------|--------------|--------------|
| 42   | 0.211    | 0.130        | 0.298        |
| 43   | 0.260    | 0.172        | 0.362        |
| 44   | 0.200    | 0.120        | 0.282        |
| **3-seed** | **0.224 +- 0.032** | **0.120 (worst lo)** | **0.362 (best hi)** |

Pre-registered thresholds: lower CI > 0.80 (strong), 0.65-0.80
(partial), < 0.65 (no-go). Worst-case lower CI = 0.120, well below
0.65. Best-case upper CI = 0.362 -- still below 0.5.

**Baseline-beat**: mean intra-cluster cosine ROC-AUC = 0.285 +- 0.052
(also inverted). delta-AUC over baseline = -0.061 (PR is *worse* than
the baseline it was supposed to beat). Pre-registered requirement
delta-AUC >= 0.10 not met.

**Replication on `knowledge_update_20topics.json`** (440 events, 20
topics): PR AUC = 0.199 [0.120, 0.287]. Same direction, tighter
magnitude. Negative result is structural, not data-specific.

**Size-controlled re-analysis (N=4 only, post-fix):** PR AUC = 0.165,
CI [0.112, 0.225] across 214 N=4 clusters (81 clean / 133 contaminated).
The inversion is *stronger* under size control than across all sizes,
falsifying the alternative hypothesis "PR inversion is a
PR-bound-by-(N-1) artifact." See "N confound" below.

## What actually happened: the hypothesis was inverted

The issue predicted: contaminated (role-mixed) clusters have **higher**
PR than clean (same-role) clusters because role-mismatch adds independent
variance to the residual covariance.

The data shows the opposite. Per-class PR / EVR_1 medians (3-seed
aggregate, post-fix; `mixed_topic_decisions_role_pure` is now correctly
labeled clean -- bug audit, see below):

| cluster_class                       | label | n_clusters | PR    | EVR_1 |
|-------------------------------------|-------|------------|-------|-------|
| clean_role_initial                  | clean | 60         | 3.89  | 0.307 |
| clean_role_refinement               | clean | 60         | 3.84  | 0.320 |
| clean_role_reversal                 | clean | 60         | 3.93  | 0.299 |
| clean_noise                         | clean | 60         | 3.63  | 0.371 |
| mixed_topic_decisions               | contam | 58        | 3.93  | 0.301 |
| mixed_topic_decisions_role_pure     | clean (post-fix) | 2 | 3.38 | 0.358 |
| **mixed_role_decisions_no_noise**   | contam | 60         | **2.71** | **0.479** |
| **mixed_role_in_topic**             | contam | 60         | **2.72** | **0.475** |

The two role-mismatched-within-topic classes (`mixed_role_*`) have
**~30% lower PR** than every clean class (2.7 vs ~3.5-4.0). Same-topic
v1+v2+v3 collapses to a low-rank residual covariance because BGE-small
puts the three temporal versions almost on top of each other in
embedding space. Adding one event of a *different* role from a
*different* topic shifts the centroid only slightly -- the cluster
remains dominated by one principal axis (high EVR_1 = 0.48).

Meanwhile, clean clusters spanning K different topics (clean_role_X)
spread residuals across many directions because each topic occupies a
distinct semantic neighborhood. Their PR is high because the residual
covariance is approximately uniform over min(N-1, d) directions.

So PR is detecting "is this cluster topic-tight?" not "is this cluster
role-mixed?" The fact that `mixed_role_decisions_no_noise` (decision-
only, no noise events) and `mixed_role_in_topic` (with one noise event)
land at nearly identical PR (2.69 vs 2.73) shows the signal is **role-
mismatch driven, not noise driven**. H3 from the code review is
cleanly resolved: the noise event is not the source of the PR drop.

## Where the eigenvalue spectrum *did* carry signal: EVR_1 and NormVar

The same residual covariance, viewed through different functionals,
points the predicted direction (post-fix 3-seed):

| Metric | AUC mean | stdev | direction relative to issue prediction |
|--------|----------|-------|----------------------------------------|
| PR (primary)        | 0.224 | 0.032 | **inverted** |
| EVR_1 (secondary)   | 0.760 | 0.031 | predicted, partial-band signal |
| AngDisp (tertiary)  | 0.616 | 0.055 | predicted, weak |
| NormVar (auxiliary) | 0.756 | 0.032 | predicted, partial-band signal |

Per-seed lower CIs for EVR_1: 0.700 (seed 42), 0.626 (seed 43), 0.679
(seed 44). Seed 43 is the only run where the lower CI does not clear
the 0.65 partial-band threshold; the other two clear it cleanly.

**Size-controlled (N=4 only) AUCs strengthen the EVR_1 finding:**

| Metric (N=4 only) | AUC | 95% CI | partial-band lower bound (0.65) cleared? |
|-------------------|-----|--------|-------------------------------------------|
| PR                | 0.165 | [0.112, 0.225] | inverted, no |
| **EVR_1**         | **0.824** | **[0.768, 0.879]** | **yes, cleanly** |

At fixed N=4, EVR_1 lower CI = 0.768 lands at the upper end of the
partial band, just below the 0.80 strong-band cutoff. Mean 0.824 is
solidly partial-band. This is despite EVR_1 being explicitly a
"consistency check, not independent evidence" of PR per the issue
spec line 47, so per the pre-registration we cannot promote it to a
gate of its own.

PR and EVR_1 are mathematically coupled (both functions of the
eigenvalue spectrum) but they encode *different* properties:

- PR = `(sum lambda)^2 / sum lambda^2`. Bounded by min(N-1, d). High
  when eigenvalues are uniform (residuals spread across many
  directions).
- EVR_1 = `lambda_1 / sum lambda`. High when one eigenvalue dominates.

A topic-tight + role-mixed cluster (v1+v2+v3 of one topic + one outlier)
has BOTH:

- low total spread (most variance lives in one direction, the
  v1/v2/v3 -> outlier axis) -> low PR
- high concentration in the top axis -> high EVR_1

Same eigenvalue spectrum, two functionals pointing opposite ways.

## Pairwise analysis: residual axis encodes topic, not role

Pre-registered secondary gate: same-topic-different-role pairs should
have **lower** residual cosine than diff-topic-same-role pairs (target
< control under H1). Seed-invariant pair counts: target n=150, control
n=3675 in 50topics; target n=60, control n=570 in 20topics.

| Scenario | target median res-cos | control median res-cos | cliff delta | MW-U p_two_sided |
|----------|------------------------|-------------------------|-------------|------------------|
| 50topics | 0.369                  | 0.080                   | **0.911**   | 6.31e-80         |
| 20topics | 0.399                  | 0.104                   | **0.954**   | 4.96e-34         |

Highly significant in the **opposite** direction. Same-topic pairs
have ~4.6x higher residual cosine than diff-topic-same-role pairs.

The structural reason: residuals are computed against the *global*
centroid (which is dominated by noise events, the most numerous class).
Subtracting the global mean preserves topic structure -- decision
events' residuals point toward their respective topic neighborhoods,
away from the noise mean. Same-topic pairs share that direction
regardless of fine role; cross-topic pairs do not.

This refutes the issue's pairwise framing: the residual axis (with
global centering) does **not** carry role mismatch in any direction
that helps. It carries topic. To recover role information from the
residual axis would require *cluster-local* centering -- which
presupposes the cluster, defeating the purpose of using pairwise
analysis to circumvent clustering ambiguity.

## What this means for adjacent literature

The issue motivated the experiment as testing an orthogonal hypothesis
to Zep / SmartVector / "Attention Is Not Retention" (all three argue
the fix requires *external* structure: graphs, augmented embeddings,
hash-based identity). Our pre-committed framing for failure: "even
higher-order geometry does not recover role structure; external
structure is necessary."

This experiment is **convergent evidence with that field consensus**.
The eigenvalue spectrum carries *some* signal (EVR_1 partial band) but
the practical functional needed to read it (PR) goes the wrong way
under the natural intuition, and pairwise analysis reveals that
globally-centered residuals encode topic far more strongly than role.
The structural fix lives outside the BGE-small embedding space.

## Practical implication for merken's `consolidate()`

The current `consolidate()` default merges v1+v2+v3 of one topic into
one fact, losing temporal evolution -- the bottleneck the issue set
out to attack. Three notes for follow-on work:

1. **Do not ship a PR-as-gate.** PR is the wrong functional of the
   eigenvalue spectrum to read from. The "PR median 2.7 contaminated
   vs ~3.8 clean" separation is real but goes the *opposite* of the
   intuition the gate would have to encode. Worse: PR detects the
   v1+v2+v3-of-one-topic shape that the consolidator already produces
   by default, so it cannot decide whether that merge is wrong.

2. **EVR_1 is the working geometric signal, but it is not "free."**
   EVR_1 reads the same eigenvalue spectrum as PR but in the
   predicted direction at partial-band AUC. The issue spec named it
   "consistency check, not independent evidence" of PR, so we cannot
   promote it to a gate within #38's pre-registration. Two specific
   risks before treating it as production-ready:

   a. **N confound is structural at the source data.** The
      contamination-target classes are pinned at N=4 by design;
      clean classes vary N=4-6. EVR_1 at N=4-controlled is even
      stronger (AUC 0.824) than across all sizes (0.760), but that
      is consistent with two stories: "EVR_1 captures
      role-mismatch" (the hypothesis) and "EVR_1 captures
      bimodality, which co-occurs with N=4 in this scenario family."
      Disjoint data and an explicit N-mode pre-registration are
      needed to distinguish the two.

   b. **A simple O(N) concentration metric might match EVR_1.** If
      `max(|residual_i|) / mean(|residual_i|)` (an O(N) outlier
      detector with no SVD) gives the same AUC, EVR_1 is doing
      nothing eigenvalue-specific and the spectrum interpretation
      collapses to "one event is far from the rest." Worth ruling
      out before claiming the eigenvalue spectrum carries the
      signal.

   These two risks are what #38b is set up to control for. See
   `notes/2026-04-28-residual-geometry-38b-followup-issue.md`.

3. **Geometric gates downstream of role still cannot recover role.**
   Both PR and EVR_1 are downstream of "events from the same topic
   embed close together." Without an upstream role tag, geometry
   cannot tell `(initial, refinement, reversal)` apart from
   `(initial, initial, initial)` if the three events happen to share
   a topic. This is why #39 (semantic markers) is pursued in
   parallel, not gated on #38b's outcome.

## Bug audit (after the inverted result raised user concern)

The inversion + opposite-direction pairwise finding was unusual enough
to warrant a focused bug audit before treating the result as a clean
negative. The audit looked specifically for sign-flipping or labeling
errors that could invert the result, not for general code quality.

**Eleven items checked:**

1. AUC label/score wiring (cluster-level): VERIFIED CORRECT.
   `is_contam = 1 - is_clean`, `roc_auc_score(labels=is_contam,
   scores=pr_vals)`. AUC > 0.5 iff contaminated has higher scores;
   observed 0.224 = data inverts the prediction.
2. Cluster construction labeling: VERIFIED CORRECT for clean_role_*,
   clean_noise, mixed_role_in_topic, mixed_role_decisions_no_noise.
3. **`mixed_topic_decisions_role_pure` labeling: BUG FOUND.** Original
   code labeled the role-pure subclass `is_clean=False` despite all
   members sharing one fine_role (different topics, same role = clean
   on the role axis). Effect: ~3% of one class (2 of 60 clusters per
   seed) injected clean-role clusters into the contaminated pool. The
   bias direction was *toward* the hypothesis (lowers contaminated PR
   slightly), so fixing it pushes AUC further from 0.5, not closer.
   Post-fix 3-seed PR AUC: 0.224 (mean), unchanged at the rounding
   we report.
4. Centroid + residuals (cluster_metrics): VERIFIED CORRECT. axis=0
   collapses rows, broadcasting (N,d)-(d,) -> (N,d).
5. SVD + PR + EVR_1 normalization: VERIFIED CORRECT. PR and EVR_1
   are both scale-invariant, so missing 1/N or 1/(N-1) normalization
   cannot affect them.
6. Pairwise residual computation: VERIFIED CORRECT.
7. Same-topic / same-fine-role mask pairing: VERIFIED CORRECT.
8. MW-U direction: VERIFIED CORRECT. `alternative="less"` directly
   tests the pre-registered prediction; observed p_two_sided=6.3e-80
   in the opposite direction means the data refutes H1.
9. Pairwise AUC sign: VERIFIED CORRECT. AUC ~0.045 is the correct
   numerical reflection of "data refutes hypothesis," not a sign
   error.
10. Embedder sanity: VERIFIED CORRECT. cos(cache, cache) = 0.765 >
    cos(cache, db) = 0.541 > cos(cache, noise) = 0.402. Sensible
    space.
11. No other potential sign-flippers identified.

**Conclusion of audit**: no sign error. The 0.224 cluster PR AUC and
the 4.6x pairwise inversion both correctly reflect the data given the
wiring. The one labeling bug is real but trivially small and biases
*toward* the hypothesis.

## N confound

PR is bounded by `min(N-1, d)`; with d=384 and N <= 6 in this study,
the bound is N-1. The two contamination-target classes are forced to
N=4 (each (topic, fine_role) cell holds 1 event in the data, so we
cannot grow the role-mixed-within-topic cluster beyond v1+v2+v3 + 1
extra). All other classes vary N=4 to 6. Mean N: clean ~5.0,
contaminated ~4.4.

This is a real confound: lower N -> lower PR ceiling. To rule it out,
we re-computed AUCs restricted to N=4 clusters only (the largest
fully-comparable subset). Result:

| Metric (N=4 only, 214 clusters) | AUC | 95% CI |
|---------------------------------|-----|--------|
| PR                              | 0.165 | [0.112, 0.225] |
| EVR_1                           | 0.824 | [0.768, 0.879] |

The PR inversion is *stronger* under size control (0.165 vs 0.224),
not weaker. The N confound exists and shifts numbers, but does not
explain the inversion -- if anything, it was masking how clean the
inversion is.

EVR_1 at N=4 controlled clears the partial band cleanly with lower CI
0.768. This does not rescue the primary gate (which is on PR by
pre-registration), but it documents that the eigenvalue spectrum
*does* carry signal, just oriented opposite the way PR reads it.

## What changed vs the issue spec

The script's module docstring lists three deviations forced by the
data (no role labels in `experiments/loop_quality/scenarios/`, no
N>=4 same-(topic, role) cells, no topic-tagged noise events). The
key consequence for interpretation:

- **mixed_role_in_topic was forced to N=4 by adding one random noise
  event.** Code review H3 flagged that this could confound role and
  noise contamination. The added `mixed_role_decisions_no_noise` class
  cleanly disambiguated: PR/EVR_1 are nearly identical with or without
  the noise event (PR 2.69 vs 2.73, EVR_1 0.479 vs 0.473). The signal
  is role-driven.

- **`mixed_topic_decisions_role_pure` was tracked** for accidental
  same-role realizations of the cross-topic random-decision class.
  Only 2 such clusters appeared across 60 attempts (rate ~3%); they
  do not change the AUC numbers.

- **Pairwise "same-topic same-fine-role" stratum was empty by
  construction** (each (topic, fine_role) cell has 1 event). The
  secondary gate as written ("MW-U on same-topic-same-role vs
  same-topic-different-role") was uncomputable; we substituted
  diff-topic-same-fine-role as the control. This change is documented
  in the script docstring and in the pairwise table above.

## Reproducibility

- `experiments/role_geometry/directional_residual_geometry.py` (CLI,
  default `--scenario knowledge_update_50topics.json`).
- Pre-fix 3 runs at `--seed {42, 43, 44}` produced
  `experiments/role_geometry/runs/20260428_1021*` directories
  (retained for diff against the post-fix runs).
- **Post-fix 3 runs** at `--seed {42, 43, 44}` produced
  `experiments/role_geometry/runs/post_audit_seed{42,43,44}/`. The
  numbers cited in this writeup are from the post-fix runs unless
  noted. Each directory contains `auc_summary.json` (with sha256 of
  scenario + vstash version + numpy version + bootstrap counts),
  `metrics_by_cluster.csv`, and a sampled `pairwise_role_geometry.csv`.
- 1 replication on 20topics (pre-fix): `runs/20260428_102411`. Same
  inversion direction; the labeling fix would not change the result
  because the same `mixed_topic_decisions_role_pure` accident applies
  at <= 3% rate.
- Embedder: BAAI/bge-small-en-v1.5 (dim 384) via vstash 0.35.0.
- Independent child generators (`rng.spawn(4)`) ensure cluster
  construction, cluster-level bootstrap, pairwise CSV sampling, and
  pairwise bootstrap have decoupled draws -- adding a stage downstream
  cannot retroactively shift earlier CIs.

## Files written

- `metrics_by_cluster.csv` -- 1 row per cluster, all metrics + topics +
  fine_roles. 140 clusters per seed run on 50topics.
- `pairwise_role_geometry.csv` -- subsample of the 605k pairs to
  200k rows (deterministic by seed); full arrays held in memory for
  the AUC/MW-U tests.
- `auc_summary.json` -- decision artifact. Contains every metric's
  AUC, the verdict, and full env metadata.
- `plots/hist_pr.png`, `plots/hist_evr_1.png`, `plots/hist_ang_disp.png`,
  `plots/scatter_pr_vs_evr_1.png`, `plots/scatter_pr_vs_ang_disp.png`
  -- generated for seed=42 only (others run with `--no-plot` for speed).

## Decision

Per pre-registered primary gate: **NO_GO on PR-as-gate.**

The eigenvalue spectrum carries partial-band signal in the predicted
direction when read via EVR_1 (3-seed AUC mean 0.760; N=4-controlled
0.824 with lower CI 0.768). EVR_1 was named a "consistency check, not
independent evidence" by the spec, so it cannot post-hoc lift the
gate verdict. But the finding is concrete enough to warrant a
pre-registered follow-up.

Three concrete next moves, in parallel:

1. **Issue #39 (semantic markers)** stays the active investigation,
   per the pre-committed pivot framing. Markers attack the upstream
   role-tagging gap that geometry cannot fill post-hoc.

2. **Issue #38b opens** as a pre-registered follow-up of #38: validate
   EVR_1-as-gate on disjoint data, with explicit pre-registration of
   (a) AUC threshold and CI bound, (b) N-control mode (fixed N=4 vs
   N-variable, reported separately because they validate different
   things), (c) two non-trivial baselines -- cluster size N alone,
   and a non-SVD concentration metric (e.g., max residual norm /
   mean residual norm) -- in addition to the original mean-cosine
   baseline. EVR_1 must beat all three by delta-AUC >= 0.10 to count
   as genuine eigenvalue signal rather than confounded structural
   artifact. Spec: `notes/2026-04-28-residual-geometry-38b-followup-issue.md`.

3. **Issue #40 (combined gate)** re-scopes to "EVR_1 (conditional on
   #38b passing) + role markers from #39." The original "entropy +
   markers" framing becomes "EVR_1 + markers" since the eigenvalue
   spectrum is the working geometric signal in the predicted
   direction, not entropy.

Issue #38 stays **open** with this writeup linked, awaiting #38b's
verdict before final disposition. Closing it now would discard the
EVR_1 finding without a clean follow-up to confirm or reject it.
