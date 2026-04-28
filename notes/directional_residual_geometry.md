# Directional residual geometry as a role-contamination signal -- results

Pre-registered experiment for merken issue #38. Run 2026-04-28 on
`feature/role-geometry-residuals`.

## Verdict (per pre-registered primary gate)

**NO_GO on the primary metric. Pivot to issue #39 (semantic markers).**

Cluster-level PR ROC-AUC across 3 seeds on
`knowledge_update_50topics.json` (1100 events, 50 decision topics x
{initial, refinement, reversal}, 950 noise events):

| seed | AUC mean | 95% CI lower | 95% CI upper |
|------|----------|--------------|--------------|
| 42   | 0.214    | 0.135        | 0.303        |
| 43   | 0.260    | 0.172        | 0.362        |
| 44   | 0.199    | 0.118        | 0.284        |
| **3-seed** | **0.224 +- 0.032** | **0.118 (worst lo)** | **0.362 (best hi)** |

Pre-registered thresholds: lower CI > 0.80 (strong), 0.65-0.80
(partial), < 0.65 (no-go). Worst-case lower CI = 0.118, well below
0.65. Best-case upper CI = 0.362 -- still below 0.5.

**Baseline-beat**: mean intra-cluster cosine ROC-AUC = 0.287 +- 0.050
(also inverted). delta-AUC over baseline = -0.063 (PR is *worse* than
the baseline it was supposed to beat). Pre-registered requirement
delta-AUC >= 0.10 not met.

**Replication on `knowledge_update_20topics.json`** (440 events, 20
topics): PR AUC = 0.199 [0.120, 0.287]. Same direction, tighter
magnitude. Negative result is structural, not data-specific.

## What actually happened: the hypothesis was inverted

The issue predicted: contaminated (role-mixed) clusters have **higher**
PR than clean (same-role) clusters because role-mismatch adds independent
variance to the residual covariance.

The data shows the opposite. Per-class PR medians (3-seed average):

| cluster_class                       | n_clusters | PR    | EVR_1 | direction |
|-------------------------------------|------------|-------|-------|-----------|
| clean_role_initial                  | 60         | 3.88  | 0.307 | clean     |
| clean_role_refinement               | 60         | 3.82  | 0.326 | clean     |
| clean_role_reversal                 | 60         | 4.04  | 0.300 | clean     |
| clean_noise                         | 60         | 3.52  | 0.371 | clean     |
| mixed_topic_decisions               | 58         | 4.06  | 0.295 | contam    |
| mixed_topic_decisions_role_pure     | 2          | 3.38  | 0.358 | contam    |
| **mixed_role_decisions_no_noise**   | 60         | **2.69** | **0.479** | **contam (target)** |
| **mixed_role_in_topic**             | 60         | **2.73** | **0.473** | **contam (target)** |

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
points the predicted direction:

| Metric | 3-seed AUC mean | 3-seed stdev | direction relative to issue prediction |
|--------|-----------------|--------------|----------------------------------------|
| PR (primary)        | 0.224 | 0.032 | **inverted** |
| EVR_1 (secondary)   | 0.759 | 0.030 | predicted, partial-band signal |
| AngDisp (tertiary)  | 0.618 | 0.050 | predicted, weak |
| NormVar (auxiliary) | 0.752 | 0.024 | predicted, partial-band signal |

EVR_1 (0.759 mean, lower CIs in [0.626, 0.698] across seeds) clears the
partial band on two of three seeds and lands at the 0.65 boundary on
seed 43.

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

1. **Do not ship a PR-as-gate.** PR threshold "below X = contaminated"
   would technically work as a filter (PR median 2.7 vs ~3.8 has clean
   separation in this data) but the rule is brittle: it requires the
   cluster being v1+v2+v3-of-one-topic-shaped, which is exactly the
   shape the consolidator already produces by default. PR detects the
   structure that *defines* a successful merge in the current system,
   so it cannot decide whether that merge is wrong.

2. **EVR_1 is a tempting but redundant signal.** EVR_1 = 0.48 vs ~0.30
   is real, but it is the same eigenvalue information PR has, viewed
   from the other end. It does not add an independent dimension; it
   restates the same finding (topic-tight cluster with one outlier =
   bimodal eigenvalue spectrum).

3. **Geometric gates need a role *signal*, not a role *consequence*.**
   Both PR and EVR_1 are downstream of "events from the same topic
   embed close together." Without an upstream role tag, the geometry
   cannot tell `(initial, refinement, reversal)` apart from
   `(initial, initial, initial)` if the three events happen to share
   a topic.

The pre-committed pivot: **issue #39 (semantic markers via few-shot
prompting on the existing embedder)**. Markers add the upstream role
signal that geometry cannot recover post-hoc.

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
- 3 runs at `--seed {42, 43, 44}` produced
  `experiments/role_geometry/runs/20260428_*` directories. Each
  contains `auc_summary.json` (sha256 of scenario + vstash version +
  numpy version + bootstrap counts), `metrics_by_cluster.csv`, and a
  sampled `pairwise_role_geometry.csv`.
- 1 replication on 20topics: `runs/20260428_102411`.
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

Per pre-registered gate: **NO_GO on PR-as-gate. Pivot to issue #39.**

The experiment also produced a secondary partial-signal finding
(EVR_1 AUC ~0.76) but it does not lift the verdict for two reasons:
(1) the pre-registration named PR as primary, (2) EVR_1 is mathematically
coupled to PR and adds no independent dimension. Recording the EVR_1
finding here for the record; not opening a follow-on issue on it.

Issue #38 will be closed with this writeup linked. Issue #39 (semantic
markers) is now the active investigation. Issue #40 (combined gate)
remains gated on #38+#39 and will need re-scoping given that #38 did
not produce a useable geometric signal.
