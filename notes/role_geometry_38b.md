# #38b -- EVR_1 validation on disjoint data: results

Pre-registered follow-up to #38, run 2026-04-28 on
`feature/role-geometry-residuals`. Spec at
`notes/2026-04-28-residual-geometry-38b-followup-issue.md`.

## Verdict

**NO_GO across all 3 scenarios x 2 N-modes x 2 embedders x 3 seeds (36
cells). Drop EVR_1 as a gate. Pre-committed pivot to issue #39
(semantic markers) is now the sole active geometric-substrate
investigation.**

The pre-registered baseline `MaxNormRatio = max(||residual_i||) /
mean(||residual_i||)` matches or beats EVR_1 in every cell. EVR_1's
"eigenvalue spectrum interpretation" collapses to outlier detection.

## Joint dispositions

Two embedders (BGE-small-en-v1.5, multilingual-MiniLM-L12-v2) x three
disjoint scenarios:

| scenario | BGE joint | MiniLM joint |
|----------|-----------|--------------|
| `knowledge_update_hard.json` (5 topics, 108 events) | NO_GO | NO_GO |
| `scenarios_disjoint/products.json` (30 cross-domain topics, 690 events) | NO_GO | NO_GO |
| `scenarios_disjoint/medical.json` (30 medical-domain topics, 690 events) | NO_GO | NO_GO |

Per-cell decisions (3-seed majority): all are `NO_GO_TIES_BASELINE` or
`NO_GO`. No cell reaches `STRONG` or `PARTIAL`.

## Headline numbers (3-seed mean AUC across the 3 gate scenarios)

### BGE-small-en-v1.5

| metric           | N=4 fixed | N variable (4-6) |
|------------------|-----------|-------------------|
| EVR_1 (primary)  | 0.770     | 0.796             |
| **MaxNormRatio (baseline)** | **0.786** | 0.743 |
| NormVar          | 0.783     | 0.741             |
| PR               | 0.229     | 0.201             |
| AngDisp          | 0.226     | 0.565             |
| N alone          | --        | 0.315             |
| baseline_neg_cos | 0.369     | 0.370             |

### multilingual-MiniLM-L12-v2

| metric           | N=4 fixed | N variable (4-6) |
|------------------|-----------|-------------------|
| EVR_1 (primary)  | 0.780     | 0.813             |
| **MaxNormRatio (baseline)** | **0.792** | 0.760 |
| NormVar          | 0.785     | 0.768             |
| PR               | 0.222     | 0.180             |
| AngDisp          | 0.213     | 0.541             |
| N alone          | --        | 0.315             |
| baseline_neg_cos | 0.330     | 0.321             |

EVR_1 vs MaxNormRatio delta-AUC (3-seed mean):

| condition | BGE | MiniLM |
|-----------|-----|--------|
| N=4 fixed | -0.016 | -0.012 |
| N variable | +0.053 | +0.053 |

Pre-registered threshold: EVR_1 must beat MaxNormRatio by delta-AUC
>= 0.10. Best observed: +0.053 in N-variable mode, **half the required
margin**. Worst: MaxNormRatio is *better* by ~0.015 in N=4 fixed mode.

## Discovery-scale replication (sanity check, not part of the gate)

Re-ran the discovery scenario `knowledge_update_50topics.json` (1100
events, 50 topics) with proper a-priori N=4 cluster construction (vs
#38's post-hoc filter to N=4 from the N-variable build):

| metric         | N=4 fixed (a-priori) | N variable (4-6) |
|----------------|----------------------|-------------------|
| EVR_1          | 0.710                | 0.797             |
| **MaxNormRatio** | **0.748**           | 0.749             |
| PR             | 0.283                | 0.191             |
| NormVar        | 0.740                | 0.752             |

Delta EVR_1 vs MaxNormRatio: **-0.038** (fixed), **+0.048** (variable).

Important correction to #38's record: the writeup reported EVR_1 at
N=4-controlled AUC = 0.824 [0.768, 0.879]. That number came from
post-hoc filtering of the N-variable build to clusters that happened
to land at N=4. With proper a-priori N=4 construction, EVR_1 drops to
0.710. The original #38 finding inflated EVR_1 by ~0.11 due to the
sampling difference. The corrected number falls below the partial
band threshold (0.65) lower CI even before considering baselines.

## What this means

The hypothesis from #38 (eigenvalue spectrum of residual covariance
distinguishes role-mixed clusters from same-role clusters) **does not
generalize**. Specifically:

1. **Cross-scenario:** EVR_1 fails to beat MaxNormRatio on three
   distinct scenarios (a smaller knowledge_update variant + two
   freshly-generated cross-domain synthetic scenarios in software
   products and medical decisions).

2. **Cross-embedder:** the failure replicates on a structurally
   different embedder (multilingual MiniLM), so it is not a BGE-small
   artifact.

3. **Cross-N-mode:** the failure replicates whether N is fixed at 4
   or allowed to vary 4-6.

4. **Discovery-scale corrected:** the originally reported EVR_1 N=4
   "controlled" AUC of 0.824 was a sampling artifact. With proper
   a-priori construction it drops to 0.710 -- the same band as
   MaxNormRatio.

5. **All three "spectrum" metrics are interchangeable:** EVR_1,
   NormVar, and MaxNormRatio cluster together at AUC ~0.75 across
   conditions. They are reading the same underlying property: "this
   cluster has one event far from the other three." That property is
   real and detectable, but it is not eigenvalue-specific -- a single
   division (`max_norm / mean_norm`) captures it equivalently.

The signal in #38 was not "residual covariance encodes role
mismatch." The signal was "the constructed contamination class
(v1+v2+v3+1 outsider) is dominated by the outsider's distance to the
v1/v2/v3 cluster center, and *any* outlier-detection metric picks
that up." This is a feature of how the contamination class is
*constructed*, not a property of role mismatch per se.

## Why this is the result the pre-registration was designed to find

The MaxNormRatio baseline was added to #38b explicitly because Jay
asked "is the EVR_1 signal eigenvalue-specific or just outlier
detection." Pre-committing the baseline before running the experiment
turned what would have been a "follow-up shows partial signal" result
(if we only compared EVR_1 to mean-cosine and N-alone) into a clean
"the signal is not eigenvalue-specific" result.

If we had skipped the baseline:

- EVR_1 lower CI on `knowledge_update_hard` BGE N-variable: 0.692
- That clears the partial band (0.65)
- We would have shipped EVR_1 as a partial-band gate input to #40

The baseline forced the question: is EVR_1 doing anything that a
single division does not? The answer is "no," and the gate decision
is now correct.

## Should we run Eje 3 (real data)?

Pre-registered as a stretch goal: re-run on real organic data to rule
out generation-bias. **Skipped.** The structural finding -- EVR_1 ==
MaxNormRatio at AUC ~0.75 -- is dataset-independent. Both metrics are
computed from the *same* eigenvalue spectrum of the *same* residual
matrix; their numerical equivalence is a property of the spectrum
shape, not of the data domain. Real-data validation cannot change
this. If Eje 3 ran, we would expect:

- EVR_1 and MaxNormRatio AUCs to track each other (high confidence
  from structural argument).
- The gap delta-AUC to remain below 0.10 (high confidence given the
  consistency across 4 cross-cutting axes already tested).

The cost of Eje 3 (~4-5 hours of pipeline work to extract
role-versioned events from git history or LongMemEval) buys no
additional decision-relevant information. Documenting the skip with
its rationale rather than running it.

## Implications for downstream issues

- **Issue #38 stays closed-with-followup.** The original primary gate
  failed; the secondary EVR_1 finding does not survive #38b's
  baseline check; the eigenvalue-spectrum hypothesis is falsified.
  Update #38 status accordingly.

- **Issue #39 (semantic markers)** is now the sole active
  geometric-substrate investigation. Pre-committed pivot framing from
  #38 stands without revision.

- **Issue #40 (combined gate)** re-scopes again. The original plan
  was "entropy + markers." After #38, that became "EVR_1 + markers."
  After #38b, that collapses to "markers alone" since no geometric
  signal beats a single division. If a cheap geometric add-on is
  desired, MaxNormRatio is the candidate (at AUC ~0.75 it is a real
  but not-strong signal; could pair with markers as a low-cost
  pre-filter). But this is conditional on #39 producing a strong
  marker-based gate first.

## What changed vs the issue spec

The pre-registered spec at
`notes/2026-04-28-residual-geometry-38b-followup-issue.md` named a
"secondary transfer set" of real-content scenarios (jay_vstash,
analytics_project, session_2026_04_09). Inspection on day-of-run
showed all three lack the role-versioned id pattern (`_v1/_v2/_v3` or
`_a/_b`) that the experiment's role-derivation logic requires. The
secondary transfer set was therefore replaced with two newly-generated
synthetic scenarios in disjoint content domains (`products.json`,
`medical.json`), built via a deterministic generator
(`experiments/role_geometry/scenarios_disjoint/gen.py`) from
hand-written topic vocabularies. The vocabularies are committed to
git for inspection.

This change is a **scope expansion, not a relaxation**: the original
spec planned 1 secondary scenario chosen from 3 candidates; what we
ran is 2 synthetic scenarios in disjoint domains plus the primary
`knowledge_update_hard`, totaling 3 scenarios. Cross-embedder
robustness was added on top.

## Audit + null sanity check

After the run, a focused code-reviewer pass audited
`evr_1_validation.py` and `gen.py` for any bug that could manufacture
the EVR_1 == MaxNormRatio parity. Eleven hypotheses checked:

- MaxNormRatio formula correct (sanity: outlier cluster gives 1.309
  vs clean 1.039).
- EVR_1 formula correct (imported from already-audited
  `directional_residual_geometry.py`).
- `build_synthetic_clusters` honors `k_min=k_max=4` in fixed mode;
  same function as #38, no fork.
- delta-AUC sign convention correct (`primary - baseline`,
  beats >= 0.10).
- Baseline mapping consistent across `metric_arrays`, `aucs`, and
  `delta_aucs` keys.
- Bootstrap is sound paired stratified resampling at n=1000.
- `--embed-model` flag plumbed through to `embed_texts`.
- Generator produces id `<topic>_v1/v2/v3` with shared topic field --
  the data contract, not a leak.
- N=4 a-priori (#38b) vs N=4 post-hoc filter (#38) is a structural
  change, not a bug. The difference explains the AUC gap between
  #38's 0.824 and #38b's 0.710 at the discovery scale.
- Discovery-scale snippet rng schedule differs from #38's commit
  bit-by-bit but is internally consistent.

Audit conclusion: no bug found. Both metrics implement on the same
residual vectors, with the same N distribution, in the same vector
space, compared via paired bootstrap.

**Label-permutation null** confirms the AUCs reflect real signal,
not artifact. Across 3 seeds on `knowledge_update_hard` (BGE, N=4
fixed), 100 permutations of `is_contam` per seed:

| seed | EVR_1 real | EVR_1 null mean +- stdev | MaxNR real | MaxNR null mean +- stdev |
|------|------------|---------------------------|------------|---------------------------|
| 42   | 0.770      | 0.499 +- 0.055            | 0.786      | 0.500 +- 0.056            |
| 43   | 0.748      | 0.501 +- 0.052            | 0.764      | 0.496 +- 0.049            |
| 44   | 0.718      | 0.497 +- 0.047            | 0.736      | 0.496 +- 0.048            |

Under permuted labels, both metrics center on AUC = 0.50 with stdev
0.05, as expected under H0. Max observed AUC across 300 permutations
combined: 0.632, within the noise envelope for n=140 samples. Real
AUCs (~0.75) are >= 4 standard deviations above the null mean. The
near-equivalence of EVR_1 and MaxNormRatio is therefore a structural
property of the eigenvalue spectrum, not a numerical artifact.

## Reproducibility

- Code: `experiments/role_geometry/evr_1_validation.py` (~400 LOC).
  Imports cluster construction + metrics from
  `directional_residual_geometry.py`.
- Generator: `experiments/role_geometry/scenarios_disjoint/gen.py`,
  `vocab_products.json`, `vocab_medical.json`,
  `vocab_noise_tech.json`, `vocab_noise_clinical.json`. All committed.
- Generated scenarios: `scenarios_disjoint/generated/products.json`,
  `medical.json`. **Not committed** (regeneratable from gen.py + the
  vocabularies; sha256 in `aggregate_summary.json` covers
  reproducibility).
- Runs: `experiments/role_geometry/runs/38b_20260428_114039` (BGE)
  and `38b_cross_embedder_minilm` (multilingual MiniLM). Gitignored.
- Discovery-scale sanity check inline above; the script that produced
  it is captured in this writeup as a Python snippet (no separate
  artifact).
- 3 seeds (42, 43, 44), 1000 bootstrap resamples, k_target=20,
  K_min=K_max=4 in N-fixed mode, K_min=4 K_max=6 in N-variable mode.
- Independent `rng.spawn(2)` per N-mode within each seed so adding
  one mode does not shift the other's draws.

## Decision

**NO_GO. Drop EVR_1 (and by extension PR, NormVar, AngDisp) as a
geometric gate for `consolidate()`.** The eigenvalue spectrum of the
residual covariance does not carry role-mismatch signal beyond what
a single max-norm division captures. #39 (semantic markers) is the
remaining geometric/embedding-based investigation; #40 re-scopes to
"markers alone" or "markers + MaxNormRatio low-cost prefilter" if a
geometric add-on is wanted.
