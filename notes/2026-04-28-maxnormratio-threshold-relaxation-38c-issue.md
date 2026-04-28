## Motivation (deferred 2026-04-28; revisit if the underlying problem appears)

Per-class analysis of #38b's data revealed that MaxNormRatio (the cheap
O(N) baseline that beat EVR_1) detects the specific shape "tight cluster
+ 1 stray event" at ROC-AUC 0.92 across 3 disjoint scenarios, encoder-
invariant (Pearson 0.992 between BGE and multilingual MiniLM per-cell
AUCs). This is a real, structural signal -- but it was not actionable for
#38's stated goal (role-mismatch detection in v1+v2+v3 same-topic
clusters), because role-mismatch produces no outlier and MaxNormRatio
sees nothing.

This issue captures the *next* place that MaxNormRatio's signal could
become useful, deferred until the underlying problem actually appears in
merken's production clustering: **enabling threshold relaxation in
`cluster_by_embedding`**.

## Hypothesis to test (when the time comes)

`cluster_by_embedding`'s default threshold of 0.70 with complete linkage
is conservative -- it produces 100% cluster purity on all three
loop_quality scenarios at the cost of leaving related events unclustered.
Lowering the threshold to 0.65 or 0.60 would merge more events
(potentially capturing topic-related events that the current threshold
splits) but would introduce contamination. **MaxNormRatio at the
relaxed threshold could act as a safety net, recovering purity at the
lower threshold without re-fragmenting clusters.**

Net effect if it works: clustering becomes more inclusive (better
recall) without losing precision (MaxNormRatio filters out the strays
introduced by the relaxation).

## Why this is deferred, not immediately runnable

**The underlying problem may not exist.** At threshold 0.70, complete
linkage already achieves 100% cluster purity on the existing scenarios
(per merken project memory and `experiments/loop_quality/RESULTS.md`).
If lower thresholds also stay at 100% purity, MaxNormRatio has nothing
to detect and the safety-net role is moot.

**Cheap sondeo to gate this issue (~30 minutes when revisited):**

1. Run `cluster_by_embedding` on `knowledge_update_50topics.json` at
   thresholds 0.55, 0.60, 0.65, 0.70 (with complete linkage).
2. Measure cluster purity at each threshold (fraction of clusters
   where all members share a topic, plus mean topic-purity per cluster).
3. **Decision tree:**
   - If purity stays >= 95% at all thresholds: the problem this issue
     attacks does not exist. Close.
   - If purity drops below 95% at relaxed thresholds: open the
     follow-up experiment (see "Pre-registration" below).

If the sondeo passes, then proceed with the formal #38c-extended
experiment.

## Pre-registration (for when the sondeo passes)

### Hypothesis
At thresholds where purity drops, `relaxed_threshold + MaxNormRatio`
filtering achieves purity comparable to the baseline (threshold 0.70)
while clustering more events (higher recall).

### Data
- Disjoint from #38b: probably the loop_quality scenario family
  (`session_2026_04_09.json`, `analytics_project.json`,
  `jay_vstash_2026_04_09_snapshot.json`) since these have ground-truth
  role/topic labels via the curated scenario format. Pre-commit
  selection.

### Metric: Pareto curve
For each threshold T in {0.55, 0.60, 0.65, 0.70}:
- (recall, purity) pair for the unfiltered baseline at T
- (recall, purity) pair after MaxNormRatio filter applied to T's
  clusters

Pre-registered gate: at some T < 0.70, the filtered (recall, purity)
must dominate the unfiltered (recall, purity) at T = 0.70. "Dominate"
means: recall_T_filt > recall_0.70_unfilt AND purity_T_filt >=
purity_0.70_unfilt.

### Pre-committed thresholds
- MaxNormRatio threshold for "stray event" flag: pre-commit before
  looking at the data, e.g. > 1.5 (chosen from #38b's per-cluster
  distributions where contaminated clusters had max_norm_ratio > 1.7
  median).
- Action on flag: drop the highest-norm event from the cluster (or
  split it off) and re-evaluate purity.

### Out of scope
- The original role-mismatch problem (orthogonal -- this issue is
  about over-merging, not over-staying).
- Other geometric metrics (PR, EVR_1, NormVar already shown to be
  redundant with MaxNormRatio for this signal).

## Why "open and defer" rather than "open and run now"

Three reasons:

1. The sondeo might close it cheaply. 30 minutes of measurement could
   show the relaxed thresholds are also 100% pure on existing data,
   killing the issue without expensive follow-up.

2. The current consolidator's behavior is well-understood and isn't
   actively producing the failure mode this issue would address. The
   forward path of merken's roadmap (after #39 semantic markers, after
   #40 combined gate) is more pressing.

3. It is genuinely an *insurance issue*: it pays off only when the
   underlying problem appears, and the cost of letting it sit open
   is zero.

## Triggers to revisit

- A change to merken's clustering substrate that lowers cluster purity
  in production (e.g., switching to single linkage, lowering threshold
  by policy, adopting a new embedder with different cosine
  distribution).
- A real-data scenario showing the over-merging failure mode (clusters
  that span topics).
- Issue #39 (semantic markers) producing a working role-mismatch gate
  that frees us to relax cosine clustering.
- A user request to consolidate larger event horizons that would
  require thresholds below 0.70.

## References

- Parent finding: `notes/role_geometry_38b.md` section "Per-class
  breakdown" (MaxNormRatio AUC 0.918 on the v1+v2+v3+1stray shape).
- Cross-embedder convergence evidence: same writeup section
  "Cross-embedder convergence."
- Underlying mechanism: MaxNormRatio = max(||residual_i||) /
  mean(||residual_i||) reads "is one event in this cluster far from
  the centroid." This is the structural property the eigenvalue
  spectrum was reading via PR/EVR_1, so all four metrics are
  interchangeable; MaxNormRatio is the cheapest implementation.

## Status

DEFERRED. No work scheduled. Sondeo (~30 min) should run before any
formal pre-registration if the issue is reactivated. Created from a
spinoff conversation on 2026-04-28 EOD after closing #38 + #38b
NO_GO.
