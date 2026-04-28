## Motivation

The clustering bottleneck documented in the merken paper (sec 7.3) shows that embedding similarity does not separate semantic roles within a thematically related topic.  Intra-topic similarity ranged 0.700-0.862 and cross-topic similarity 0.460-0.770 on the `knowledge_update` scenario; no single threshold cleanly separates same-topic-same-role from same-topic-different-role events.

**Hypothesis to test:** the role-mismatch signal already exists in the embedding geometry but lives in a dimension we are not measuring.  Specifically, the *directional spread* of vectors within a cluster (after centering) should be lower for clusters where all events share the same role, and higher for clusters contaminated with role-mixed events.

If the hypothesis holds, we have a cluster-quality gate that requires no training, no labels, no new dependencies -- only post-processing of vectors vstash already produces.

## Adjacent literature

- **Zep / Graphiti** -- external temporal knowledge graphs to handle supersession.
- **SmartVector** (Xu, [arXiv:2604.20598](https://arxiv.org/abs/2604.20598)) -- "Self-Aware Vector Embeddings for RAG: a neuroscience-inspired framework for temporal, confidence-weighted, and relational knowledge."  Augments embeddings with three properties (temporal validity, live confidence, graph-relational importance) and reports 62.0% vs 31.0% accuracy vs cosine RAG on a 258-vector / 138-query synthetic benchmark.
- **Orthogonality Constraint** (Zahn, Beton, Chana, [arXiv:2601.15313](https://arxiv.org/abs/2601.15313)) -- "Attention Is Not Retention."  Argues that semantic embeddings cluster similar concepts together by training, and therefore cannot achieve the orthogonal-key structure required for episodic retrieval; demonstrates "Semantic Interference" empirically on Wikipedia facts (rho=0.96), scientific data, and image embeddings.  Proposes Knowledge Objects with hash-based identity as the structural fix.

Field consensus across these three lines of work is that the fix requires *external* structure.  This experiment tests an orthogonal hypothesis: the latent geometry already contains the role signal, just not in the dimension that pairwise cosine measures.  If true, the Orthogonality Constraint claim is partially refuted -- not because semantic interference is absent at the cosine level (it is, the merken paper sec 7.3 numbers confirm it), but because the residual covariance carries a separate signal that cosine compresses away.

Parallel to vstash v2 paper sec 5 (disagreement mining as "byproduct of operation as training signal").

## Proposed experiment

Use the existing `knowledge_update` scenarios in `experiments/scenarios/` where role ground truth is known by construction (decision events vs investigation events vs noise).

Two complementary analyses, both pre-registered:

### A. Cluster-level metrics

For each cluster with N >= 4 events:

1. Embed all events with the active model (BGE-small-en-v1.5).
2. Compute centroid; subtract from each vector to get residuals.
3. Compute three metrics on the residual covariance / direction set.

**Primary metric: Participation Ratio**

```
PR = (Σ_i λ_i)² / Σ_i λ_i²
```

bounded between 1 (single direction) and the effective rank (uniform across directions).  Robust to rank-deficiency when N << dim, which is the regime we are in.

**Secondary metric (robustness check): Explained variance of leading axis**

```
EVR_1 = λ_1 / Σ_i λ_i
```

Note: PR and EVR_1 are correlated (both functions of the eigenvalue spectrum); reporting both is a consistency check, not independent evidence.

**Tertiary metric (independent dimension): Angular dispersion**

```
AngDisp = mean over all i,j in cluster of (1 - cos(residual_i, residual_j))
```

after L2-normalizing residuals.  Independent of the eigenvalue magnitudes, so disagreement between this and (PR, EVR_1) is informative about whether the signal is variance-shape or angle-distribution.

**Auxiliary metric (orthogonal axis: magnitude, not direction): Residual norm variance**

```
NormVar = Var_i(||residual_i||)
```

Reported separately because it measures intensity scatter rather than role mismatch; high NormVar is consistent with mixed event verbosity, not necessarily mixed roles.

### B. Pairwise within-cluster metrics

To increase statistical power and stratify by ground-truth tuples, also compute pairwise within each cluster:

- Cosine similarity, residual cosine similarity, residual angular distance.
- Stratify pairs by (same-role / different-role, same-topic / different-topic):
  - same-topic same-role
  - same-topic different-role  *(target signal: should look distinct)*
  - different-topic same-role
  - different-topic different-role

Pairwise gives ~N² points per cluster of size N, which makes the AUC estimate stable even with few clusters.

## Pre-registered hypotheses

Predicted directions (recorded **before** running the experiment):

| Cluster type                | EVR_1       | PR    | AngDisp | NormVar    |
|-----------------------------|-------------|-------|---------|------------|
| clean same-role             | high        | low   | low     | low/medium |
| contaminated same-topic     | medium/low  | high  | high    | medium     |
| different-topic (clustering failure) | variable | high | high  | variable   |

Terminology: "different-topic cluster" = a cluster where the underlying clustering algorithm incorrectly merged two distinct topics.  This is itself a clustering failure mode; the experiment includes it because it should also score high on contamination metrics, providing an additional positive class.

## Success criteria (decided before looking at results)

**Primary gate:** ROC-AUC of the primary metric (PR) as a binary classifier `clean=positive class, contaminated=negative class` on the cluster-level analysis.  Bootstrap 95% CI required (1000 resamples) since cluster-level N may be small.

| ROC-AUC (lower bound 95% CI) | Verdict |
|------------------------------|---------|
| > 0.80                       | strong signal, proceed to integration design |
| 0.65 - 0.80                  | partial signal, useful as one input to a combined gate |
| < 0.65                       | no go, pivot to semantic markers (approach 2) or external structure (Zep-style) |

**Secondary gate:** Pairwise analysis stratified by tuples should show distribution separation between `same-topic same-role` and `same-topic different-role` pairs (Mann-Whitney U test, p < 0.01 with effect size Cliff's δ > 0.3).  This validates the signal at a different granularity from the cluster-level AUC.

**Mandatory reporting alongside ROC-AUC:** PR-AUC for both gates, since real cluster populations are imbalanced (clean >> contaminated).

**Baseline to beat:** Mean intra-cluster cosine similarity, the metric that the motivation says fails.  Our metrics must outperform this baseline by a margin (delta AUC >= 0.10) to count as a real geometric finding rather than a relabeling.

## Implementation notes

- **N_min = 4** events per cluster.  Below this PR / EVR_1 are unstable and excluded from cluster-level analysis (still included in pairwise).  Report fraction of clusters excluded.
- **Dimensionality.** BGE-small produces 384-d residuals; clusters of size N ≤ 384 give rank-deficient covariance.  PR is robust to this (bounded by min(N-1, d)) but EVR_1 trends to 1 trivially for very small N.  Reporting min(N-1, d) alongside the metric makes this auditable.
- **L2 normalization.** Residuals are NOT L2-normalized for PR / EVR_1 (we want magnitude-aware variance), ARE L2-normalized for AngDisp (we want pure angular content).
- **Sample size for the gate decision.** Require >= 20 clusters in each ground-truth class.  If the scenarios produce fewer, augment by re-clustering with multiple seeds before deciding the experiment is complete.

## Out of scope

- Fixing the clustering algorithm itself.
- Training new models.
- Comparing against Zep / SmartVector head-to-head.
- Production integration into `consolidate()`.

If the experiment succeeds, those follow as separate issues -- explicitly: "role-contamination gate" wired into `consolidate()`, with a participation-ratio threshold that routes high-PR clusters to brief-generation or role-aware-split paths instead of consolidating into a retrieval fact.

## Deliverable

`experiments/role_geometry/directional_residual_geometry.py` plus a short markdown writeup under `notes/` containing:

- AUC numbers (ROC + PR) with bootstrap CIs, both cluster-level and pairwise.
- Mann-Whitney U test result for the pairwise stratified gate.
- Distribution plots: histogram of PR clean vs contaminated; histogram of EVR_1 clean vs contaminated; scatter PR vs EVR_1; scatter PR vs AngDisp.
- Comparison vs the baseline (mean intra-cluster cosine similarity).
- Go / no-go recommendation against the success criteria above.

Output files:

- `metrics_by_cluster.csv` (one row per cluster, all metrics + ground-truth label).
- `pairwise_role_geometry.csv` (one row per pair, stratification labels + metrics).
- `auc_summary.json` (ROC-AUC, PR-AUC, bootstrap CIs, decision).
- `notes/directional_residual_geometry.md` (writeup with go/no-go).

## If the experiment fails

Pre-committed framing: "Even higher-order geometry does not recover role structure; external structure is necessary."  This strengthens, rather than weakens, the merken paper's argument for graph-based or marker-based approaches.

## Related issues

- #39 (sibling): few-shot semantic markers using existing embedder as role classifier.  Tests an orthogonal approach to the same underlying problem (role contamination in consolidation clusters).
- #40 (depends on this and #39): combined entropy + marker gate for consolidation decisions.  Opens only if both #38 and #39 pass their individual gates.

## References

- merken paper section 7.3 (clustering bottleneck).
- merken paper section 7.6 (future work: role-aware clustering).
- vstash v2 paper section 5 (disagreement mining as parallel example of "byproduct of operation as training signal").
