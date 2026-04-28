## Meta issue for the merken paper -- geometric methods finding

Captures the consolidated empirical claim from #38 + #38b for inclusion
in the merken paper, likely as section 7.3 expansion or a new
subsection 7.6.x. This is a meta issue (no code, no experiment) -- the
deliverable is paper text grounded in the experiments already run.

## The empirical claim

> Direct geometric analysis of the residual covariance of cosine-
> clustered events succeeds at detecting one shape of cluster
> contamination ("tight cluster + 1 outlier event," AUC 0.92,
> encoder-invariant) and fails at detecting the role-mismatch shape
> that motivated the experiment (v1/v2/v3 versioned events of one
> topic, AUC indistinguishable from chance under role-permutation
> nulls). The two failures occur in the same residual eigenvalue
> spectrum read via different functionals; PR (the originally
> pre-registered metric) goes the wrong direction, EVR_1 and
> NormVar go the right direction at AUC ~0.75-0.92 depending on the
> contamination shape -- but a cheap O(N) outlier detector
> (MaxNormRatio = max||residual|| / mean||residual||) matches them
> within 0.05 AUC across all conditions tested. The eigenvalue
> interpretation collapses to outlier detection.

## Why this matters for the paper

The merken paper section 7.3 documents the clustering bottleneck (intra-
topic similarity 0.70-0.86 vs cross-topic 0.46-0.77, no clean threshold).
Section 7.6 lists "role-aware clustering" as future work. This finding
sharpens 7.6 in three ways:

1. **A specific class of geometric methods is ruled out**, not just
   "we haven't tried it yet." Pre-registered residual-covariance
   functionals (PR, EVR_1, NormVar, AngDisp) plus the O(N) baseline
   were tested and converge to "outlier detection at best."

2. **The shape merken needs to detect (temporal supersession of one
   topic) is shown to be invisible to all spectral functionals**
   tested. v2 (refinement) and v1 (initial) of one topic are equally
   close to the topic centroid; their residuals point in similar
   directions; no spectral statistic distinguishes them.

3. **The result is encoder-invariant.** Per-cell AUCs across BGE-small
   and multilingual MiniLM correlate at Pearson 0.992 over 72 cells.
   The conclusion is a property of the underlying text structure, not
   of any specific embedder. This is a stronger claim than the paper
   currently makes about embedding-space limitations -- it survives
   the swap from monolingual to multilingual encoders.

## Convergent evidence with adjacent literature

The merken paper already cites Zep / Graphiti, SmartVector
(arXiv:2604.20598), and "Attention Is Not Retention" (Zahn et al.,
arXiv:2601.15313) as advocating *external* structure for the same
problem. #38 + #38b add a fourth independent line:

| Source | Mechanism advocated | Argument |
|--------|----------------------|----------|
| Zep / Graphiti | Temporal knowledge graph | Supersession can't live in the embedding |
| SmartVector | Augmented embeddings | Three additional dims (temporal validity, confidence, graph) |
| "Attention Is Not Retention" | Hash-based identity | Semantic similarity != episodic identity |
| **#38 + #38b (this work)** | **External role signal (#39)** | **Eigenvalue spectrum collapses to outlier detection** |

The four lines converge on the same negative claim about pure cosine
geometry, each from a different empirical angle:

- Zep argues by construction (you can't represent "this fact replaced
  that fact" in a vector).
- SmartVector argues by gain (augmented embeddings beat cosine RAG
  62% vs 31% on a synthetic temporal benchmark).
- "Attention Is Not Retention" argues by structural correlation
  (rho=0.96 semantic interference on Wikipedia).
- #38 + #38b argues by direct test (residual eigenvalue spectrum
  fails, encoder-invariant, baseline-equivalent to a cheap outlier
  metric).

Each line is necessary and none is sufficient on its own. Together
they form a tight empirical case for the merken paper's thesis that
external role structure is required -- which is what merken's
upcoming #39 (semantic markers via few-shot prompting on the existing
embedder) provides.

## What to write for the paper

Suggested addition to section 7.6 (future work) or a new 7.7
(geometric ablations):

> **Geometric ablations (deferred to follow-up; preliminary results in
> [appendix or supplementary])**
>
> Before adopting external role structure, we asked whether the
> residual covariance of cosine-clustered events carries a role-
> contamination signal directly recoverable by spectral statistics.
> Two pre-registered experiments (#38, #38b) tested four functionals
> of the eigenvalue spectrum (participation ratio, top-eigenvalue
> ratio, angular dispersion, residual-norm variance) and a cheap
> outlier baseline (MaxNormRatio = max-residual-norm /
> mean-residual-norm). Results across two embedders (BGE-small-en,
> multilingual MiniLM-L12), three disjoint scenarios, and three
> seeds are summarized in [table]. The pre-registered primary
> metric (participation ratio) was inverted relative to the
> hypothesis. The most promising secondary metric (top-eigenvalue
> ratio) detected a specific contamination shape ("tight cluster +
> one outlier event") at ROC-AUC 0.92, but its signal was matched
> within delta-AUC < 0.05 by the cheap outlier baseline -- the
> eigenvalue interpretation collapses to outlier detection. The
> shape merken's clustering bottleneck actually produces (multiple
> versions of one topic, all close to the topic centroid) was
> invisible to every spectral functional tested. Consistent with
> Zep/Graphiti, SmartVector, and Zahn et al. (arXiv:2601.15313),
> we conclude that role identification requires external structure.
> Section 7.7 [or wherever] describes our chosen approach: few-shot
> semantic role markers on the existing embedder.

## Status

OPEN, no code work. Deliverable is paper text. Will fold into the
merken paper draft alongside #39's results, since the negative result
from #38/#38b motivates #39's positive direction. Should NOT be
published in isolation -- the paper needs both halves (failed
geometric attempt + working marker-based attempt) to make a complete
argument.

## References

- `notes/directional_residual_geometry.md` (#38 writeup with
  inversion finding).
- `notes/role_geometry_38b.md` (#38b writeup with NO_GO + per-class
  breakdown + cross-embedder convergence).
- merken paper section 7.3 (clustering bottleneck).
- merken paper section 7.6 (current "future work: role-aware
  clustering").
- vstash v2 paper section 5 (parallel example of "byproduct of
  operation as training signal").
