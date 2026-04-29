# Draft comment for issue #44 (benchmark monoculture meta)

Status: draft. Jay to review and post via `gh issue comment 44 -F notes/2026-04-28-issue-44-monoculture-comment-draft.md`
or paste the body below as a comment.

---

## New evidence: matched-shape eval inflates gates (role program, 2026-04-28)

The role program (#38, #38b, #39, #39b) re-validated this week against
production-shape inputs is a clean instance of the monoculture concern
this issue tracks -- not on benchmark axis, but on the **shape** axis
of the data the gate measures against. Surfacing because the dynamic
generalizes and the numbers are concrete.

### What was originally validated

Both #38 (residual geometry NO_GO) and #39b (V2 SCR semantic markers
STRONG, macro-F1 0.929) were measured on **sentence-level events**
(prototypes/eval median 74-97 chars). Both decisions were treated as
production-relevant.

### What re-validation at production shape showed

`notes/2026-04-28-role-program-prod-shape-validation.md` has the full
writeup. Headlines:

| signal | sentence-level (original) | production-shape (3-seed) | direction |
|---|---|---|---|
| #38 PR AUC | 0.211 | 0.95-1.00 | sign-flipped at production scale |
| #38 EVR_1 AUC | 0.78 (claimed primary) | 0.00-0.15 (anti-correlated) | sign-flipped |
| #38 baseline raw cosine | weak | **AUC 1.000** (ceiling) | dominated geometry signal at production |
| #39b V2 SCR macro-F1 | 0.929 +- 0.007 | 0.794 +- 0.000 | -13.5pp |
| #39b INV recall | 0.95 | **0.40** | INV->OBS collision |
| #39b 4-role STRONG verdict | YES | NO_GO | shape-leak |

Both NO_GO and STRONG verdicts moved when shape moved. The
geometric/marker signals are not noise -- they are real and
shape-dependent. Sentence-level evaluation gave a DIFFERENT
production-shape answer than production-shape evaluation.

This is the same failure mode #44 tracks for retrieval/builder
benchmarks (LME+LoCoMo as the only axis), now seen on the input-shape
axis.

### Salvage: production-shape signal that survives

Subset experiment (`notes/2026-04-28-role-classifier-subset-probe.py`,
output in `experiments/role_markers/runs/subset_20260428_182036/`):

| config | macro-F1 (3-seed mean +- stdev) | CI lower (worst seed) | verdict |
|---|---|---|---|
| 2-class (OBS, PREF) | **1.000 +- 0.000** | 1.000 | **STRONG_PROCEED** |
| 3-class (SCR, OBS, PREF) | **0.933 +- 0.000** | 0.822 | PARTIAL (SCR bleeds 20% to OBS) |
| 4-class baseline | 0.794 +- 0.000 | 0.658-0.677 | NO_GO_EMBEDDER_LACKS_ROLE_SIGNAL |

The classifier IS production-grade for the 2-role partition that
matters most operationally (subjective stance vs factual observation).
The 4-role failure was a taxonomy-level overreach, not a wholesale
embedder failure at production shape.

### Implication for the #44 protocol

The Tier 1 / Tier 2 / Tier 3 framework in this issue is the right
shape, but should explicitly include **input-shape disjointness** as
an axis distinct from benchmark-name disjointness. Concretely:

- A future #39-style classifier or #38-style geometric experiment
  whose authoring shape (say, sentence-level) does not match the
  production ingest shape (say, session blobs) **must report metrics
  at both shapes** before any production-grade verdict.
- The pre-registered "Tier 1 disjoint benchmark" should include the
  question "is the eval shape representative of the production input
  shape?" not only "is the eval domain disjoint from the training
  domain?"

For retrieval/builder decisions on LME+LoCoMo: granularity sweeps
(per-turn vs per-session) already address this de facto, because the
sweep itself is a shape disjointness check. The role program's
sentence-level vs production-shape failure suggests that any future
classifier or filter built into merken's substrate should adopt the
same discipline.

### What we are NOT proposing

Reopening #38, #38b, or #39b to re-author at every possible shape.
The verdicts are settled (NO_GO with corrected reasoning + STRONG
reverted). The salvage of 2-class is a smaller scope, not a return to
the full taxonomy.

### Concrete next-step ask for this issue

Add to the protocol section of #44 (or open a sibling issue):

> When a tuning experiment produces a STRONG / PARTIAL gate and the
> evaluation data shape (sentence vs paragraph vs session vs document)
> does not match the production ingest shape, the verdict is provisional
> until re-validated at production shape. Re-validation is required
> before any production default is changed.

This gives matched-shape eval its own checkbox in the gating flow,
parallel to the cross-benchmark axis #44 already establishes.

### Files referenced

- `notes/2026-04-28-role-program-prod-shape-validation.md`
- `notes/2026-04-28-role-program-audit-open.md` (the EOD thread that
  this re-validation closes)
- `experiments/role_geometry/production_shape_validation.py`
- `experiments/role_geometry/runs/prod_shape_20260428_175015/auc_summary.json`
- `experiments/role_markers/{prototypes,eval}_v3_production.json`
- `experiments/role_markers/runs/v3_prod_seed{42,43,44}/`
- `experiments/role_markers/runs/subset_20260428_182036/subset_results.json`
- `notes/2026-04-28-role-classifier-subset-probe.py`
