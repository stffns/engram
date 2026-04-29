# Role program audit -- production-shape re-validation

Date: 2026-04-28 EOD
Session: from-scratch re-validation of #38 and #39, both flagged as
"evaluated incorrectly" at the close of the prior audit.

## TL;DR

- **#38 NO_GO holds** but for a different reason than the original.
  The prior-audit interpretation ("residual geometry at N=4/d=384 is
  dominated by dimensionality") is **false** -- at production shape
  the geometry does carry content. What remains is that **baseline
  raw cosine reaches AUC 1.000** on the topic-coherence axis, so the
  residual geometry adds no operational value over a much cheaper
  signal. Verdict: NO_GO valid, with corrected reasoning.
- **#39 STRONG REVERTED to NO_GO** at production shape. macro-F1
  drops from **0.929 (sentence) to 0.794 (production)** consistent
  across 3 seeds; CI lower 0.658-0.677 < 0.70. New failure mode:
  **INV->OBS collision (recall=0.40)**, structurally parallel to V1's
  DEC->PREF collapse but on a different role pair.
- **Salvage**: 2-class subset (observation, preference) **passes
  STRONG production-grade**: macro-F1 1.000 across 3 seeds, perfect
  confusion matrix. 3-class drop_INV sits at PARTIAL (0.933, CI lower
  0.822). Investigation is the lost distinction.
- **Decision on PR #46 (`Memory(role_classifier=...)`)**: close
  without merge for the 4-class form. Open a separate issue to expose
  the 2-class taxonomy as `classify_role(taxonomy="obs_pref")`.

## Experiment construction (summary)

### #38 production-shape AUC validator

`experiments/role_geometry/production_shape_validation.py` reuses
`cluster_metrics` and `bootstrap_auc` from the original module. Builds
labeled clusters at production shape:

- Pos (clean / topic-coherent): N items from the **same** session
  (LME) or **same** conversation (LoCoMo).
- Neg (mixed / topic-incoherent): N items from N **distinct** sessions
  or convs.
- Neg cluster items disjoint from the pos pool to avoid AUC inflation
  from item-level overlap (LoCoMo only has 10 distinct convs).
- N is pool-specific due to structural constraints:
  - LoCoMo (10 convs, 19-32 sessions per conv): N in {4, 8}.
  - LME (948 namespaced sessions, median 12 turns): N in {4, 12}.
- 3 seeds (42/43/44), k_target=30 pos + 30 neg per cell, bootstrap
  resamples=1000. RNG hashed over (pool, N, seed) for cross-cell
  independence.

### #39 V3-prod classifier

- `experiments/role_markers/prototypes_v3_production.json` -- 4 roles
  (state_change_report, investigation, observation, preference), 5
  prototypes per role, median 783 chars.
- `experiments/role_markers/eval_v3_production.json` -- 40 events
  (10 per role), median 797 chars, 20 topics disjoint from the
  prototype topic pool (anti-leak verified).
- `experiments/role_markers/few_shot_classifier_v2.py` (unchanged)
  --k-list 1 3 5 --k-primary 5, 3 seeds (42/43/44), bootstrap=1000.

### Pre-flight code review

Code-reviewer subagent corrected the following before the run:

- BLOCKER: N=20 on LoCoMo crashed with only 10 groups -> N pool-specific.
- BLOCKER: V3 has 5 protos/role vs default --k-list 1 3 5 7 10 ->
  passed --k-list 1 3 5 --k-primary 5.
- WARNING: pos/neg item overlap on LoCoMo -> enforced disjointness.
- WARNING: RNG without pool/N awareness -> hashed seed.
- WARNING: `min_lo`/`max_hi` was misleading -> renamed to
  `worst_seed_lo`/`best_seed_hi`, added `stderr_3seed`.
- WARNING: silent qid fallback "q?" -> raise.
- WARNING: `proto_scr_04` contained "approved the policies"
  (V2-forbidden) -> rewritten as state-report.
- WARNING: `eval_scr_05` "is scheduled" modal-future -> rewritten.

## Results #38 -- LME-turn pool (median 482 chars)

| N | metric | mean(AUC) +- stderr | worst lo | best hi | verdict (3-seed) | beats baseline |
|---|---|---|---|---|---|---|
| 4  | pr        | **0.955 +- 0.006** | 0.890 | 0.999 | STRONG/STRONG/STRONG       | NO/NO/NO |
| 4  | evr_1     | 0.061 +- 0.011 | 0.002 | 0.157 | NO_GO/NO_GO/NO_GO       | NO/NO/NO |
| 4  | ang_disp  | 0.848 +- 0.056 | 0.601 | 0.980 | NO_GO/STRONG/STRONG     | NO/NO/NO |
| 4  | norm_var  | 0.227 +- 0.072 | 0.054 | 0.526 | NO_GO/NO_GO/NO_GO       | NO/NO/NO |
| 4  | baseline_neg_cos | **0.999** | -- | -- | -- | -- |
| 12 | pr        | **1.000 +- 0.000** | 1.000 | 1.000 | STRONG/STRONG/STRONG    | NO/NO/NO |
| 12 | evr_1     | 0.000 +- 0.000 | 0.000 | 0.007 | NO_GO/NO_GO/NO_GO       | NO/NO/NO |
| 12 | ang_disp  | 0.915 +- 0.018 | 0.801 | 0.991 | STRONG/STRONG/STRONG    | NO/NO/NO |
| 12 | norm_var  | 0.216 +- 0.047 | 0.046 | 0.429 | NO_GO/NO_GO/NO_GO       | NO/NO/NO |
| 12 | baseline_neg_cos | **1.000** | -- | -- | -- | -- |

## Results #38 -- LoCoMo-session pool (median 2808 chars)

| N | metric | mean(AUC) +- stderr | worst lo | best hi | verdict (3-seed) | beats baseline |
|---|---|---|---|---|---|---|
| 4 | pr        | **0.875 +- 0.039** | 0.680 | 0.984 | STRONG/PARTIAL/STRONG  | NO/NO/NO |
| 4 | evr_1     | 0.146 +- 0.052 | 0.024 | 0.373 | NO_GO/NO_GO/NO_GO      | NO/NO/NO |
| 4 | ang_disp  | 0.851 +- 0.031 | 0.667 | 0.969 | PARTIAL/PARTIAL/STRONG | NO/NO/NO |
| 4 | norm_var  | 0.242 +- 0.044 | 0.086 | 0.468 | NO_GO/NO_GO/NO_GO      | NO/NO/NO |
| 4 | baseline_neg_cos | **0.999** | -- | -- | -- | -- |
| 8 | pr        | **0.987 +- 0.004** | 0.950 | 1.000 | STRONG/STRONG/STRONG   | NO/NO/NO |
| 8 | evr_1     | 0.036 +- 0.011 | 0.000 | 0.121 | NO_GO/NO_GO/NO_GO      | NO/NO/NO |
| 8 | ang_disp  | 0.921 +- 0.028 | 0.764 | 0.999 | STRONG/STRONG/PARTIAL  | NO/NO/NO |
| 8 | norm_var  | 0.219 +- 0.030 | 0.082 | 0.419 | NO_GO/NO_GO/NO_GO      | NO/NO/NO |
| 8 | baseline_neg_cos | **1.000** | -- | -- | -- | -- |

## Reading #38

1. **PR escapes saturation at production shape.** At sentence-level
   it gave 0.211 (post_audit_seed42 baseline). At production it rises
   to 0.875-1.000 -- **opposite direction** to the original #38
   prediction (which expected high PR on contaminated; in reality
   topic-coherent pos clusters have higher PR).
2. **EVR_1 also flips.** AUC 0.000-0.146 (anti-correlated). The
   topic-coherent clusters concentrate residuals in fewer dominant
   directions. Originally #38 expected the opposite.
3. **AngDisp shows real signal (0.85-0.92)** in the correct direction
   at production scale.
4. **NormVar inverted (0.22-0.24)** -- pos clusters have higher norm
   variance than neg.
5. **CRITICAL: baseline raw cosine is at ceiling (AUC 0.999-1.000).**
   Topic-coherent vs topic-mixed clusters are trivially separable by
   raw cosine because the difference is topical, not structural.
   **Zero residual metrics pass the delta>=0.10 baseline-beat gate.**

**Meaning for #38**: the prior-audit claim ("residual geometry at
N=4/d=384 is dominated by dimensionality, not content") is
**refuted** -- at production shape the geometry does carry content.
But the detectable content reduces to topic coherence, and raw cosine
already captures it for free. **The original NO_GO survives with
corrected reasoning**: not because the geometry is inert, but because
its content is redundant with a much cheaper baseline.

**What this does NOT refute**: the original #38 prediction was
specifically about role-mixing detection (clean_role vs
mixed_role_in_topic). LoCoMo/LME do not have role labels, so this run
cannot directly test the original hypothesis. **What this run does
settle** is the audit's suspicion that the NO_GO might be shape-bias:
it is not; the numbers operationally confirm it.

## Results #39 -- V3-prod classifier (3-seed)

| seed | macro-F1 (point) | CI lower | CI upper | SCR rec | INV rec | OBS rec | PREF rec | Decision |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.794 | 0.658 | 0.896 | 0.80 | **0.40** | 1.00 | 1.00 | NO_GO_EMBEDDER_LACKS_ROLE_SIGNAL |
| 43 | 0.794 | 0.667 | 0.896 | 0.80 | **0.40** | 1.00 | 1.00 | NO_GO_EMBEDDER_LACKS_ROLE_SIGNAL |
| 44 | 0.794 | 0.677 | 0.901 | 0.80 | **0.40** | 1.00 | 1.00 | NO_GO_EMBEDDER_LACKS_ROLE_SIGNAL |

The K-saturation curve is uninformative because at K=5 (max
available) all prototypes are in use; the K=1/K=3 variation between
seeds is from the subset permutation only.

**Confusion matrix identical across all 3 seeds** (rows=true, cols=pred):

```
                SCR  INV  OBS  PREF
SCR              8    0    2    0
INV              0    4    6    0
OBS              0    0   10    0
PREF             0    0    0   10
```

## Reading #39

1. **macro-F1 falls 0.929 -> 0.794 from sentence to production
   shape.** The -13.5pp delta is consistent across seeds.
2. **CI lower 0.658-0.677 falls below the 0.70 threshold that
   separates PARTIAL from NO_GO** in the classifier's pre-registered
   gate -> NO_GO_EMBEDDER_LACKS_ROLE_SIGNAL.
3. **New failure mode: INVESTIGATION collapses into OBSERVATION
   (recall=0.40, 6/10 misclassified).** The V2 style guide separates
   investigation from observation by presence/absence of inquiry verbs
   ("investigating", "diagnosing" vs "the data shows", "we observed").
   At sentence-level shape this marker wins margin; at production
   shape the 700-900 chars dilate the lexical markers in context, and
   BGE-small loses them.
4. **STATE_CHANGE_REPORT also bleeds** (recall 0.80, 2/10 to
   OBSERVATION). The structural parallel to V1's "DECISION collapses
   into PREFERENCE" is exact: the lexically nearest neighbor role
   absorbs ambiguous cases.
5. PREFERENCE and OBSERVATION are perfect at 100% recall, but the
   cost is that they become sinks for the other two roles.

**Meaning for #39**: the V2 STRONG verdict from the prior session
(macro-F1 0.929 at sentence shape) was shape-bias. **The role markers
program with the V2 SCR taxonomy is not production-grade in BGE-small
on chat/dialogue-sized inputs.** The embedder does not preserve the
INV vs OBS distinction at production text length.

## Salvage: subset reduction recovers production-grade

After the 4-class NO_GO, a probe restricted to the role subset whose
markers do survive at production
(`notes/2026-04-28-role-classifier-subset-probe.py`, output in
`experiments/role_markers/runs/subset_20260428_182036/`):

| config | macro-F1 (3-seed mean +- stdev) | CI lower (worst seed) | verdict |
|---|---|---|---|
| 2-class (observation, preference) | **1.000 +- 0.000** | 1.000 | STRONG_PROCEED |
| 3-class (state_change_report, observation, preference) | **0.933 +- 0.000** | 0.822 | PARTIAL_MARKER_ASSISTED |
| 4-class full baseline | 0.794 +- 0.000 | 0.658-0.677 | NO_GO |

**2-class is production-grade STRONG**: BGE-small at production shape
**distinguishes factual observation from subjective preference with
macro-F1 1.000** consistently across 3 seeds, with a perfect
confusion matrix. This is the partition that matters most
operationally (credibility ranking: data vs opinion).

**3-class is PARTIAL**: adding STATE_CHANGE_REPORT introduces a 20%
bleed into OBSERVATION (8/10 SCR correct, 2/10 wrong). CI lower 0.822
falls below the 0.85 STRONG threshold but stays above the 0.70 NO_GO
threshold. Operable as an auxiliary signal with a confidence
threshold.

**What is salvaged**: the V2 SCR is not production-grade as a 4-class
classifier, but the pretrained embedder does preserve 2 (and possibly
3) distinctions at production shape. The lost distinction is
INVESTIGATION (open inquiry vs factual recording), which requires
more linguistic context than BGE-small retains in 700-900 chars.

## Decision on PR #46

PR #46 (`feature/role-classifier-memory-tagging`,
`Memory(role_classifier=...)`) opt-in tagging + `classify_role()`
helper was open pending this validation.

**Decision: close PR #46 without merge in its current 4-class form.**
Reasons:

1. The V2 SCR 4-class classifier at production shape reaches only
   0.794 macro-F1, with INV recall 0.40 -- not useful for operational
   tagging without strict domain restriction.
2. The only path to keep the feature 4-class would be a very visible
   restriction (e.g., RuntimeError if the average length of text
   passed to `classify_role()` exceeds some threshold). That would be
   a guardrail without a measured threshold and is exactly the kind
   of "feature gated by an arbitrary threshold" that CONSTITUTION
   avoids.
3. The PR's logic is not lost: `merken/role_classifier.py` (PR #43,
   merged) stays in repo for future experiments. Closing #46 only
   removes the opt-in integration in `Memory.remember` for the
   4-class taxonomy.
4. **Possible re-scope (not in this session):** the 2-class V2 SCR
   restricted to (observation, preference) passes STRONG. A separate
   PR exposing `classify_role(text, taxonomy="obs_pref")` with the
   restricted taxonomy is defensible. Close current #46 and open a
   new issue to pursue the 2-class form.

## Next steps (next session)

1. Close PR #46.
2. Close issue #39 with tag "shape-bias / NO_GO_PRODUCTION (4-class);
   2-class STRONG salvageable in separate issue".
3. Close issue #38 with tag "NO_GO_PRODUCTION via baseline-redundancy".
4. Open a new issue for the `classify_role(taxonomy="obs_pref")`
   2-class production-grade form.
5. Issue #44 (benchmark monoculture meta) stays open and is the right
   thread for the general lesson: "matched-shape eval inflates
   gates" should be added there. Comment draft in
   `notes/2026-04-28-issue-44-monoculture-comment-draft.md`.

## Files generated

- `experiments/role_geometry/production_shape_validation.py` -- new
  pos/neg AUC validator.
- `experiments/role_geometry/runs/prod_shape_20260428_175015/auc_summary.json`
  -- 12 cells of #38 (2 pools x 2 N x 3 seeds).
- `experiments/role_markers/prototypes_v3_production.json` -- 5
  prototypes per role at production shape.
- `experiments/role_markers/eval_v3_production.json` -- 40 events at
  production shape.
- `experiments/role_markers/runs/v3_prod_seed{42,43,44}/phase1_metrics.json`
  -- per-seed classifier outputs.
- `notes/2026-04-28-role-classifier-subset-probe.py` -- 2/3/4-class
  subset probe.
- `experiments/role_markers/runs/subset_20260428_182036/subset_results.json`
  -- subset probe output.
- `notes/2026-04-28-issue-44-monoculture-comment-draft.md` -- draft
  comment for issue #44.
- `notes/2026-04-28-role-program-audit-open.md` -- the prior thread
  open at 04-28 EOD; this writeup closes it.
