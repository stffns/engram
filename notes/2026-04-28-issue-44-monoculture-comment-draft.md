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

### Apparent salvage that did NOT survive (issue #47, 2026-04-29 update)

Subset experiment on the V3-prod data
(`notes/2026-04-28-role-classifier-subset-probe.py`, output in
`experiments/role_markers/runs/subset_20260428_182036/`):

| config | macro-F1 (3-seed mean +- stdev) | CI lower (worst seed) | verdict on V3-prod |
|---|---|---|---|
| 2-class (OBS, PREF) | **1.000 +- 0.000** | 1.000 | STRONG_PROCEED on V3-prod |
| 3-class (SCR, OBS, PREF) | **0.933 +- 0.000** | 0.822 | PARTIAL (SCR bleeds 20% to OBS) |
| 4-class baseline | 0.794 +- 0.000 | 0.658-0.677 | NO_GO_EMBEDDER_LACKS_ROLE_SIGNAL |

Issue #47 was filed to ship the 2-class form as a production-grade
classifier and added an explicit acceptance criterion (AC#2) that
required production-shape validation on **real benchmark chat**:
100 LoCoMo per-session + 100 LongMemEval per-turn samples, oracle-
labeled with `gpt-oss-120b` on Cerebras into
{observation, preference, neither}, gated at macro-F1 >= 0.85 on the
in-taxonomy subset.

AC#2 hard-FAILED:

| pool | n in-tax | n neither | macro-F1 in-tax | rec(obs) | rec(pref) |
|---|---|---|---|---|---|
| LoCoMo per-session | 99 | 1 | **0.290** | 0.39 | 0.25 |
| LME per-turn | 75 | 25 | **0.444** | 0.09 | 0.95 |
| pooled | 174 | 26 | **0.471** vs 0.85 gate | 0.31 | 0.89 |

Failure mode: the V3-prod prototypes are **matched on length** (700-
1200 chars) but stylized on **content**. Observation prototypes are
ops log/metric paragraphs ("Logs show ...", "metrics indicate ...");
preference prototypes are first-person ops opinions ("I lean toward
..."). BGE-small reads real first-person biographical chat (LoCoMo
life-event recaps, LME assistant advice turns) as closer lexically
to the preference prototypes than to the observation prototypes,
collapsing observation recall to 31% and predicting "preference"
77% of the time on real chat.

Confidence margins on this distribution **never exceed 0.10** -- the
0.10 reject threshold from AC#3 would suppress 199/200 predictions.
There is no high-confidence subset to salvage.

Issue #47 was closed NO_GO. Independent code-review of the AC#2
path + manual confusion-matrix cross-check on
`runs/extension_20260428_*/all_rows.json` reproduced 0.471 exactly
-- no implementation bug. See
`notes/2026-04-29-issue-47-ac2-failure.md` for the full writeup.

### What this means for #44

The role program now has TWO independent shape-axis failures:
**4-class collapsed sentence -> production**, and the 2-class salvage
that PASSED on V3-prod **collapsed when re-evaluated on real
benchmark chat** -- even though V3-prod was already authored at
production length. So the lesson is sharper than "match the shape":

> Length-matched is not shape-matched. A prototype set that controls
> only the **length** axis but not the **distribution** axis (tone,
> domain, first-person mix, evaluative-vs-factual ratio) can still
> produce a STRONG gate that does not survive contact with the
> actual production input. The gate has to be measured on data
> drawn from the same distribution as production, not just the same
> length.

This applies symmetrically to retrieval/builder decisions: a
benchmark whose questions and chunks are length-matched to
production but stylistically synthetic can mislead in the same way.

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
> evaluation data is synthetic / authored, the verdict is provisional
> until re-validated on a sample drawn from the same distribution as
> production input -- both at the right length AND at the right
> tone/domain/mix. Length-matching alone is not sufficient.
> Re-validation is required before any production default is changed.

The role program (#39 4-class, #47 2-class) showed both axes
matter independently: matching length without matching distribution
gave the same false-STRONG that matching domain without matching
length gave. This gives matched-distribution eval its own checkbox
in the gating flow, parallel to the cross-benchmark axis #44
already establishes.

### Files referenced

- `notes/2026-04-28-role-program-prod-shape-validation.md`
- `notes/2026-04-29-issue-47-ac2-failure.md` (the AC#2 writeup)
- `experiments/role_geometry/production_shape_validation.py`
- `experiments/role_markers/{prototypes,eval}_v3_production.json`
- `experiments/role_markers/bundles_research_only/role_prototypes_obs_pref.json`
- `experiments/role_markers/extension_test_obs_pref.py` (AC#2 reproducer)
- `notes/2026-04-28-role-classifier-subset-probe.py` (AC#1 reproducer)

All committed to develop in PR #48.
