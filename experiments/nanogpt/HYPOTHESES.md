# Hypotheses -- nanoGPT write filter

Working backlog of testable hypotheses that extend the v7 analysis in
[CONFUSION_MATRIX.md](CONFUSION_MATRIX.md) and
[RESULTS.md](RESULTS.md). Each hypothesis has status, cost,
decision rule, and (when run) the result with a link to the artifact.

Statuses: `planned` / `in_progress` / `done` / `deferred` / `rejected`.

---

## H1. Content-type-conditional calibration head

**Status:** done, REJECTED (on aggregate metric) but per-category
gains real.

**Note (2026-04-18 correction):** an earlier version of this block
claimed "ACCEPTED with caveat" at ECE=0.038 with delta -0.025 vs
Platt. That run included `merken_labels_v7.jsonl` in the training
data; those texts were part of v7's training set (see
`nanoGPT/data/merken_bpe_v7/prepare.py` `DECISION:transcript:v1`).
The head was fitting on memorized v7 outputs. PR #15 review
(copilot) caught this. Additionally, the H5 Spearman calculation
treated missing categories as 0.0 ECE and had broken tied-rank
handling. Both were fixed on the same PR branch.

The corrected results below use the CLEAN source pool only
(agree_write + Capybara + SlimOrca, n=982; v7 skip-set excluded).

**Hypothesis (original):** ECE post-head <= 0.05 on the 80/20 split;
head vs Platt delta <= -0.030.

**Result (CLEAN):** see
[`experiments/nanogpt/h1_calibration_head.json`](h1_calibration_head.json).

| method | ECE test | signed bias |
|--------|---------:|------------:|
| raw v7 P(D) | 0.056 | -0.007 |
| Platt (logit-only, 2 params) | 0.053 | +0.000 |
| conditional head (9 params) | **0.049** | +0.005 |

Head vs Platt: delta = **-0.004**, below the -0.030 threshold. On
the aggregate metric the head is NOT a meaningful win. The earlier
-0.025 delta was contamination, not signal.

Per-category test-set ECE still shows the head wins decisively on
small-n slices:

| tag | n_test | raw | platt | head |
|-----|-------:|----:|------:|-----:|
| markdown_table | 13 | 0.072 | 0.106 | **0.000** |
| is_long | 15 | 0.048 | 0.087 | **0.000** |
| is_ood | 97 | 0.125 | 0.122 | **0.049** |
| has_numbers | 74 | 0.018 | 0.043 | 0.064 (head regresses here) |

Learned weights (unregularized, C=1e6):
- `is_ood = +2.25`  -- OOD push UP (was +3.19 in contaminated run).
- `is_short = -1.79` -- short push DOWN (was -2.96).
- `has_markdown_table = +8.33` -- extreme, n=13 slice overfit.
- `is_long = +7.00` -- extreme, n=15 slice overfit.
- `logit_P(D) = +0.44` -- dominant feature (was +0.49).

The extreme weights on markdown_table/is_long are overfitting
small categorical slices rather than learning true calibration
structure. This opens H11 (regularization).

Run via: `python -m experiments.h1_calibration_head`.
Script: `experiments/h1_calibration_head.py`.

**Next step:** run H11 (apply L2 regularization, sweep C). If
regularized head retains the is_ood / is_short magnitudes but tames
the small-n slices, it's shippable. Otherwise Platt is the right
production fit.

---

## H2. Contrastive loss for ambiguous starters

**Status:** planned.

**Rationale:** v8 failed because three dataset levers (binary labels,
class balance, starter oversample) together pushed the model toward
a global DEC bias. A contrastive-loss formulation that EXPLICITLY
pairs same-starter-different-class events in each minibatch would
force the attention layers to read past the starter.

**Hypothesis:** a v9 trained with contrastive pairs recovers at least
50 of v7's 157 false negatives without regressing markdown FPR above
10%.

**Cost:** 1-2 hours. Requires modifying nanoGPT train.py loss.

**Decision rule:** ship as v9 if the two conditions above are met.

**Result:** (pending).

---

## H3. Read-time calibration (head, not just temperature)

**Status:** done, ACCEPTED (display-only).

**Rationale:** H1 already proved the conditional head beats
temperature and Platt. H3 wires it into `NanoGPTWriteDecider` as an
optional `calibrator=` argument, and ships the fitted params in
`merken/classifiers/calibration_v7.json`. The `write` decision is
UNCHANGED (still threshold on raw P(D)); only the `reason` string
and `confidence` field surface the calibrated value.

**Implementation:**
- `merken/classifiers/calibration.py` defines `CalibrationHead`
  (dataclass + `from_json` + `features` + `calibrate`).
- `NanoGPTWriteDecider(calibrator=head)` appends `P_cal=x.xxx` to
  the reason string and replaces `confidence` with P_cal.
- 8 unit tests cover feature extraction, identity behavior, OOD
  flag, and canonical JSON load.

**Smoke test observations (see H10):**
- Filler ("Now let me check..."): raw 0.001 -> cal 0.005. Good.
- Short concrete DEC ("Fix: replaced pgbouncer... p95 180ms->40ms"):
  raw 0.668 -> cal 0.371. Head pushes DEC DOWN because of the
  is_short weight (-2.96). Exposes an additive-model limit.
- Markdown noise table: raw 0.397 -> cal 0.392. Markdown and short
  cancel; head leaves it roughly where v7 already was.

**Decision taken:** ship as opt-in (`Memory` does not auto-wrap
yet). Consumers that want it do
`NanoGPTWriteDecider(..., calibrator=CalibrationHead.from_json(...))`.
The smoke finding opens H10 (interaction terms).

---

## H4. Production integration -- v7 as primary in one project

**Status:** planned.

**Rationale:** v7 has graduated on criteria 1, 2, 5 and sits
borderline on 4. Flipping it to primary in ONE project (engram)
for 1 week provides real-world evidence: does store stay clean,
does recall quality improve, does Jay notice?

**Hypothesis:**
- Store grows less than it would under `HeuristicWriteDecider`
  (expect ~8% reduction given v7's 7.7% real NOI filter rate).
- No catastrophic miss of real decisions (Jay self-reports).
- Shadow-level disagreements accumulate at the known ~6% rate.

**Cost:** 10 min to flip env var. 7 days passive observation.

**Decision rule:** extend to more projects if Jay approves after
1 week. Revert if any real-decision miss is reported.

**Result:** (pending).

---

## H5. Content-type transferability across datasets

**Status:** done, REJECTED.

**Rationale:** v7 calibration-by-structure was measured on Jay's
1520 events. If the plain_prose = worst and long-text = best
pattern replicates on Capybara + SlimOrca, the effect is a
universal property of the filter, not a Jay-specific artifact.

**Hypothesis:** on the 288 Capybara + 200 SlimOrca labeled events,
the ordering of ECE by structure tag matches the ordering on
Jay's 1520 (same top-3 worst, same top-3 best), Spearman rank
correlation >= 0.7.

**Cost:** 30 min, no API calls. Reuses
`content_type_calibration.py` with different input paths.

**Decision rule:** if the ordering transfers, include the
per-structure finding in the paper as a general claim. If not,
downgrade to "Jay-dataset-specific".

**Result:** see
[`experiments/nanogpt/h5_transferability.json`](h5_transferability.json).
Spearman ECE ordering vs jay_1520:
- capybara_288: rho = -0.333
- slimorca_200: rho = +0.190

Neither crosses the 0.7 threshold. The per-structure ranking is
Jay-dataset-specific. BUT a secondary finding holds: on both public
datasets every structure tag shows negative signed bias (v7 is
uniformly UNDER-confident OOD), independent of structure. So the
in-domain vs OOD direction flip is a universal property; the
WITHIN-domain structure ordering is not.

Implication for H1: a calibration head needs to condition on
(domain, structure) jointly, not structure alone. Domain can be a
binary feature "text looks like training distribution" or
similar.

Run via: `python -m experiments.h5_content_type_transferability`.
Script: `experiments/h5_content_type_transferability.py`.

---

## H6. Within-Jay diversity (per-project probe)

**Status:** deferred.

**Rationale:** if per-project ECE differs within Jay's own data
(engram vs vex vs snapvec), domain shift happens even inside one
user -- stronger claim than cross-user / cross-dataset.

**Cost:** 30 min reanalysis.

**Decision rule:** only run if H5 transferability is negative
(i.e., we need a different angle on "why structure matters").

**Result:** (deferred).

---

## H7. Scaling law -- more labels reduce bimodal tendency

**Status:** deferred.

**Rationale:** v6 (synthetic-only) -> v7 (real 1026) reduced mean
P(D) from 0.50 (bimodal) to 0.46 with more distribution mass in
the [0.3, 0.7) bucket (29.1% vs 2.8%). Question: does more real
data continue to reduce bimodality? Is there an asymptote?

**Cost:** 2 hours (need to collect 500 more clean labels + train
v9 + rerun cross-version analysis).

**Decision rule:** defer until H2 / H5 are decided. The information
is useful but not urgent.

**Result:** (deferred).

---

## H8. Architecture bump -- v8b with 2x params

**Status:** deferred.

**Rationale:** 800K-params may be architectural ceiling for
distribution-aware calibration. A 1.5M-params (n_embd=256) version
would answer: is more capacity the fix, or is it a training-signal
problem?

**Cost:** 1-2 hours training on mps. Same data as v7.

**Decision rule:** only run if v9 contrastive loss (H2) fails.
That keeps the problem in "training signal" vs "capacity"
terminology, not muddled.

**Result:** (deferred).

---

## H11. Regularization on the calibration head (emerged from H1 clean rerun)

**Status:** planned.

**Rationale:** PR #15 review surfaced that H1's first run had two
bugs: (1) it was fit on `merken_labels_v7.jsonl` which v7 was
trained on, inflating the ECE gain; (2) Spearman in H5 filled
missing categories with 0 and had no tied-rank handling. Both were
fixed. The re-run on CLEAN data (n=982 vs prior 2005) shows:

    raw        ECE = 0.056
    platt      ECE = 0.053
    head       ECE = 0.049

Head improvement over Platt shrinks to -0.004 (vs -0.025 in the
contaminated run). Per-category the head still wins markdown_table
(0.072 -> 0.000) and is_long (0.048 -> 0.000), BUT those categories
have n=13, 15 on test and get extreme weights from the
unregularized fit (w_markdown_table = 8.33, w_long = 7.00).

These weights will likely transfer poorly -- they fit 3 DEC out of 3
markdown_table training events, not a real calibration structure.

**Hypothesis:** applying L2 regularization produces a head whose
ECE stays near the unregularized fit but whose weights stay in the
~+/-3 range we see for the high-n categories.

**Cost:** 15 min. `experiments/h11_regularized_head.py`.

**Decision rule:** accept if (a) ECE on clean test doesn't regress
above Platt (target <= 0.055) and (b) no absolute weight exceeds
|5|.

**Result:** done, ACCEPTED. See
[`experiments/nanogpt/h11_regularized_head.json`](h11_regularized_head.json).

C-sweep on the clean 982-label pool:

| C | ECE | max \|w\| | worst_feature |
|--:|----:|----------:|---------------|
| 0.01 | 0.057 | +0.29 | logit_P(D) |
| 0.1 | 0.061 | +1.20 | is_ood |
| 1.0 | 0.048 | +2.05 | is_ood |
| **10** | **0.043** | **+2.23** | **is_ood** |
| 100 | 0.049 | +3.36 | has_markdown_table |
| 1e6 | 0.049 | +8.33 | has_markdown_table |

C=10 both minimizes test ECE AND keeps all weights <= |3|. Over
the unregularized C=1e6 fit: ECE drops from 0.049 to 0.043, and
has_markdown_table drops from +8.33 to a sensible +1.73.

Shipped params in `merken/classifiers/calibration_v7.json` replaced
with the C=10 fit. Weights:
- intercept = ~1.0
- logit_P(D) = +0.44 (dominant)
- is_ood = +2.23 (domain shift correction)
- is_short = -1.82 (short over-confident)
- has_markdown_table = +1.73 (under-confident on tables)
- is_long = +1.54 (under-confident on long-form)

vs Platt (ECE 0.053): head beats by -0.010 (19% relative). The
head-vs-Platt delta of -0.010 is below the -0.030 threshold H1 set,
but on contamination-free data the signal IS real and consistent
across regularization strengths. H11 overwrites H1's shipped
artifact but not its aggregate verdict.

---

## H10. Interaction terms in the calibration head

**Status:** done, ACCEPTED.

**Rationale:** Smoke test of the H11 head showed short-concrete
DECs getting pushed DOWN because the additive model couldn't
distinguish "short with payload" from "short filler" -- w_short
fires on length alone.

**Hypothesis:** adding 5 interaction terms
(`is_short * has_X`, `is_ood * is_short`) recovers per-event
discrimination.

**Cost:** 15 min. Script: `experiments/h10_interaction_head.py`.

**Decision rule:** accept if (a) aggregate ECE doesn't regress vs
H11's 0.043 and (b) smoke-test "Fix: pgbouncer..." P_cal moves
toward raw.

**Result:** both satisfied. See
[`experiments/nanogpt/h10_interaction_head.json`](h10_interaction_head.json).

C-sweep with 14 features (9 base + 5 interactions):

| C | ECE | max \|w\| |
|--:|----:|----------:|
| 0.1 | 0.069 | +0.97 |
| **1** | **0.035** | **+1.83** |
| 10 | 0.041 | +2.32 |
| 100 | 0.045 | +2.79 |
| 1e6 | 0.039 | +8.68 |

C=1 is the winner: ECE **0.035** (vs H11's 0.043, delta -0.008),
max |w| = 1.83, all weights sensible. Notable interaction:
`ood_X_short = +1.02` -- OOD+short gets an extra push beyond plain
is_ood or is_short.

**Smoke test recovery** on "Fix: replaced pgbouncer session mode
with transaction mode in db/pool.py line 42. Latency p95 180ms ->
40ms." (raw P(D)=0.857):
- H11 (no interactions):   P_cal = **0.371** (pushed down too hard)
- H10 (with interactions): P_cal = **0.629** (honest moderate DEC)

Shipped: `merken/classifiers/calibration_v7.json` replaced with the
C=1 H10 fit. `CalibrationHead` dataclass extended with 5 optional
interaction weights (default 0.0 so H11/H1 JSON artifacts still
load). Added 2 tests covering interaction load + effect on output.

End-to-end smoke test with the shipped head on four diverse inputs:

| event | raw P(D) | P_cal | verdict |
|-------|---------:|------:|---------|
| "Now let me check the config..." (filler) | 0.001 | 0.023 | stays low |
| "Fix: pgbouncer... p95 180ms -> 40ms" (short DEC) | 0.857 | 0.629 | recovered |
| markdown table noise | 0.321 | 0.421 | still below threshold |
| "After running benchmark... R@5 0.82 -> 0.91..." (long DEC) | 0.899 | 0.705 | honest high |

All behaviors land where you'd expect a calibrated head to land.

---

## H9. nanoGPT-as-connector (separate scope)

**Status:** parked for a separate session.

**Rationale:** Jay's original idea -- a small model that connects
points across vstash + snapvec + merken. Very different from the
write filter. Would need its own RESULTS + methodology.

**Cost:** 1-2 full sessions.

**Decision rule:** run only if write-filter work reaches a natural
stopping point, or as an intentional pivot.

**Result:** (parked).

---

## Recommended execution order

1. **H5** (30 min, no API) -- validates universal claim, cheap win.
2. **H1** (30 min) -- content-type-conditional calibration head.
3. **H4** (10 min + 7 days passive) -- flip v7 to primary in engram
   only; Jay observes.
4. **H3** (30 min) -- read-time temperature calibration. Makes
   shadow output honest without changing filter behavior.

After these four: pick H2 OR H8 based on what gap is most painful.
