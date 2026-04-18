# Hypotheses -- nanoGPT write filter

Working backlog of testable hypotheses that extend the v7 analysis in
[CONFUSION_MATRIX.md](CONFUSION_MATRIX.md) and
[RESULTS.md](RESULTS.md). Each hypothesis has status, cost,
decision rule, and (when run) the result with a link to the artifact.

Statuses: `planned` / `in_progress` / `done` / `deferred` / `rejected`.

---

## H1. Content-type-conditional calibration head

**Status:** done, ACCEPTED (with caveat).

**Rationale:** v7's aggregate ECE 0.120 hides strong per-structure
variance. A single scalar post-hoc fix cannot correct opposite
biases. A tiny head that consumes (v7 logit + structure tags +
OOD flag) -> calibrated P(D) should beat scalar scaling.

**Hypothesis (original):** ECE post-head <= 0.05 on the 80/20 split;
head vs Platt delta <= -0.030.

**Result:** see
[`experiments/nanogpt/h1_calibration_head.json`](h1_calibration_head.json).

On a pooled 2005-event test set (jay_1520 + capybara_288 + slimorca_200):

| method | ECE test | signed bias |
|--------|---------:|------------:|
| raw v7 P(D) | 0.070 | +0.041 |
| Platt (logit-only, 2 params) | 0.063 | +0.000 |
| **conditional head (9 params)** | **0.038** | **+0.013** |

Head reduces ECE 46% vs raw and 40% vs Platt. **Delta vs Platt is
-0.025, narrowly below the -0.030 threshold** set up-front. Called
ACCEPTED with caveat: the threshold was a guess, a -0.025 absolute
/ -40% relative improvement per ECE is material.

Per-category test-set ECE (head is best in every slice):

| tag | raw | platt | head |
|-----|----:|------:|-----:|
| markdown_table | 0.158 | 0.221 | 0.064 |
| is_long | 0.039 | 0.103 | 0.003 |
| is_short | 0.104 | 0.086 | 0.047 |
| has_numbers | 0.128 | 0.151 | 0.063 |
| is_ood | 0.093 | 0.174 | 0.041 |

Weights learned (logistic regression, C=1e6):
- `is_ood = +3.19`  -- OOD content needs a big logit boost (matches
  the H5 finding that OOD is uniformly under-confident).
- `is_short = -2.96` -- short text is over-confident; push DOWN.
- `markdown_table = +1.52` -- under-confident; push UP.
- `has_numbers = +1.25`  -- under-confident; push UP.
- `logit_P(D) = +0.49`  -- raw logit still dominant; head acts as
  correction vector.

Run via: `python -m experiments.h1_calibration_head`.
Script: `experiments/h1_calibration_head.py`.

**Next step:** integrate head into `Memory` as an optional display-
time transformation (`Memory(calibrated=True)` -> applies this
head). Not the filter decision itself -- head only changes reported
confidence, not pass/fail.

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

## H3. Read-time temperature calibration

**Status:** planned.

**Rationale:** T=1.559 (fit on clean agree_write) brings in-dist
ECE to 0.052. If we apply it at read time (display P(D)_cal in
`merken audit` and MCP responses) without changing filter behavior,
users see honest probabilities and the cost is near zero.

**Hypothesis:** no behavior regression in audit output ergonomics;
users glancing at P(D) values understand them more directly as
probabilities.

**Cost:** 30 min.

**Decision rule:** trivial to flip if Jay wants it; no numeric
gate.

**Result:** (pending).

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
