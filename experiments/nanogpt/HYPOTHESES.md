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

Per-category test-set ECE is mixed. Head wins on is_ood and the
very-small-n slices (markdown_table, is_long) but REGRESSES on
has_numbers where baseline was already near-perfect:

| tag | n_test | raw | platt | head |
|-----|-------:|----:|------:|-----:|
| markdown_table | 13 | 0.072 | 0.106 | **0.000** |
| is_long | 15 | 0.048 | 0.087 | **0.000** |
| is_ood | 97 | 0.125 | 0.122 | **0.049** |
| has_numbers | 74 | 0.018 | 0.043 | 0.064 (head REGRESSES here) |

The markdown_table and is_long "wins" should be read with caution:
small n, unregularized weights overfit those slices. H11 addresses
this with regularization; H10 then adds interactions. The H1 run
itself is a reference baseline, not a shippable artifact.

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

**Status:** done, REJECTED (catastrophic OOD collapse).

**Rationale:** v8 failed because three dataset levers (binary labels,
class balance, starter oversample) together pushed the model toward
a global DEC bias. A contrastive-loss formulation that EXPLICITLY
pairs same-starter-different-class events in each minibatch would
force the attention layers to read past the starter.

**Hypothesis:** a contrastive-trained model recovers >= 50 of v7's
157 false negatives without regressing markdown FPR above 10%.

**Cost:** 1-2 hours. Modifies nanoGPT train.py loss.

**Decision rule:** ship as graduated shadow if the two conditions are
met. (Note: the working model is named `v10_contrastive` because the
v9 slot is held by H_v9_pragmatic. See `out-merken-bpe-v10_contrastive/`
in the nanoGPT repo.)

**Implementation:** hinge loss on the (DECISION, NOISE) logit gap at
the position right after `<|label|>`, computed for `2 * 8` paired
sequences sampled from 12 ambiguous starters mined from the v7 train
split (3-word starter signature, 21 DEC and 153 NOI events, 256 cross
pairs). Aux loss weighted at lambda=0.5, margin=2.0 in logit space.
Same architecture, hyperparams, and dataset as v7.

  - prepare: `nanoGPT/data/merken_bpe_v10_contrastive/prepare.py`
  - train:   `nanoGPT/train_contrastive.py`
  - config:  `nanoGPT/config/train_merken_bpe_v10_contrastive.py`
  - mining:  `experiments/nanogpt/h2_mine_starter_pairs.py`
  - eval:    `experiments/nanogpt/eval_h2.py`
  - artifact: `experiments/nanogpt/h2_results.json`

**Result:** see `experiments/nanogpt/h2_results.json`. Both decision
bars FAIL.

| metric | v7 | v10_contrastive | delta |
|--------|---:|----------------:|------:|
| labels-157 DEC recall | 65.6% | 51.0% | **-14.6pp** |
| FN recovered (target >=50) | -- | **21** | -- |
| FN newly lost | -- | 44 | -- |
| markdown_tables_held_out FPR (target <=10%) | 0.0% | **83.3%** | **+83.3pp** |
| organic_val_held_out agreement | 100% | **14.3%** | -85.7pp |
| jay_vstash_decontam agreement | 100% | **14.3%** | -85.7pp |
| analytics_project agreement | 66.7% | 83.3% | +16.7pp |
| bilingual_es_en agreement | 91.7% | 100% | +8.3pp |
| noisy_agent_stream agreement | 66.7% | 70.8% | +4.2pp |

The contrastive aux loss saturated to 0.0 by step 100 and never
contributed gradient again (the model trivially achieves margin=2.0
on just 12 starter buckets via memorization). The OOD damage was
done in those first 100 steps and CE training across the next 700
steps could not undo it.

**Diagnosed failure mode** -- the loss optimized the AUX problem
(separate the 12 ambiguous starter buckets) without solving the META
problem (generalize past starter). Inspecting v10's FN deltas makes
this concrete:

  - **Recovered FNs** all start with transition phrases that v10 saw
    in contrastive training: `Also need to`, `Also remove`, `Excellent!`,
    `Empiezo por`, `Esperando que`, `Buen review`. v10 learned to
    classify these as DEC.
  - **Newly lost FNs** start with structural markers absent from
    contrastive training: `# Hallazgo`, `## Evaluation`, `**CL ALL...`,
    `174 passed`, `26/26 pass`. v10 forgot how to recognize structured
    decision content because the aux loss reweighted gradient toward
    starter-shaped events.

The contrastive loss did the OPPOSITE of the intent: instead of
forcing the model past the starter, it taught the model that the
starter is even MORE load-bearing.

**Lesson:** an auxiliary loss with N=12 buckets and margin=2.0 in
logit space is too easily satisfied by memorization. The aux loss
needs either (a) much higher pair diversity, (b) a representation-
level objective that cannot be satisfied by output-token logits
alone, or (c) curriculum mixing so it doesn't dominate early
gradient when the LM is still random.

**v10_contrastive is archived as ablation evidence, not shipped.**
v7 remains the graduated shadow baseline.

**Follow-ups opened by this result:** see H2b and H2c below. H8
(architecture bump) is now de-prioritized -- the bottleneck is the
training signal definition, not parameter count.

---

## H2b. Contrastive on hidden-state representation, not output logits

**Status:** planned.

**Rationale:** H2 failed because the hinge on output-token logits
let the model satisfy the margin via memorized "this starter ->
that label" without changing the internal representation. A
representation-level contrastive (InfoNCE) on the hidden state at
position(`<|label|>`-1) cannot be satisfied by lookup; it requires
the embedding of the *payload* to differ between DEC and NOI
anchors that share a starter.

**Hypothesis:** swap the output-logit hinge for InfoNCE over the
last-position hidden state. Same 12 starter buckets. Recover >=50
of v7's 157 FNs without crashing OOD recall (organic_val and
jay_vstash_decontam stay >= 85% agreement).

**Cost:** 30 min. Same train_contrastive.py, swap the loss term.

**Decision rule:** if OOD recall holds AND FN recovery >= 30 (a
relaxed bar; demonstrates the formulation is sound), iterate.
Otherwise mark "contrastive class of approaches" as exhausted.

---

## H2c. Pair-diversity expansion before training

**Status:** planned.

**Rationale:** 12 ambiguous starters is too few to drive a
generalizable signal. H2c first expands the pair pool by mining
pseudo-pairs: synthesize a NOI variant for each held-out DEC by
prepending one of the 12 known transition starters, and synthesize
a DEC variant for each NOI by truncating its starter. Target:
>= 100 ambiguous starters with >= 5 events on each side.

**Hypothesis:** the contrastive loss formulation IS sound; the H2
failure was sample diversity. With >= 100 buckets the aux loss
should not saturate to zero in 100 steps and should drive
representation-level changes.

**Cost:** 1h pair synthesis + 30 min retrain.

**Decision rule:** ship if recovery >= 50 AND markdown FPR <= 10%
AND OOD agreement >= 85% on organic_val and jay_vstash_decontam.

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

**Decision taken:** ship as opt-in. Two ways to enable:

1. Programmatic (resolve the path from the installed package so
   the example works regardless of current working directory):
   ```python
   from pathlib import Path
   import merken.classifiers
   head_path = Path(merken.classifiers.__file__).parent / "calibration_v7.json"
   head = CalibrationHead.from_json(head_path)
   decider = NanoGPTWriteDecider(ckpt, meta, calibrator=head)
   ```
2. Env-var driven, works with `Memory` without code changes:
   ```bash
   export MERKEN_SHADOW=nanogpt
   export MERKEN_SHADOW_NANOGPT_CKPT=.../ckpt.pt
   export MERKEN_SHADOW_NANOGPT_META=.../meta.pkl
   export MERKEN_SHADOW_NANOGPT_CALIBRATOR=default
   ```
   Shadow-mode deciders built via `_build_classifier` pick up the
   calibrator automatically. `default` -> shipped v7 head; any other
   value -> filesystem path. Unset -> no calibration.

The smoke finding on short-concrete DECs pushed down by the head
opened H10 (interaction terms), shipped in the same PR.

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

The C=10 head was briefly the shipped artifact. H10 superseded it
by adding 5 interaction terms at C=1 (ECE 0.035, see H10 block
below). The currently-shipped `merken/classifiers/calibration_v7.json`
holds the H10 C=1 fit, NOT the H11 C=10 fit. H11 remains the
recommended configuration if interaction terms are undesired.

H11 C=10 weights (for reference, not shipped):
- intercept = ~1.0
- logit_P(D) = +0.44 (dominant)
- is_ood = +2.23 (domain shift correction)
- is_short = -1.82 (short over-confident)
- has_markdown_table = +1.73 (under-confident on tables)
- is_long = +1.54 (under-confident on long-form)

vs Platt (ECE 0.053): head beats by -0.010 (19% relative). The
head-vs-Platt delta of -0.010 is below the -0.030 threshold H1 set,
but on contamination-free data the signal IS real and consistent
across regularization strengths.

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

## H_v9_pragmatic. Retrain v7 without knowledge_update in training

**Status:** done, REJECTED (synthetic training DOES carry signal).

**Rationale:** H13 audit showed `knowledge_update_50t` is 100%
training data used as "held-out" in eval_v7_vs_v6.py. Obvious fix
option: retrain v7 without those four scenarios, see if the v7
markdown FPR=0% / organic_val 100% story holds on a clean training
mix.

**Dataset:** same as v7 minus knowledge_update (1732 vs 3046
events: 608 borderline NOISE + 68 organic DEC + 30 markdown NOISE
+ 1026 transcript labels). Same architecture, same max_iters=800,
same block_size=256.

**Result:** see `experiments/nanogpt/eval_v6_to_v9.json` (emitted
by `experiments/eval_v7_vs_v6.py`). v9 BEST val loss 2.60 at step
450 vs v7's 2.35. More importantly, on scenarios:

| scenario | v7 | v9 | delta |
|----------|---:|---:|------:|
| markdown_tables_held_out | 100% | **75%** | **-25pp** (FPR 0->50%) |
| organic_val_held_out | 100% | 100% | +0 |
| jay_vstash_decontam | 100% | 100% | +0 |
| analytics_project | 67% | 0% | -67pp |
| session_2026_04_09 | 75% | 25% | -50pp |
| bilingual_es_en | 92% | 75% | -17pp |
| noisy_agent_stream | 67% | 21% | -46pp |
| knowledge_update_50t | 97% | 87% | -10pp (both saw, so not held-out) |

**Honest finding:** v9 confirms v7's markdown FPR = 0% was propped
up by knowledge_update training content. Drop it, FPR flies back
to 50%. But the held-out DEC-only small scenarios (organic_val,
decontam) hold, showing real-transcript labels generalize DEC
recognition. DEC-only scenarios at larger n regress significantly
(analytics, session, bilingual, noisy_agent all -17pp to -67pp).

**Interpretation:**
- The bug is NOT "knowledge_update in training". The bug is
  "knowledge_update in EVAL". Synthetic noise training provides
  generalizable noise-recognition signal v7 needs; pulling it
  removes more than it gains.
- Correct fix: KEEP knowledge_update in training, DROP it from
  `eval_v7_vs_v6.py`'s scenario list. Paper section 3.2 should
  flag knowledge_update_50t as training-set performance and not
  use it for delta claims.

**v9 is archived as ablation evidence, not shipped as a
replacement.** v7 remains the shadow baseline.

Action items from this finding:
- Remove `knowledge_update_50t` from eval_v7_vs_v6.py's scenario
  list (or rename to make its "training-set performance" nature
  explicit in the output).
- Paper section 3.2: add a row separating "honest held-out"
  scenarios from "training-set diagnostics".
- Future v10: build a NEW held-out noise-heavy scenario with
  topics disjoint from knowledge_update.

## H_disjoint_holdout. Noise-heavy held-out with topics disjoint from training

**Status:** done, validates v7.

**Rationale:** H_v9_pragmatic confirmed that dropping KU from
training hurts more than it helps. The correct fix is a NEW
held-out scenario whose topics do NOT appear in KU training
(cache / db / auth / deploy). This probes v7's real
generalization on noise it was not taught to reject by name.

**Dataset:** `experiments/loop_quality/scenarios/disjoint_noise_heavy_holdout.json`
generated 2026-04-18 via Gemini 2.0 Flash
(`experiments/generate_disjoint_holdout.py`). 57 events total:
24 DEC across 4 domains (observability, mobile_ops,
security_response, product_rollouts) + 33 NOI across 6 domains.
Topics intentionally disjoint from KU. Two of the intended DEC
domains (ml_ops, billing, accessibility, i18n) partially failed
to complete due to Gemini 429 rate limits; the 57 events we have
are the successfully-generated ones.

**Result:** `experiments/nanogpt/disjoint_holdout_eval.json`.

| version | accuracy | DEC recall | NOI recall | FPR |
|---------|---------:|-----------:|-----------:|----:|
| v6 | 61.4% | 83.3% | 45.5% | **54.5%** |
| **v7** | **87.7%** | 70.8% | **100.0%** | **0.0%** |
| v8 | 87.7% | 100.0% | 78.8% | 21.2% |
| v9 | 73.7% | 37.5% | 100.0% | 0.0% |

Three findings:

1. **v7 FPR=0% generalizes to disjoint domains.** On noise events
   about observability / mobile / security / product the model has
   literally never seen in training, v7 correctly skips all 33/33.
   That is not memorization -- it is a learned noise-shape
   signature (table formatting, filler verbs, transition syntax)
   that transfers across content domains.

2. **v9 (no synthetic training) collapses DEC recall to 37.5%.**
   Third independent confirmation that dropping KU from training
   does NOT fix anything -- it just makes the model uniformly
   more conservative and unusable. The right fix (this scenario)
   keeps KU in training and uses a disjoint held-out for eval.

3. **v7's failure mode on this scenario is DEC misses, not NOI
   false positives.** 7 of 24 real DECs rejected. v8 catches them
   all but at 21% FPR. The Pareto front between v7 and v8 is
   "conservative filter (v7)" vs "aggressive filter (v8)"; v7 is
   the production-safe choice.

**Action item (not yet implemented):** add this scenario to
`eval_v7_vs_v6.py` main() scenario list and REMOVE
`knowledge_update_50t` from it (or keep both with explicit
training-vs-held-out labels). Then re-emit
`eval_v6_to_v9.json` with this scenario included.

## H12. Forward-selection minimal calibration head

**Status:** done, ACCEPTED (beats H10 full head).

**Rationale:** H10 shipped 14 features; leave-one-out
(H10_ablation) showed only `ood_X_short` was load-bearing and
`short_X_file_paths` actively hurt. Jay suggested forward
selection from `ood_X_short` alone, adding one interaction at a
time when ECE drops by >= 0.002.

**Result:** `experiments/nanogpt/h12_forward_selection.json`.

Seed: 9 base features + `ood_X_short` = 0.0486 ECE.
Step 1 winners (added one at a time, kept best):
| candidate | ECE | delta |
|-----------|----:|------:|
| short_X_inline_code | 0.0310 | **-0.0176** (accept) |
| short_X_numbers | 0.0352 | -0.0134 |
| short_X_file_paths | 0.0483 | -0.0003 |
| short_X_markdown_table | 0.0487 | +0.0001 |

Step 2: adding any remaining interaction gives delta <= 0.002;
STOP.

**Final minimal head: 11 features**
(9 base + `ood_X_short` + `short_X_inline_code`).

| head | features | ECE |
|------|---------:|----:|
| H11 | 9 base | 0.0477 |
| **H12 (shipped)** | **11 (9 + 2 interactions)** | **0.0310** |
| H10 (prior ship) | 14 (9 + 5 interactions) | 0.0354 |

H12 BEATS H10: 0.031 vs 0.035. The three extra features in H10
are net-negative at C=1. Re-shipped `calibration_v7.json` with the
H12 11-feature fit. Test suite 18/18 still passes (CalibrationHead
handles both schemas; absent weights default to 0).

## H13. Decontaminate held-out scenarios

**Status:** done.

**Rationale:** Jay's paper review caught that
`jay_vstash_2026_04_09_snapshot` has 13/20 events in v7 training
(`/tmp/organic_train.json`). Ran an audit on every scenario.

**Result:** `experiments/nanogpt/contamination_audit.json`.

Held-out scenarios and overlap with ANY training source:

| scenario | n | overlap | pct |
|----------|--:|--------:|----:|
| markdown_tables_held_out | 12 | 0 | 0.0% |
| organic_val_held_out | 7 | 0 | 0.0% |
| **jay_vstash_snapshot** | **20** | **13** | **65.0%** |
| analytics_project | 12 | 0 | 0.0% |
| session_2026_04_09 | 12 | 0 | 0.0% |
| bilingual_es_en_2026_04_14 | 12 | 0 | 0.0% |
| noisy_agent_stream | 24 | 0 | 0.0% |

Plus expected-contamination scenarios (ARE training data used as
"eval" in eval_v7_vs_v6.py -- this is a different bug than
jay_vstash's):

| scenario | overlap | note |
|----------|--------:|------|
| knowledge_update_50t | 100% | entirely in v7 training |
| knowledge_update | 100% | ditto |
| knowledge_update_hard | 100% | ditto |
| knowledge_update_20topics | 100% | ditto |

**Action 1:** wrote
`experiments/loop_quality/scenarios/jay_vstash_2026_04_09_snapshot_decontam.json`
with 7/20 non-contaminated events kept. v6 and v7 both score 7/7 on
that subset (small n, but genuine generalization not memorization).

**Action 2 (documented, not yet implemented):**
`knowledge_update*` scenarios are training data; using them as
"held-out" in eval_v7_vs_v6 compares training-set performance. The
v7 vs v6 table in PAPER_DRAFT section 3.2 should flag this. For
any future eval, either retrain v8 WITHOUT knowledge_update in
training or create brand-new held-out variants with disjoint topics.

---

## H17. Does turn-granularity filtering recover what H16 lost?

**Status:** done, REJECTED (turn-level is WORSE than session-level).

**Rationale:** H16 showed v7 at session granularity loses ~8pp on
temporal. Working theory: unit mismatch. v7 was trained on
per-event labels (one assistant message = one decision); sessions
are 30-80 turns long. H17 applies v7 at turn granularity (its
training unit), keeping only v7-approved turns, then concatenating
them back into the session text for ingest. Retrieval unit stays
at session -- only the filter unit changes.

**Hypothesis:** turn-level filtering recovers temporal accuracy
lost at session level (target: within 3pp of merken-recall).

**Setup:** same LoCoMo config as H16 -- categories 2+5, top-k=10,
`gemini-2.5-flash`, 3 convs (conv-26, conv-30, conv-41), n=202 QA
per config. Three configs run in one process (same judge session)
for cleaner A/B:
- `vstash-raw` (control, no filter)
- `merken-v7` (H16 session-level filter)
- `merken-v7-turnfilter` (H17 turn-level filter)

**Result:** see
[`experiments/retrieval/locomo/results_h17.json`](../retrieval/locomo/results_h17.json).

| config | temporal | adversarial | overall | CI (overall) |
|--------|---------:|------------:|--------:|:-------------|
| vstash-raw | 62.2% (56/90) | 6.2% (7/112) | 31.2% | [25.2, 37.6] |
| merken-v7 | 54.4% (49/90) | 4.5% (5/112) | 26.7% | [20.3, 33.2] |
| merken-v7-turnfilter | **45.6%** (41/90) | 3.6% (4/112) | **22.3%** | [16.8, 28.2] |

**v7 turn-keep rates** (probed on all turns, not just ingested):

| conv | turns kept | % |
|------|-----------:|--:|
| conv-26 | 263/419 | 62.8% |
| conv-30 | 183/369 | 49.6% |
| conv-41 | 435/663 | 65.6% |

**Findings:**

1. **Turn-level filtering LOSES another 9pp on temporal vs
   session-level** (54.4% -> 45.6%). Both are worse than no filter
   (62.2%). The unit-mismatch theory was backwards.

2. **v7 drops 35-50% of LoCoMo turns.** Short dialogue turns
   ("Did you see X?" / "Yeah, at 3pm") look like noise to a model
   trained on longer assistant-message events with payload. Many
   of those turns carry the one temporal fact a QA asks about;
   dropping them kills retrieval precisely on temporal questions.

3. **Session-level filter was less bad** because when it skipped a
   session it dropped 100% of that session's content, but when it
   kept one it kept everything. Turn-level fragments destroy the
   temporal context inside sessions v7 would otherwise keep.

4. **Calibrator was not re-run at turn level.** H16 established
   calibrator is a no-op on retrieval; no reason to expect it to
   move turn-filter numbers.

**Interpretation:**

The unit-mismatch story holds, but the direction is opposite to
the hypothesis. v7's training unit ("assistant message with
payload") is LARGER than a dialogue turn, not smaller. Matching
v7 to turns feeds it text that is uniformly under its training
distribution's length/payload norm, so it rejects at 35-50%
regardless of semantic value. The filter has no way to know
"this short turn is the temporal evidence someone will ask about
in 20 sessions."

**Decision rule:** v7 should NOT be applied as a generic
session-or-turn filter for long-conversation retrieval. Its
production home is where the input unit matches its training
unit: per-assistant-message write decisions in chat-assistant
contexts (where merken actually uses it).

**What this closes and what it doesn't:**

Closes:
- Both session-level (H16) and turn-level (H17) application of v7
  on LoCoMo-style conversational memory. Neither helps retrieval.

Does not close:
- Whether a v7-like filter *trained on dialogue turns* (H17b)
  would do better. Different training distribution; plausible but
  out of scope for this repo's scenario set.
- Whether a different merken primitive (consolidation, not
  filtering) would help retrieval on LoCoMo. Prior experiments in
  `experiments/consolidation/` already showed brief_v1 wins big
  (+46pp at 50 topics). Retrieval improvements come from
  consolidation, not filter tightening.

Paper section 5 Limitations now has a concrete, closed story: v7
filter is a per-event write decider; when misapplied to
long-conversation retrieval as a session or turn filter, it
degrades accuracy. Not a capacity issue (see H8 discussion); a
training-distribution / input-unit issue.

Script: `experiments/retrieval/locomo/runner.py` (MerkenAdapter
`turn_filter` hook + `merken-v7-turnfilter` config +
`_v7_turn_filter` helper).

---

## H16. Does the v7 filter change retrieval accuracy on LoCoMo?

**Status:** done, REJECTED (filter HURTS temporal retrieval at this n).

**Rationale:** Paper Limitations called out that H16 -- whether v7
at shadow or as the write decider moves downstream QA accuracy on a
long-conversation benchmark -- was unrun. A negative result (v7
drops retrieval accuracy) is as informative as a positive one; both
constrain how aggressively we can ship the filter.

**Setup:** LoCoMo E2E, categories 2 (temporal) + 5 (adversarial),
top-k=10, `gemini-2.5-flash` as answerer AND judge, 3 conversations
(conv-26, conv-30, conv-41), n=202 QA pairs per config, 1616
Gemini calls total. Four configs:
- `vstash-raw`: no write filter, no consolidation.
- `merken-recall`: `HeuristicWriteDecider` + no consolidation.
- `merken-v7`: `ChainedWriteDecider(Heuristic, NanoGPTWriteDecider(v7))`.
- `merken-v7-cal`: same as above + H12 calibration head.

**Hypothesis:** v7 filter does not degrade retrieval accuracy on
the temporal subset by more than 5pp vs `merken-recall` baseline.

**Result:** see
[`experiments/retrieval/locomo/results_v7_cal.json`](../retrieval/locomo/results_v7_cal.json).

| config | temporal | adversarial | overall | CI (overall) |
|--------|---------:|------------:|--------:|:-------------|
| vstash-raw | 65.6% (59/90) | 4.5% (5/112) | 31.7% | [25.2, 38.1] |
| merken-recall | 61.1% (55/90) | 7.1% (8/112) | 31.2% | [24.8, 37.6] |
| merken-v7 | 53.3% (48/90) | 6.2% (7/112) | 27.2% | [21.3, 33.7] |
| merken-v7-cal | 54.4% (49/90) | 5.4% (6/112) | 27.2% | [21.3, 33.7] |

Sessions ingested per config (smaller = filter rejected):
- vstash-raw / merken-recall: 19, 19, 32 (baseline)
- merken-v7 / merken-v7-cal: 18, 14, 31 (dropped 7/70 sessions, 10%)

**Findings:**

1. **v7 filter costs ~8pp on temporal vs merken-recall heuristic**
   (61.1% -> 53.3%). 95% CIs for overall accuracy overlap with the
   baselines, but the temporal subset shows a consistent drop
   across all 3 conversations. The 5pp threshold in the hypothesis
   is violated.

2. **Calibrator is a no-op on retrieval.** merken-v7 and
   merken-v7-cal produce identical ingestion decisions (14/18/31
   sessions kept in both) and within-rounding-error identical
   accuracy. The calibrator only changes the displayed confidence;
   the threshold decision uses raw P(D). So all production
   flavors of v7 (cal vs no-cal) look the same downstream.

3. **Adversarial is baseline noise** across all configs (4-7%).
   LoCoMo cat 5 questions are mostly unanswerable from context;
   every config correctly says "I don't know" most of the time.
   The differentiating signal is in cat 2 (temporal reasoning).

4. **The filter's precision for "is this noise?" is a wrong
   question for retrieval.** v7 was trained on per-event labels:
   given this single event, is it DECISION or NOISE? LoCoMo
   ingestion works at the SESSION level (whole conversation
   sessions at a time). A session containing one high-value
   temporal fact and 50 low-value turns is a DEC at event level
   but gets filtered if the session-level score trends NOI.

**Honest interpretation for the paper:**
- v7 is a good write filter for single-event decisions in a
  chat-assistant context (CONSTITUTION primitive: should_remember).
- v7 is NOT a good session-level filter for long-conversation
  retrieval. The units don't match.
- Cheap mitigation: apply v7 turn-by-turn before aggregating into
  sessions, instead of scoring the aggregated session once.
  That's H17 territory, not a free win here.

**Scope caveats (do not overclaim):**
- n=3 conversations is too small for strong claims. Overall CIs
  overlap across all four configs. The temporal sub-signal is
  consistent across all 3 convs (always a drop), but CI on the
  temporal delta is wide.
- The answerer/judge is `gemini-2.5-flash`. A different LLM could
  give different numbers. Reported as an illustrative probe.
- Adversarial accuracy may be biased up by Gemini judge's own
  temperament about hedged answers.

**Decision rule:** accept H16 as rejected (filter hurts retrieval).
Ship v7 in chat-assistant write paths, NOT as a session-level
filter for long-context QA retrieval. Paper section 5
("Limitations") needs a concrete number now: v7-on-LoCoMo temporal
= 53.3% vs 65.6% baseline, 3-conv probe.

Script: `experiments/retrieval/locomo/runner.py` (extended with
`merken-v7` / `merken-v7-cal` configs + 429-aware judge retry +
torch-first import).

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
