# Latent Calibration in an 800K-Parameter Memory Filter

**Working title.** N=1 engineering memo. **Not a workshop submission**
as-is; see "What this is / is not" below. Numbers are load-bearing;
everything cites a script in `experiments/` or an artifact in
`experiments/nanogpt/`.

## What this is / is not

This is an **engineering memo from a single developer** (Jay)
documenting the design + honest measurement of a small
specialized classifier for one memory system, on one user's
transcript distribution, oracled by Gemini 2.0 Flash in a
specific prompt.

Core methodological limits, stated up front so no reviewer has to
dig for them:

- **N=1 user.** All real-content measurement is on transcripts from
  one author.
- **Gemini-in-this-prompt is the "oracle".** There is no human
  ground truth at scale. A 20-sample manual audit agreed with
  Gemini on 15/20 DECISIONs (75%) -- the oracle itself has a
  non-trivial error rate. Every "accuracy" number below is
  accuracy-against-Gemini-in-this-prompt, not accuracy-vs-truth.
- **One held-out scenario partially contaminated.** Original
  `jay_vstash_2026_04_09_snapshot` had 13/20 events also in
  `/tmp/organic_train.json` (v7 training). A decontaminated
  sibling file with 7/20 events was derived by
  `experiments/contamination_audit.py`; v6 and v7 both score 7/7
  there (small n; generalization rather than memorization).
  `eval_v7_vs_v6.py` still points at the ORIGINAL file so its
  score matches the historical record; the decontam variant is an
  opt-in alternative held-out. See section 3.3.
- **Four "held-out" scenarios are actually training data.** The
  `knowledge_update*` scenarios loaded by `prepare.py` are both
  training corpus and (incorrectly) used as "held-out" in
  `eval_v7_vs_v6.py`. Section 3.2's delta v7 vs v6 on
  knowledge_update_50t compares training-set performance, not
  generalization. A pragmatic ablation (v9 in HYPOTHESES.md)
  retrained v7 without knowledge_update and found the markdown
  FPR=0% collapses to 50%: v7's markdown capability was propped
  up by that training content. The correct fix is therefore to
  drop knowledge_update from EVAL (not training), and to build a
  disjoint-topic held-out for future v10.
- **H10 feature set was post-hoc feature engineering. Forward
  selection (H12) found a smaller stable head that beats H10.** The
  shipped `calibration_v7.json` is now the 11-feature H12 fit (ECE
  0.031 vs H10's 0.035). See section 4.5.

With those caveats the rest of the document is what we measured.
This framing targets a **blog post / tech report**, not a venue
submission. To get to a workshop we would need N >= 3 users, a
larger human-labeled subset as a second oracle, retrieval E2E
evaluation, and a clean held-out that drove no design decisions.

## Abstract

We measured a 4-layer 800K-parameter BPE write-filter transformer
on two class distributions: a synthetic scenario with 86% NOI
prevalence (designed to expose filter behavior on heavy noise) and
real Claude Code transcripts with ~24% NOI prevalence. Store
reduction is 86% on synthetic and **7.7% on real**; population-
weighted accuracy-vs-Gemini is 99.9% on synthetic and **80.9% on
real**. We document this Production-Benchmark Gap and show that
**scalar post-hoc calibration cannot represent opposite-direction
bias across domains** -- a structural limit of scalar methods, not
of post-hoc calibration in general (a per-distribution head would
resolve it; we did not fit one). A leave-one-out ablation on our
14-feature calibration head reveals that the 5 interaction terms
are mostly not load-bearing: only ``ood_X_short`` contributes
meaningfully; the interaction we designed from a specific smoke
case (``short_X_numbers``) contributes zero. An ablation with v6 +
the same head shows calibration is largely a post-hoc fit that
works on either base model (v6+head ECE 0.047, v7+head ECE 0.035);
real-label training redistributes raw output mass (bimodal 93% ->
59%) but does NOT confer calibration the head can't recover.

## 1. Setup

`merken` is a memory layer for LLM agents. Each assistant turn is a
candidate event; the ``should_remember`` policy decides write vs
skip. Production default is ``HeuristicWriteDecider``: novelty +
dedup + length gates. ``NanoGPTWriteDecider`` is a specialized
classifier that runs in "shadow mode" alongside the heuristic;
after labeling disagreements it can be graduated to primary.

Architecture across v4-v8:

- n_layer = 4, n_head = 4, n_embd = 128
- vocab_size = 512 (BPE), ~800K params
- v4-v6: block_size = 128. v7+: block_size = 256 (see 3.1).

## 2. Dataset bootstrap

No labeled benchmark for "keep-this-in-long-term-memory" exists for
agent transcripts. We built one.

The live Claude Code PreCompact hook had been silently broken:
``json.load`` on JSONL transcripts, error swallowed by
``2> /dev/null``, zero writes since the transcript format change.
All per-project ``~/.merken/*.db`` had ``shadow_agree =
shadow_disagree = 0``.

After the hook fix, three scripts:

1. `experiments/bootstrap_from_transcripts.py` replays primary +
   shadow over every `~/.claude/projects/*.jsonl` transcript (415
   files, 18,220 candidate events). Labels the 1,040 disagreements
   with Gemini 2.0 Flash using a strict prompt (see
   `experiments/oracle_model_bench.py`). A manual audit flagged
   the first pass as too permissive; a re-label with an explicit
   filler-pattern definition flipped 588 of 739 (80%) DECs to
   NOISE. Final ratio: 157 DEC / 869 NOI.

2. `experiments/oracle_agree_write_sample.py` reservoir-samples
   500 events from the 12,492 agree_write population to complete
   the confusion matrix. 494 labeled (6 rate-limit skips).

3. `experiments/oracle_public_dataset.py` adds 288 LDJnr/Capybara
   and 200 Open-Orca/SlimOrca chunks for cross-distribution
   probes.

Total Gemini-labeled pool: 1520 (Jay transcripts) + 488 (public).
A 20-sample manual audit on DECs showed 15/20 (75%) agreement with
the Gemini label; the other 5 the human would have flipped to
NOISE. This bounds the useful strength of our oracle: ~80% reliable
on the DEC class. All "agreement" numbers below should be read with
that ceiling in mind.

## 3. v7 measurement

`experiments/eval_v7_vs_v6.py` compares v4-v8 on held-out scenarios.

### 3.1 Block_size measurement

| class | median tokens | p90 | max | % > block_size=128 |
|-------|--------------:|----:|----:|-------------------:|
| DECISION | 116 | 305 | 557 | **45.9%** |
| NOISE | 38 | 60 | 357 | 0.7% |

At block_size=128 the model could not see the payload of half the
real DECs. v6's 66.7% FPR on markdown_tables_held_out was
partially a window-size defect, not purely a training defect.
v7 doubled block_size to 256.

### 3.2 Scenario scores (v6 vs v7)

| scenario | v6 agreement | v7 agreement | note |
|----------|-------------:|-------------:|------|
| markdown_tables_held_out | 66.7% | 100% | held-out by construction |
| organic_val_held_out | 100% | 100% | |
| jay_vstash_snapshot | 100% | 100% | **CONTAMINATED (see 3.3)** |
| knowledge_update_50t | 99.9% | 97.4% | synthetic; v7 dilutes stereotype |

### 3.3 Contamination disclosure

Of the 20 events in `jay_vstash_2026_04_09_snapshot.json`,
**13 also appear in `/tmp/organic_train.json`**, which is loaded
into v7's training set as ``DECISION:organic:v1`` (see
`nanoGPT/data/merken_bpe_v7/prepare.py`). So 65% of the "held-out"
events are actually training data. The 100% agreement is largely
memorization.

This is the same kind of bug that invalidated H1's first result
(copilot caught it on PR #15). Our graduation criterion #2
("jay_vstash recall >= 95%") is a contaminated metric in its
current form. Fix: remove the organic events from the held-out
scenario OR retrain v8+ without the organic DECs. We did neither,
so this number should be read as evidence of memorization, not
generalization.

### 3.4 Full population-weighted confusion matrix

`experiments/confusion_matrix_v7.py`:

```
               oracle=DEC   oracle=NOI
v7 WRITE         398 (TP)     96 (FP)     n=494
v7 SKIP          157 (FN)    869 (TN)     n=1026
```

Projected onto the observed ~13,539-event population:

| metric | value |
|--------|------:|
| accuracy-vs-Gemini (pop-weighted) | **80.9%** |
| DEC recall | 98.4% |
| NOI recall (specificity) | **26.8%** |
| real store reduction | **7.7%** |

The 84.7% we first reported was skip-set only. Adding the WRITE
column drops accuracy to 80.9%. The 86% store reduction was
measured on a scenario with 86% NOI events. Real distribution is
~76% DEC / ~24% NOI. v7 is a CONSERVATIVE filter: rarely costs a
real decision (98.4% DEC recall), rarely prunes noise (26.8% NOI
recall).

Oracle reliability: the 15/20 (75%) manual audit covers the DEC
class only. NOI-side human audit was not run. A rough upper bound
on overall oracle reliability is therefore unverified; if NOI
accuracy matches DEC at ~75-85%, the pooled upper bound is also
~75-85%. The 80.9% population-weighted number sits inside that
band and should not be read as a hard accuracy claim against
ground truth.

## 4. Calibration

### 4.1 Raw-output distributions

From `experiments/v6_with_h10_head_ablation.py` on the 979-event
clean pool:

| model | raw ECE | bimodal frac (<0.1 or >=0.9) | middle frac (0.3-0.7) |
|-------|--------:|------------------------------:|----------------------:|
| v6 | 0.166 | 92.9% | 1.5% |
| v7 | 0.056 | 59.4% | 7.6% |

v6 is a confident 2-class voter: 93% of events at the extremes,
1.5% in the middle range. v7 has more distribution mass in the
ambiguous zone: 59% at extremes, 7.6% middle. **Real-label training
redistributes raw output mass; it does not obviously confer
calibration.**

### 4.2 Post-hoc head (H10) on both

Same H10 head architecture (14 features: v7 logit + 7 structure
tags + 1 OOD flag + 5 interactions), fit separately on v6 outputs
and on v7 outputs, both at C=1:

| base model | raw ECE | +H10 head ECE | delta |
|------------|--------:|---------------:|------:|
| v6 | 0.166 | 0.047 | -0.119 |
| v7 | 0.056 | 0.035 | -0.020 |

**The head recovers calibration for EITHER base model.** Delta
between v6+head and v7+head is 0.012.

**Threshold disclosure:** the threshold of 0.02 for "meaningful
difference" was written into the verdict logic of
`experiments/v6_with_h10_head_ablation.py` before the script was
run for the first time, but it was NOT recorded in HYPOTHESES.md
with a timestamp prior to the experiment. In the weaker sense of
pre-registration (chosen before seeing the number), 0.02 is
pre-registered. In the stronger sense (public dated artifact), it
is post-hoc. Stating this plainly because a hostile reviewer
would rightly ask.

Regardless of threshold semantics, the 0.012 delta is about half
the size of the standard "0.05 = well calibrated" gap that
framed the entire calibration section, so calling it "small" is
the honest read.

This **falsifies** the stronger version of our earlier claim
("uncertainty emerges from training, not architecture"). Honest
rewrite: real-label training **redistributes raw output mass**,
but a small post-hoc head recovers ECE regardless. What training
buys you is **interpretable probabilities without a calibration
wrapper**, not calibration per se.

### 4.3 Structure-dependent calibration

`experiments/content_type_calibration.py` tags each of the 1520
Jay events by structure and measures v7 ECE per tag:

| tag | n | ECE | signed |
|-----|--:|----:|-------:|
| long (>= 1000c) | 78 | 0.045 | -0.045 |
| has_file_paths | 191 | 0.084 | +0.026 |
| has_numbers | 385 | 0.104 | -0.062 |
| has_inline_code | 502 | 0.103 | -0.051 |
| has_code_fence | 60 | 0.107 | -0.107 |
| short (< 300c) | 1254 | 0.168 | +0.148 |
| has_markdown_table | 79 | 0.184 | **-0.184** |
| plain_prose | 786 | 0.191 | **+0.191** |

Aggregate ECE 0.120 is dominated by plain_prose (n=786, ECE 0.191).
Structured content is well-calibrated. This decomposition motivated
the conditional head (section 4.5).

### 4.4 Transferability probe (H5, REJECTED)

`experiments/h5_content_type_transferability.py` computed Spearman
over the categories present with n >= 5 in both datasets:

- jay_1520 vs capybara_288: rho = -0.40 (4 shared bins)
- jay_1520 vs slimorca_200: rho = +0.50 (3 shared bins)

Both below the pre-registered 0.7 acceptance bar. Per-structure
ordering is dataset-specific. What DOES transfer: every
structure tag on the public IT datasets shows negative signed
bias, i.e. v7 is uniformly under-confident OOD regardless of
structure.

### 4.5 Post-hoc head: three fits

Honest history of feature selection:

- **H1** (flawed): full 2005-label pool, found ECE 0.038 vs 0.070
  raw. Review caught that the pool included `merken_labels_v7.jsonl`
  -- v7 training data. Rerun on 982 clean events: ECE 0.049,
  delta vs Platt only -0.004. Aggregate verdict: REJECTED.
- **H11**: same features, C-sweep for L2 regularization. C=10
  yields ECE 0.043, max |w| = 2.23 (unregularized fit had |w|=8.33
  on n=13 slices).
- **H10 (shipped)**: added 5 interaction features after smoke
  testing H11 on "Fix: replaced pgbouncer..." (short + numbers +
  file_paths, raw 0.857 -> H11 head 0.371 -> desired recovery).
  C=1, ECE 0.035.

**Feature-engineering disclosure + leave-one-out ablation.** The 5
interaction terms were added after observing a specific smoke case,
which is post-hoc feature engineering. We ran a leave-one-out
ablation
(`experiments/h10_interaction_ablation.py` +
`experiments/nanogpt/h10_interaction_ablation.json`) to see how
load-bearing each interaction actually is:

| configuration | ECE | delta vs full |
|---------------|----:|---------------:|
| full (14 features) | 0.0354 | -- |
| drop `short_X_numbers` | 0.0354 | +0.0000 |
| drop `short_X_file_paths` | **0.0312** | -0.0043 (IMPROVES) |
| drop `short_X_markdown_table` | 0.0372 | +0.0018 |
| drop `short_X_inline_code` | 0.0414 | +0.0060 |
| drop `ood_X_short` | 0.0439 | +0.0085 |
| no interactions (9 features, H11) | 0.0477 | +0.0122 |

Two findings that are unkind to the original H10 narrative:

1. **The interaction I introduced to fix the smoke case
   (`short_X_numbers`) contributes zero**. The smoke case was
   "short + numbers + file_paths = concrete DEC"; I added
   `short_X_numbers` to capture exactly that. In the fit it takes
   weight -0.28 and removing it does not move ECE.

2. **Dropping `short_X_file_paths` improves ECE**. It has weight
   -0.33 in the full fit, i.e. it actively pushes the fit away
   from where the smoke case wanted to land. The full head
   compensates with other terms; a leaner head without this
   feature is measurably better-calibrated.

3. **Most of the gain over H11 (ECE 0.043 -> 0.035) comes from
   `ood_X_short` alone** (+0.0085 when dropped). The other four
   interaction terms net out to ~0.0037 of improvement.

Honest read: the 5-feature interaction set is over-specified for
this data. A 10-feature head (9 base + `ood_X_short` only) would
ship the essential signal with less overfitting risk. We have not
re-shipped the smaller head; the currently-shipped
`merken/classifiers/calibration_v7.json` still carries the full
14-feature fit. Proposed follow-up: re-ship with a 10-feature
minimal head and re-run the ablation against fresh data once
available.

### 4.6 Cross-distribution bias is opposite-direction

Combining the 494 agree_write events from Jay's transcripts
(where v7 is over-confident DEC by construction: all P(D) >= 0.6)
with the 98 Capybara events whose v7 P(D) fell below 0.3 (where
v7 is under-confident; total 592 events)
(`experiments/calibrate_v7_platt.py`): the two subsets have
opposite bias signs. A scalar Platt fit REGRESSES ECE from 0.145
(raw) to 0.177. Temperature scaling also regresses.

**This is a limit of scalar post-hoc methods, not a Pareto limit
of post-hoc methods overall.** A two-head calibrator (one per
domain, selected by a lightweight OOD indicator) would resolve it
trivially. We didn't fit one because we don't have enough
per-domain labels to support it and because the clean use case for
this classifier is Jay's transcripts (in-distribution), where a
single head suffices. Stating this as a "fundamental" or "Pareto"
limit would overclaim.

## 5. What this contributes

Three things. We pulled the two we had listed as "contributions" in
an earlier draft (shadow-mode methodology; honest iteration) --
neither is novel; both are basic engineering practice.

1. **The Production-Benchmark Gap is quantified on one concrete
   pipeline.** Same 800K-param model on the same week gives 86%
   store reduction on a synthetic scenario with 86% NOI events, and
   7.7% on real transcripts with 24% NOI events. Class-distribution
   mismatch fully accounts for the gap. This is a well-known
   phenomenon in ML more broadly, cited here as an instance + a
   specific magnitude for calibrated-filter work.

2. **Content-type-conditional calibration beats scalar scaling in
   distribution, fails equally across distributions.** We fit
   scalar Platt (2 params), regularized 9-param head, and
   14-param head with interactions on the same 982-event pool.
   Delta over Platt is -0.018 aggregate (0.053 -> 0.035), with
   per-structure wins up to -0.184 on markdown_table. A per-
   distribution head would plausibly cross-transfer too, but we
   did not fit one. Feature design for the 14-param head was
   post-hoc and is called out as a methodological limit.

3. **Real-label training redistributes raw output mass (bimodal
   93% -> 59%) but does NOT confer calibration the head can't
   recover.** v6+head reaches ECE 0.047, v7+head reaches 0.035,
   delta 0.012. Our earlier claim that "uncertainty detection is
   training-driven, not architectural" was too strong. The right
   framing: real-label training gives you interpretable probabilities
   without a calibration wrapper; for actual ECE either path works.

## 6. Limitations (expanded)

- **N=1 user** (same author as this memo).
- **Gemini-in-this-prompt is the oracle.** Human audit showed 15/20
  (75%) agreement on DECs; every accuracy number is bounded by
  Gemini's own reliability.
- **One held-out scenario contaminated** (jay_vstash_snapshot,
  13/20 events in training).
- **H10 feature design motivated by one smoke case.** Post-hoc
  feature engineering. No ablation.
- **No retrieval E2E.** Filter accuracy != memory usefulness. LoCoMo
  was run on prior versions but not v7 + calibration.
- **Benchmarks internal.** knowledge_update / organic_val_held_out
  / markdown_tables_held_out live in this repo and drove design.
  MemBench / LoCoMo evaluations are older and on prior versions.
- **Synthetic-test contamination risk unchecked on other scenarios.**
  Only verified jay_vstash_snapshot. Other 100% scores deserve an
  audit we didn't do.

## 7. Reproducibility

Every numeric claim above traces to a script in `experiments/` and
an artifact in `experiments/nanogpt/*.json`. One-shot regen:

```bash
# Label bootstrap (Gemini-backed, ~1-2K API calls, ~$1-2)
PYTHONPATH=. python -m experiments.bootstrap_from_transcripts
PYTHONPATH=. python -m experiments.relabel_decision_pile
PYTHONPATH=. python -m experiments.oracle_agree_write_sample
PYTHONPATH=. python -m experiments.oracle_public_dataset --dataset LDJnr/Capybara
PYTHONPATH=. python -m experiments.oracle_public_dataset --dataset Open-Orca/SlimOrca

# Analysis
PYTHONPATH=. python -m experiments.confusion_matrix_v7
PYTHONPATH=. python -m experiments.content_type_calibration
PYTHONPATH=. python -m experiments.h5_content_type_transferability
PYTHONPATH=. python -m experiments.h10_interaction_head
PYTHONPATH=. python -m experiments.h11_regularized_head
PYTHONPATH=. python -m experiments.v6_with_h10_head_ablation
PYTHONPATH=. python -m experiments.cross_version_calibration
```

Decision-rule log for every hypothesis: `HYPOTHESES.md`.
Checkpoint inventory: `MODELS.md`.

## 8. What would make this a submission

To move from N=1 memo to workshop paper, the gap is:

- N >= 3 users' transcripts (consent + labeling pipeline).
- A ~200-sample human-labeled subset as a second oracle so the
  accuracy numbers become accuracy-vs-human-adjudicated instead
  of accuracy-vs-Gemini.
- Retrieval E2E (LoCoMo or equivalent) with v7 +/- calibration.
- A clean scenario set explicitly separated from training data;
  re-audit every "100%" score.
- v9 with contrastive loss (H2) OR v10 with more params (H8) --
  to close the 157 FN remaining on real DECs without regressing
  markdown FPR.

Estimated: 4-6 weeks of focused work. Out of scope for this memo;
potential follow-up if and when multi-user data becomes available
or if a collaborator wants to drive the venue path.

## 9. Recommended disposition

Blog post / tech report to Jay's existing audience (r/LocalLLaMA,
merken users, vstash users). The honest-memo framing is the pitch:
specific numbers, acknowledged limits, negative results included.
Publishable as a workshop paper only after section 8 is done.
