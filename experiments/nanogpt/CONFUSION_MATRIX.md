# v7 Confusion Matrix, Distribution Gap, and Emergent Uncertainty

Findings from the 2026-04-18 evaluation session. Three scripts
produced the numbers in this doc; all are reproducible:

- `experiments/confusion_matrix_v7.py`
- `experiments/pd_distribution_analysis.py`
- `experiments/calibration_curve_v7.py`

## 1. Full confusion matrix on 1520 oracled events

Built from two label sets:
- `data/merken_labels_v7.jsonl` (1026 shadow_skip events, strict-oracle).
- `data/merken_labels_agree_write.jsonl` (494 agree_write, reservoir
  sample, same strict-oracle via `gemini-2.0-flash`).

```
                 oracle_DEC   oracle_NOI
  v7 WRITE        398 (TP)     96 (FP)
  v7 SKIP         157 (FN)    869 (TN)
```

### Sample-level metrics (n=1520)

| Metric | Value |
|---|---:|
| Accuracy | 83.4% |
| DEC precision | 80.6% |
| DEC recall | 71.7% |
| NOI precision | 84.7% |
| NOI recall (specificity) | 90.1% |

### Population-weighted metrics

Sample shares are projected onto the observed population of 13,539
events (12,492 agree_write + 1,047 disagree_skip from the full
bootstrap scan over 415 transcripts).

| Metric | Value | Notes |
|---|---:|---|
| True DEC share in population | 75.5% | content is mostly decisions |
| True NOI share | 24.5% | the noise the filter should catch |
| v7 accuracy | **80.9%** | honest number, not the 84.7% we first reported (that was skip-only) |
| v7 DEC recall | **98.4%** | rarely drops a real decision |
| v7 NOI recall | **26.8%** | catches only 1 of 4 noise events |
| v7 store reduction | **7.7%** | real vs synth's 86% claim |

## 2. Reality Gap: 86% -> 7.7% store reduction

v6 was reported at ~86% store reduction on the
`knowledge_update_50topics` scenario. That number is real, but the
scenario is synthetic: 950 NOI of 1100 events. When class balance in
the test matches the filter's training bias, the filter looks
powerful.

On Jay's real transcripts the class share flips. NOI is 24.5% of
events, not 86%. The upper bound on store reduction is 24.5% even
for a perfect classifier. v7 captures about a third of that ceiling
(7.7% / 24.5% ~= 31% of attainable noise caught).

**Implication:** benchmark scores on synthetic class-imbalanced
datasets overestimate production store reduction by an order of
magnitude. This is not a nanoGPT-specific failure mode -- it applies
to any classifier benchmarked on datasets with non-representative
class priors.

## 3. Distribution gap across three data sources

From `pd_distribution_analysis.py`, v7 P(D) distribution by source:

| Source | n | mean P(D) | median | ambiguous [0.3-0.7) |
|---|---:|---:|---:|---:|
| REAL Jay transcripts (all) | 1520 | 0.455 | 0.404 | **29.1%** |
| SYNTH knowledge_update_50t | 1100 | 0.106 | 0.001 | 2.8% |
| SYNTH markdown_tables_held_out | 12 | 0.600 | 0.746 | 25.0% |
| PUBLIC Capybara (LDJnr) | 288 | 0.828 | 0.918 | 10.1% |

Two findings:

**(a) v7 spends more probability in mid-range on real than on
synthetic.** The ambiguous bucket is 10x wider on real data than on
knowledge_update. This is not because v7 becomes "unsure" as a
reaction to real data; it's because real data contains more content
whose class is actually ambiguous by construction (transitional
sentences carrying some payload, task announcements with numbers,
etc.).

**(b) Capybara (public IT-tuning dataset) is pushed hard toward
confident DEC.** 84% of Capybara chunks get P(D) >= 0.7. Gemini
agrees: oracle labeled 273/288 (94.8%) as DECISION. Capybara is
curated instruction-response content -- it has almost no filler.
This tells us public IT-tuning datasets are not drop-in substitutes
for agent-transcript data when training a write filter.

Class distribution across sources:

| Source | DEC share | NOI share |
|---|---:|---:|
| REAL Jay (population est.) | 75.5% | 24.5% |
| Capybara (oracle-labeled) | 94.8% | 4.9% |
| synth knowledge_update | 13.6% | 86.4% |

Capybara's 94.8% DEC is the opposite problem from synth
knowledge_update's 86.4% NOI. Neither matches real.

## 4. Calibration curve: uncertainty without calibration

From `calibration_curve_v7.py` on the 1520 events, binned by v7 P(D):

| bin | n | mean P(D) | actual DEC | gap |
|---|---:|---:|---:|---:|
| [0.0, 0.1) | 376 | 0.034 | 0.096 | 0.062 |
| [0.1, 0.2) | 134 | 0.144 | 0.142 | 0.003 |
| [0.2, 0.3) | 106 | 0.247 | 0.217 | 0.030 |
| [0.3, 0.4) | 140 | 0.348 | 0.186 | 0.162 |
| [0.4, 0.5) | 123 | 0.451 | 0.195 | 0.256 |
| [0.5, 0.6) | 147 | 0.554 | 0.197 | **0.356** |
| [0.6, 0.7) | 32 | 0.651 | 0.531 | 0.120 |
| [0.7, 0.8) | 53 | 0.754 | 0.528 | 0.226 |
| [0.8, 0.9) | 66 | 0.858 | 0.697 | 0.161 |
| [0.9, 1.0) | 343 | 0.962 | 0.895 | 0.067 |

**ECE = 0.120 (mildly miscalibrated).**

v7 is well-calibrated at the extremes (gap < 0.07 in the [0.0, 0.2)
and [0.9, 1.0) bins). Mid-range is systematically overconfident in
DEC: when v7 emits P(D)=0.55 the true DEC rate is only 19.7%. The
training class prior (~87% DEC) leaks into the mid-range outputs as
a lean toward DEC.

**Two observations, both important:**

- Uncertainty DETECTION emerges. v7 places more real-content events
  in the mid-range than any synthetic scenario does -- it "knows"
  when the input is ambiguous.
- Calibration does NOT emerge. v7's P(D)=0.5 is not "50% chance DEC"
  in practice; it's closer to "20% DEC, 80% NOI, but the training
  prior pulls the output up to 0.5".

## 5. Paper angle

Three distinct findings that together form a narrative:

1. **Production-Benchmark Gap on store reduction.** Same model, same
   day: 86% synthetic -> 7.7% real. Traceable to class-distribution
   mismatch between benchmark and deployment.

2. **Emergent uncertainty detection in 800K params.** v7 spends 10x
   more probability mass in the ambiguous bucket on real content vs
   synthetic. The tiny model recognizes when input is hard without
   being explicitly supervised to do so.

3. **Calibration does not emerge.** Mid-range P(D) outputs are
   systematically overconfident in DEC because training priors leak
   through. Temperature / Platt scaling on mid-range bins is a
   plausible fix and would be a tight follow-up experiment.

Working title: *"Emergent Uncertainty Detection without Calibration
in a 800K-Parameter Memory Filter: Evidence from the Production-
Benchmark Gap."*

## 6. What this session shipped

- `experiments/oracle_agree_write_sample.py`: reservoir sample of
  agree_write events + strict-oracle labeling. Produces the 494-row
  write-side dataset that closes v7's confusion matrix.
- `experiments/oracle_public_dataset.py`: generic oracle runner for
  HuggingFace datasets. Default extractor for `LDJnr/Capybara`;
  easily extended to other sources.
- `experiments/confusion_matrix_v7.py`: computes the full 2x2 matrix
  + sample metrics + population-weighted metrics.
- `experiments/pd_distribution_analysis.py`: P(D) histograms by data
  source + H1-vs-H2 ambiguity-share verdict.
- `experiments/calibration_curve_v7.py`: 10-bin calibration table
  + ECE.

All five scripts are idempotent and depend only on the JSONL
artifacts already committed / gitignored.

## 7. Next-session choices

Natural follow-ups, each self-contained:

- **A. Platt / temperature scaling on mid-range bins.** Adds a
  post-hoc calibration layer without touching v7. Re-compute ECE,
  report delta. ~30 min work.
- **B. Expand public-dataset probes.** `SlimOrca`, `OpenHermes-2.5`,
  and `orca-agentinstruct-1M-v1` have different class distributions
  still. Would strengthen the "no public dataset matches real agent
  transcripts" claim. ~30 min per dataset.
- **C. Write the paper / tech report.** All numbers needed are now
  in place.

Jay explicitly said no publishing interest yet, so **C is parked**.
A and B are low-cost additions.
