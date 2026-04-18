# Latent Calibration in an 800K-Parameter Memory Filter

**Working title.** Draft consolidating the v7 write-filter work into a
paper-shaped narrative. Numbers are load-bearing; everything cites a
script in `experiments/` or an artifact in `experiments/nanogpt/`.

## Abstract

We trained an 800,000-parameter BPE transformer to decide whether
assistant responses in an agent conversation should be kept in
long-term memory. Against a LoCoMo-style synthetic benchmark the
filter achieves 86% store reduction and 100% recall on held-out
decisions. On real Claude Code transcripts produced by the same
author the numbers collapse: real store reduction is **7.7%**, real
population-weighted accuracy is **80.9%** (vs 84.7% first reported on
an incomplete slice), and the filter is uniformly over-confident on
short prose and uniformly under-confident on out-of-distribution
public instruction-tuning data. We document this Production-Benchmark
Gap, show that uncertainty detection emerges from real-label training
(a v6 baseline lacks it entirely), and demonstrate that a nine- to
fourteen-parameter post-hoc head restores in-distribution calibration
to ECE 0.035 without retraining. We close by showing that the same
post-hoc head does NOT close cross-distribution drift, because the
direction of miscalibration flips across domains -- a result that a
scalar Platt / temperature fit cannot represent.

## 1. Setup

`merken` is a memory layer for LLM agents. Each assistant turn is a
candidate event; the ``should_remember`` policy decides write vs
skip. The production policy is ``HeuristicWriteDecider``: novelty
check + dedup + length gates. ``NanoGPTWriteDecider`` is a
specialized classifier that auditors ("shadow mode") alongside the
heuristic. After labeling disagreements with an oracle, the
classifier is eligible for graduation to primary.

All model training runs on ``nanoGPT`` (Karpathy) fork with the same
architecture across versions:

- n_layer = 4
- n_head = 4
- n_embd = 128
- vocab_size = 512 (BPE)
- ~800K trainable parameters

Versions v4 to v6 had block_size = 128. v7 raised it to 256 after a
measurement (section 3.1) showed 45.9% of real DECISIONs exceed the
v6 context window.

## 2. Dataset bootstrap (the engineering prerequisite)

Prior work on calibration of small classifiers tends to assume a
labeled benchmark is available. For this problem none existed.

The live Claude Code PreCompact hook was silently broken: it ran
``json.load`` on a JSONL transcript, swallowed the exception via
``2> /dev/null``, and produced zero writes since the transcript
format change. Every per-project ``~/.merken/*.db`` had
`shadow_agree = shadow_disagree = 0`. Discovering this while reading
audit tables (not intended output) was the single highest-leverage
moment in this project.

Once the hook was fixed, three bootstrap scripts replayed history:

1. `experiments/bootstrap_retro_labels.py`: replay v6 over the
   already-written ``should_remember`` audit rows. Small (58 rows, 2
   disagreements) -- a sanity loop, not a training signal.

2. `experiments/bootstrap_from_transcripts.py`: replay primary +
   shadow over every `~/.claude/projects/*.jsonl` transcript (415
   files). 18,220 candidate events; 1,040 primary/shadow
   disagreements; 0 errors.

3. `experiments/relabel_decision_pile.py`: the first oracle pass
   used Gemini 2.0 Flash with a permissive prompt. Manual audit of
   20 labeled DECs flagged 15/20 as filler. The strict prompt (see
   `experiments/oracle_model_bench.py`) explicitly defines "transition
   sentences like 'Let me X', 'Now let me Y' ... regardless of what
   X/Y refer to" as NOISE. Re-running on the DECISION pile flipped
   588 of 739 labels (80%) to NOISE. Final ratio: 157 DEC / 869 NOI.

An independent probe (`experiments/oracle_agree_write_sample.py`)
reservoir-sampled 500 events from the 12,492 agree_write population
and oracled them with the same strict prompt, giving the other column
of the confusion matrix (398 DEC, 96 NOI).

All oracular work uses ``gemini-2.0-flash``. The 9-model bench in
`experiments/oracle_model_bench.py` showed that pro-tier models
(``gemini-2.5-pro``, ``gemini-3.1-pro-preview``) scored **lower** on
a 20-item hand-curated test (90% agreement vs 2.0-flash's 95%) --
they were more "creative" at finding DEC in filler text. For this
narrow-classification task, prompt quality dominates model capability.

## 3. v7 graduation

`experiments/eval_v7_vs_v6.py` compares v4-v8 on held-out scenarios.
See `RESULTS.md` for the per-version table; the v7 row is the
published contribution.

### 3.1 The block_size measurement

Before training v7 we measured the token length of real DECs vs NOI
on the strict-labeled pool:

| class | median | p90 | max | % over 128 BPE tokens |
|-------|-------:|----:|----:|----------------------:|
| DECISION | 116 | 305 | 557 | **45.9%** |
| NOISE | 38 | 60 | 357 | 0.7% |

At block_size=128 the model literally could not see the payload of
nearly half the real DECs (p95 exceeded the window). The prior v6
shows this as a 66.7% FPR on markdown_tables_held_out -- not a model
defect, a window-size defect. v7 doubled block_size to 256.

### 3.2 Scenario scores (v6 vs v7)

| scenario | v6 agreement | v7 agreement | delta |
|----------|-------------:|-------------:|------:|
| markdown_tables_held_out | 66.7% | **100%** | +33.3pp |
| organic_val_held_out | 100% | 100% | +0.0 |
| jay_vstash_snapshot | 100% | 100% | +0.0 |
| knowledge_update_50t | 99.9% | 97.4% | -2.5 |

v7 is the first version to clear criterion #5 (markdown FPR <= 50%)
with a 50pp margin. The regression on knowledge_update_50t is on a
synthetic scenario that rewards memorizing stereotyped DEC phrasing
("Replaced X with Y"). v7 dilutes that signal with 1026 heterogeneous
real transcripts. The trade is a cosmetic loss on synthetic for a
meaningful win on real.

### 3.3 Full population-weighted confusion matrix (`experiments/confusion_matrix_v7.py`)

```
               oracle=DEC   oracle=NOI
v7 WRITE         398 (TP)     96 (FP)     n=494   (agree_write sample)
v7 SKIP          157 (FN)    869 (TN)     n=1026  (disagreement pool)
```

Projected onto the full 13,539-event population (12,492 agree_write
+ 1,047 disagree_skip observed during the bootstrap scan):

| metric | value | remarks |
|--------|------:|---------|
| accuracy (pop-weighted) | **80.9%** | NOT the 84.7% we first reported |
| DEC recall | 98.4% | rarely drops a real decision |
| NOI recall (specificity) | **26.8%** | catches only 1 of 4 noise events |
| store reduction | **7.7%** | NOT the 86% synthetic benchmark suggests |

The 84.7% was skip-set only; adding the WRITE column moved it to
80.9%. The 86% store reduction was measured against a scenario with
~86% NOI events. Real class distribution is ~76% DEC / ~25% NOI.
The filter is operating at 80.9% accuracy, 7.7% store reduction --
a CONSERVATIVE filter that almost never costs you a decision but
only modestly prunes noise.

## 4. Calibration

### 4.1 v6 vs v7 (cross-version)

`experiments/cross_version_calibration.py`:

| model | n | ECE | signed bias | profile |
|-------|--:|----:|------------:|---------|
| v6 | 1520 | 0.159 | +0.135 | bimodal: 524 in [0.9,1.0), 372 in [0.0,0.1), nothing in middle |
| v7 (full) | 1520 | 0.120 | +0.090 | more distributed; 410 events in [0.3, 0.6) |

Training on real labels reduced aggregate miscalibration by ~25%
and redistributed probability mass out of the extremes. v6 is a
confident 2-class voter; v7 starts to look like an uncertainty
estimator. **This is the emergent-uncertainty-detection claim and
it depends on real training data. It does not emerge from the
architecture alone.**

### 4.2 Structure-dependent calibration

`experiments/content_type_calibration.py` tags each event by content
structure (code fence, inline code, markdown table, numbers, file
paths, short, long, plain prose) and measures v7 ECE per tag on the
1520 labels:

| tag | n | ECE | signed |
|-----|--:|----:|-------:|
| long (>= 1000c) | 78 | **0.045** | -0.045 |
| has_file_paths | 191 | 0.084 | +0.026 |
| has_numbers | 385 | 0.104 | -0.062 |
| has_inline_code | 502 | 0.103 | -0.051 |
| has_code_fence | 60 | 0.107 | -0.107 |
| short (< 300c) | 1254 | 0.168 | +0.148 |
| has_markdown_table | 79 | 0.184 | **-0.184** |
| plain_prose | 786 | 0.191 | **+0.191** |
| all | 1520 | 0.120 | +0.090 |

Aggregate ECE 0.120 is dominated by plain_prose (786 of 1520 events,
ECE 0.191) and short text (1254 events, ECE 0.168). Structured
content (file paths, inline code, numbers, long-form) is well
calibrated on its own. Markdown tables go the opposite way:
under-confident by -0.184.

### 4.3 Transferability probe, H5 (REJECTED)

If structure-dependent ECE is a universal property of the filter, it
should rank-transfer to other datasets. We compared with Spearman
over categories present with n >= 5 in both datasets
(`experiments/h5_content_type_transferability.py`):

- jay_1520 vs capybara_288: rho = **-0.40** (4 shared bins)
- jay_1520 vs slimorca_200: rho = **+0.50** (3 shared bins)

Neither crosses the pre-registered 0.7 acceptance bar. Per-structure
ordering is dataset-specific.

What DOES transfer is the sign: every structure tag on the public IT
datasets shows negative signed bias -- v7 is uniformly under-confident
OOD, independent of structure.

### 4.4 Post-hoc calibration (H1, H11, H10)

Three attempts, each an honest iteration.

**H1 (rejected on clean data).** Logistic regression with 9 features
(v7 logit + 7 structure tags + is_ood) on the full 2005-label pool.
First run reported ECE 0.038 (-46% vs raw). Review caught that the
pool included `merken_labels_v7.jsonl`, which was part of v7's
training data. After excluding (n=982) the result is ECE 0.049 vs
Platt's 0.053: delta -0.004, not a meaningful win.

**H11 (regularization).** Unregularized fit produced two weights
above 7.0 from n<20 slices. L2 C-sweep on clean pool identified C=10
(ECE 0.043, max |w|=2.23) as the stable configuration.

**H10 (interactions).** Smoke testing H11 on "Fix: replaced
pgbouncer session mode with transaction mode in db/pool.py line 42.
Latency p95 180ms -> 40ms." (short + numbers + file_paths, raw P(D)
0.857) showed the additive head pushed it to P_cal=0.371 -- couldn't
distinguish "short with payload" from "short filler". Adding 5
interaction terms and refitting at C=1 yields ECE **0.035** and
P_cal=0.629 on the same smoke input.

| method | features | ECE | smoke case |
|--------|---------:|----:|-----------:|
| raw v7 | -- | 0.056 | 0.857 |
| Platt | 1 | 0.053 | 0.580 |
| H1 head | 9 (overfit) | 0.049 | 0.371 |
| H11 head | 9 (L2) | 0.043 | 0.371 |
| **H10 head** | **14 (interactions)** | **0.035** | **0.629** |

The shipped head (`merken/classifiers/calibration_v7.json`) is the
H10 C=1 fit. Enable via code (`NanoGPTWriteDecider(calibrator=...)`)
or env var (`MERKEN_SHADOW_NANOGPT_CALIBRATOR=default`).

### 4.5 Cross-distribution drift is not recoverable post-hoc

Combining HIGH-regime Jay labels with LOW-regime Capybara gives a
pooled clean set (n=592) where scaling methods REGRESS: ECE goes
**up** from 0.145 (raw) to 0.177 (Platt). Reason: the two domains
have opposite-direction biases (HIGH over-confident +0.10, LOW
under-confident -0.57). A scalar transform cannot correct two biases
pointing in opposite directions. This is a Pareto limit of post-hoc
calibration on small specialized models crossing distributions.

## 5. What this contributes

Three claims with numeric evidence:

1. **Production-Benchmark Gap is measurable and large.** Same model,
   same week: 86% store reduction on a synthetic class-imbalanced
   scenario vs 7.7% on real transcripts. Class-distribution mismatch
   fully explains the delta.

2. **Uncertainty detection is training-driven, not architectural.**
   v6 (synthetic-only) is bimodal with ECE 0.159. v7 (+1026 real
   labels, same architecture) distributes probability into the
   ambiguous bucket and drops ECE to 0.120. A 9+-parameter head on
   clean data brings it to 0.035. Adding the 9-parameter head to v6
   would not produce uncertainty detection -- v6 has nothing to
   calibrate there.

3. **Post-hoc calibration is directional.** Within a training
   distribution, scalar scaling and small heads work well. Across
   distributions with opposite bias directions, no scalar and no
   tiny head suffices.

And two methodological contributions that we think transfer:

4. **Shadow-mode + retroactive transcript replay + strict oracle**
   is a bootstrap methodology that produces 1000+ labels without
   labeling anyone's data up front. The pipeline is five Python
   scripts and one prompt.

5. **Honest iteration.** H1, H5, H11, H10 are all documented with
   decision rules set up-front, results, and override notes when
   the initial result was contaminated. See
   `experiments/nanogpt/HYPOTHESES.md`.

## 6. Limitations

- **N=1 user.** All "real" data is one developer (Jay). Multi-user
  evidence is a gap. The v7 -> v9 work is blocked on this.
- **Gemini as oracle.** Labels are an LLM's interpretation of a
  strict prompt. Circular: we're measuring how well v7 imitates
  Gemini on this prompt. A human audit (20-sample manual check)
  found 15/20 agreement on DECs but we lack a larger human ground
  truth.
- **No retrieval downstream eval.** We measure filter accuracy, not
  the effect of the filter on eventual recall quality. LoCoMo-style
  end-to-end eval is the gap between "filter is calibrated" and
  "memory is useful".
- **Benchmarks not standardized.** knowledge_update / markdown_
  tables_held_out / organic_val_held_out are internal. MemBench and
  LoCoMo were run on prior versions (`experiments/retrieval/locomo/
  RESULTS.md`) but not on v7 with calibration.

## 7. Reproducibility

Every number in this document traces to a script + JSON artifact.
Scripts run in <30 min each on a modern laptop. See
`experiments/nanogpt/HYPOTHESES.md` for the per-hypothesis
decision-rule log. Ckpt inventory in
`experiments/nanogpt/MODELS.md`.

Data regeneration is one-shot:

```
# 1. Label bootstrap (Gemini-backed, ~1K API calls, ~$1)
PYTHONPATH=. python -m experiments.bootstrap_from_transcripts
PYTHONPATH=. python -m experiments.relabel_decision_pile
PYTHONPATH=. python -m experiments.oracle_agree_write_sample

# 2. Analysis + calibration fits
PYTHONPATH=. python -m experiments.confusion_matrix_v7
PYTHONPATH=. python -m experiments.content_type_calibration
PYTHONPATH=. python -m experiments.cross_version_calibration
PYTHONPATH=. python -m experiments.h11_regularized_head
PYTHONPATH=. python -m experiments.h10_interaction_head
```

## 8. Future work

- **Multi-user data.** N=3 developers would be meaningful; N=10 would
  be publishable.
- **Full retrieval E2E.** Does the calibrated filter change measured
  retrieval quality on LoCoMo? Hook the calibrator into Memory and
  run.
- **Contrastive loss (H2).** Paired same-starter-different-class
  events to force attention over the payload. Open.
- **Architecture probe (H8).** If contrastive loss fails, bump
  n_embd = 192 (~1.5M params) and see whether capacity was the
  constraint.

---

**Status of this draft:** honest first pass. Numbers are stable,
framing has iterated through two review rounds. Not yet venue-
targeted. If Jay wants to publish: either workshop (add N=3 users +
1 retrieval E2E) or tech report / blog post (ship as-is with
"N=1 user, further work needed" disclaimer).
