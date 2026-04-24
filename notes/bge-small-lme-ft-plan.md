# v5-ft: contrastive fine-tune of bge-small-en-v1.5 on LongMemEval

Plan of record for the next session. Supersedes the filter-classifier
paths (v7 OOD, v8a/v8b nanoGPT, v8c frozen-encoder MLP) which all
failed on this benchmark.

## Why this is the path

Three consecutive filter experiments failed on LongMemEval seed=44
N=5 validation (2026-04-24):
- v7 (production, distilled on work-log): 86.3% mean / 70.8% min
  answer-session recall. Bordered the gate.
- v8a (nanoGPT, 1:177 imbalanced): 0% recall. Collapsed to
  always-NOISE.
- v8b (nanoGPT, 1:5 balanced 5508 examples): 0% recall. P(DECISION)
  mean 0.115 for positives vs 0.146 for negatives -- worse than
  chance on the label, despite val loss 2.4252.
- v8c (frozen bge-small + 128-hidden MLP): 2.7% mean / 0% median.
  Val-set recall best 24.8% @ 2.9% precision.

**Diagnostic**: v8c's failure is the informative one. A classifier
on TOP of bge cannot recover the "contains a specific fact" feature
because **bge's embedding space encodes topic, not content type**.
Two turns -- "women hold 20 leadership positions" and "we should
discuss leadership structure" -- live near each other in bge space
because they share the leadership topic. bge is *designed* to do
this for passage retrieval; for our task it's the wrong invariant.

The feature has to be learned **inside the encoder**, not on top.

## Load-bearing invariants (Jay 2026-04-24 EOD)

1. **BEIR baseline must NOT regress.** bge-small-en-v1.5 is
   benchmarked on BEIR with the claim "hybrid > ColBERTv2 on 5/5".
   v5-ft ships as a SIBLING, not a REPLACEMENT, and must be opt-in
   per collection.
2. **LongMemEval R@k must go up.** v5-ft target: +5-10pp over
   zero-shot bge on seed=44 N=30 R@5 answer-session recall. Above
   that ceiling means the encoder acquired the fact-vs-topic feature.
3. **No training on test set.** seeds 42/43/44 N=30 qids (86 unique)
   NEVER enter training or validation data. Only 414 holdout
   questions.

## Architecture

Starting from `BAAI/bge-small-en-v1.5` (33M params, 384 dim).

Two fine-tune variants to try, in order:

### v5-lora (first attempt)
- LoRA adapters on the transformer blocks (rank 8-16, alpha 16-32).
- Trainable params: ~0.3-0.8M (vs 33M full). Lightweight.
- Ships as (base model, adapter weights). vstash needs adapter loading.
- Faster training, smaller risk of catastrophic BEIR regression.
- Expected: some but not maximal LongMemEval improvement; BEIR delta near zero.

### v5-full (fallback if v5-lora underdelivers)
- Full fine-tune, all 33M params trainable.
- Faster specialization but higher BEIR regression risk.
- Ships as complete model file (~100 MB).
- Validation on BEIR-mini gates the decision.

## Training data shape

From the 414 LongMemEval holdout questions (seed=42/43/44 N=30
decontaminated):

- Positive pairs (anchor, positive):
  - anchor = question text
  - positive = turn from answer_session that gpt-oss-120b labeled
    as "yes" (contains specific fact relevant to question)
  - Source: `experiments/nanogpt/longmemeval_positive_labels.jsonl`
    (1128 yes labels across 414 questions)
- Hard negative pairs (anchor, negative):
  - anchor = same question
  - negative = turn from SAME haystack but distractor_session OR
    answer_session-turn labeled "no" by gpt-oss (same topic,
    different content type -- the exact failure mode bge has today)
  - Source: same labels file + all distractor-session turns per qid
- In-batch easy negatives: other questions' turns cross-mixed
  (standard InfoNCE).

## Loss

InfoNCE / NT-Xent contrastive:
```
L = -log(exp(sim(q, p) / tau) / sum_k exp(sim(q, k_k) / tau))
```
where k_k ranges over {positive, hard negatives, in-batch easy negatives}.

Temperature tau: start 0.05 (bge's sentence-transformers default).

## Validation gates

### Gate 1: BEIR no-regression (non-negotiable)
- Run `experiments/results/beir_benchmark.json` style mini-BEIR
  subset (quick, ~5 min on bge-small). The benchmark already lives
  in the vstash repo.
- Threshold: R@5 on BEIR-mini must not drop >= 1.0pp vs zero-shot
  bge-small-en-v1.5.
- If > 1.0pp regression: LoRA rank too high OR data mix dominates
  LongMemEval. Reduce scope or mix more generic pairs.

### Gate 2: LongMemEval R@5 improvement (target, not gate)
- `experiments/retrieval/longmemeval/runner.py` already measures
  R@5. Run it with the fine-tuned encoder.
- Target: +5-10pp over zero-shot bge's 0.964 = >=0.98.
- Stretch: get top-1 session match rate up -- that's the real lever
  for Mode C / RAG-k3 answer quality (R@5 is near-ceiling; top-1 is
  where the Builder anchors).

### Gate 3: Downstream pipeline effect
- If Gates 1+2 pass, run pipeline_runner.py v2 with v5-ft encoder
  (via vstash collection-scoped model). Compare to baseline
  RAG-k3 = 63.3% on seed=44 N=30.
- Target: +3-8pp. The R@5 reframing (substrate is already 96.4%) means
  filter/encoder gains are bounded by top-1 precision improvements,
  not recall gains.

## Implementation plan

### Step 0: data prep (~30 min)
- `experiments/retrieval/longmemeval/build_contrastive_pairs.py`
  reads `longmemeval_positive_labels.jsonl` + the 414 holdout
  haystacks, emits `{anchor, positive, hard_negatives[3-5]}` triples.
- Save to `data/bge_lme_ft/train.jsonl`.

### Step 1: LoRA fine-tune (~10-30 min on MPS)
- `experiments/retrieval/bge_lme_ft/train_lora.py`.
- Use `sentence-transformers`'s `MultipleNegativesRankingLoss` or
  roll own with PyTorch + `peft` library for LoRA.
- Save adapter weights + base model snapshot.

### Step 2: BEIR-mini no-regression check
- Copy vstash's BEIR runner, point at v5-lora encoder.
- If pass, proceed to Step 3. Otherwise tune rank/data-mix and re-run.

### Step 3: LongMemEval R@5 validation
- Existing `experiments/retrieval/longmemeval/runner.py` with the
  new encoder as the vstash config.
- Compare zero-shot vs v5-lora on seed=44 N=30.

### Step 4: pipeline validation (conditional on Steps 2+3 passing)
- Run pipeline_runner_v2.py with v5-lora encoder.
- Seed=44 N=30. Compare to baseline RAG-k3 63.3%.

### Step 5: document + commit
- `experiments/retrieval/longmemeval/RESULTS.md` section.
- Head-to-head table: zero-shot bge vs v5-lora vs (optional) v5-full.

## Risk list

- **Data scarcity**: 1128 positives after gpt-oss labeling may be too
  few for encoder-level training. If LoRA doesn't converge, consider
  augmenting with (a) session-level positives (all answer-session
  turns regardless of gpt-oss verdict), (b) synthetic pairs via
  Cerebras.
- **Domain leakage**: LongMemEval answer patterns might correlate
  with specific phrasings that the encoder learns to match even
  without real generalization. Mitigation: held-out seed=42/43
  validation AFTER seed=44 gate, not before.
- **BEIR regression**: expected near-zero for LoRA at rank <= 16.
  For full fine-tune, real risk. Ship only if BEIR-mini delta
  <= -1.0pp.
- **Deployment surface change**: vstash needs a "per-collection
  encoder" config to ship this opt-in. Check current vstash API
  supports this; if not, it's a vstash PR first.

## What this plan is NOT

- **Not a bge replacement**. v5-ft is a sibling, opt-in per collection.
- **Not seed=44-specific tuning**. If v5-lora shows +8pp on seed=44
  but -5pp on seed=42/43, it's overfit to seed=44 shape. Reject.
- **Not a nanoGPT experiment**. v7 stays shipped. v8a/b/c results
  are filed; not resurrected.

## First session move

Step 0 data prep. Read `longmemeval_positive_labels.jsonl` + the 414
holdout questions, emit contrastive triples. Inspect 20 triples
manually for quality. Cost: zero API. Wall: 15 min.

Before Step 1 (LoRA training), get code-review pass on the trainer
per CLAUDE.md invariant.

## Files to create next session

- `experiments/retrieval/bge_lme_ft/build_contrastive_pairs.py`
- `experiments/retrieval/bge_lme_ft/train_lora.py`
- `experiments/retrieval/bge_lme_ft/beir_mini_check.py` (no-regression guard)
- `experiments/retrieval/bge_lme_ft/RESULTS.md` (initially empty)

## Files to re-read on resume

- `notes/cerebras-writer-loop-plan.md` -- the parent plan; v5-ft
  is a sibling pivot.
- `experiments/retrieval/longmemeval/RESULTS.md` -- R@5 0.964
  baseline, Step 3 rejection.
- `experiments/nanogpt/longmemeval_positive_labels.jsonl` -- the
  gpt-oss labels (source of truth for contrastive triples).
- `experiments/nanogpt/v8c_frozen_encoder/meta.json` -- confirms
  zero-shot bge encoder id.
