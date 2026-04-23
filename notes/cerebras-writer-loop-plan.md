# Cerebras-writer loop: nanoGPT v8 for LongMemEval domain

Plan of record for the next session (2026-04-25+). Supersedes
the pipeline+brief_v1 path which was REJECTED 2026-04-24 with
-18.5pp on N=30 seed=44 (see `experiments/retrieval/longmemeval/
RESULTS.md` Step 3 section).

## Framing

v7 write filter was trained via distillation on Jay's work-log
(sprint/ticket/meeting/code vocabulary). It scored:
- 100% / 100% precision+recall on easy novel
- 90% on moderate novel
- 19% on subtle noise
(from `experiments/nanogpt/RESULTS.md`, 2026-04-16).

On LongMemEval seed=44 it landed 86.3% median answer-session
recall (2026-04-24 Step 1 smoke). Out-of-distribution bite.

v8 goal: train a LongMemEval-domain write filter so we can
measure whether substrate curation (via a filter, not via
briefs) moves RAG-k3 on LongMemEval.

## Labels: session-level is free

LongMemEval haystacks are structured as `answer_session_ids`
(ground-truth sessions containing the needle) + distractor
sessions. Every turn gets a free session-level label:

- `positive`: turn is in any `answer_*` session
- `negative`: turn is in a distractor session

Across 414 holdout questions (seed=42/43/44 N=30 removed),
this gives ~200k turns labeled at ZERO API cost.

Distribution (rough): ~15% positive / ~85% negative. Matches the
noise-heavy prior v7 was trained on.

Caveat: session-level labels are coarse. Chit-chat turns inside
answer sessions get `positive` even if they don't carry the
needle. That adds noise to positives. Two options:
- Accept it; ~95% of answer-session turns still contain useful
  context, and the filter's job is to distinguish "answer-like
  session" from "distractor-like session", not to isolate the
  single needle turn.
- Refine with Cerebras turn-level labels ONLY on ambiguous
  cases (step 2b below). Gated behind Phase 1 validation.

## Phase 1: train v8 on session-level labels (zero API spend)

### Data pipeline
- Input: 414 holdout LongMemEval_s questions = 19,753 sessions,
  203,913 turns.
- Train/val split: 80/20 at the QUESTION level (not turn level),
  to avoid train/val contamination where the same session is
  split across sets.
- Decontamination: seed=42/43/44 N=30 qids (86 unique) held out
  as TEST SET. Never used in training or validation.

### Model
- Same nanoGPT architecture as v7: BPE tokenizer, 4L/4H/128d,
  512 vocab, block_size=256.
- Same verb-marker prefix scheme if it applies to LongMemEval
  vocabulary. Otherwise train without markers.
- New tokenizer trained on LongMemEval-domain text (personal
  conversation about shopping, health, recipes, travel...) --
  v7's tokenizer was trained on work-log and will fragment
  LongMemEval words poorly.

### Training
- Binary classification: KEEP vs DROP (mapped to DECISION vs
  NOISE tokens for nanoGPT continuity).
- Standard cross-entropy on the label token.
- Training budget: ~2-4 hours on CPU, or 20-40 min on a
  consumer GPU. Same scale as v7.
- Output: `out-merken-bpe-v8-longmemeval/ckpt.pt`,
  `data/merken_bpe_v8_longmemeval/{meta.pkl, tokenizer.json}`.

### Validation gates
1. Val accuracy on held-out 20% questions >= 80%. Training
   sanity check; not the load-bearing metric.
2. Re-run Step 1 smoke with v8: answer-session recall on
   seed=44 N=5 questions. Target: **median >= 95%** (vs v7's
   90% median on same set).
3. If gate 2 passes, run N=30 smoke across all three seeds
   (42, 43, 44). Target: no regression on seed=42/43.

### Decision
- Gate 2 passes >= 95% median -> proceed to Phase 2.
- Gate 2 at 75-94% -> proceed with caveat, log as half-win.
- Gate 2 < 75% -> STOP. Session-level labels are too coarse.
  Move to Phase 2b (Cerebras refinement).

## Phase 2: pipeline integration and head-to-head vs RAG-k3

### Configuration
- v8 filter at ingestion (replaces AlwaysWrite from Step 3).
- NO briefs. Pure filtered-episodic RAG-k3.
- Dual retrieval (same as baseline).
- Same Builder, same oracle, same temperature 0.0.
- Arm 1: `v8 + RAG-k3` vs baseline `AlwaysWrite + RAG-k3`
  (19/30 = 63.3%).

### Hypothesis
**Critical reframing (Jay 2026-04-24 EOD):** vstash raw R@5 on
longmemeval_s N=500 is already **0.964** (RESULTS.md commit
5a6c820). The substrate is not the recall bottleneck. The gap
between 0.964 R@5 and 0.633 answer-correctness is the
**Builder composition gap**, not a retrieval gap.

So v8's win cannot come from R@5 (ceiling at ~0.99). It must
come from **top-1 precision**: a filtered pool means the top-1
dual-retrieval hit is more likely the needle, not a distractor
that looks similar. The Builder anchors on top-1 strongly under
the `rag_baseline` prompt; improving top-1 precision is the
lever.

Expected delta: +3 to +8pp on seed=44. Smaller than originally
drafted because the substrate is already near-ceiling on R@5.
If the hypothesis holds, the gain concentrates on questions
where the baseline picks the wrong top-1 (distractor session
ranked above answer session). Measurement: track top-1
source_id session match rate on both arms.

### Gate
Same as Step 3 plan:
- >= +10pp -> ship, replicate seeds 42/43 before headline.
- +5 to +10pp -> suggestive. Run arm 2 (`v8 + RAG-k3 + Mode C`).
- 0 to +5pp -> null. Diagnose and move on.
- negative -> filter HURTS. Diagnose.

## Phase 2b (conditional): Cerebras turn-level refinement

Only triggered if Phase 1 gate 2 fails (< 75% median recall).

### Approach
- For each training question, identify ambiguous turns: turns
  within answer sessions that look like chit-chat (short, no
  named entities, opening/closing greetings).
- Send ~2k of these turns to Cerebras `gpt-oss-120b` with a
  targeted label prompt: "Given this turn plus the question
  it's supposed to help answer, does this turn contain a
  specific fact relevant to the question?".
  - Model choice: `gpt-oss-120b` over `qwen-3-235b-a22b-instruct-2507`
    because (a) chain-of-thought is native to gpt-oss, matching
    the binary-relevance-with-reason task; (b) ~2-3x cheaper
    per token at similar quality for classification; (c) reasoning
    models express uncertainty natively, letting us gate on
    high-confidence labels.
- Cost: ~$0.15-$0.30 for 2k labels at gpt-oss-120b rates.
- Retrain v8 with refined labels on ambiguous cases.

### When to escalate from 2b to full Cerebras labeling
- If 2b improves recall by < 3pp on Phase 1's val set, the
  turn-level label quality isn't the bottleneck. Return to
  architecture / data-scale interventions.
- If 2b improves recall by >= 3pp, consider scaling to full
  2k -> 20k Cerebras labels (~$3).

## Budget

- Phase 1: zero API spend. Engineering: 4-6 hours.
- Phase 2: ~$0.30 oracle for N=30 validation. Bounded.
- Phase 2b (conditional): ~$0.30.
- Phase 3 (scaling): ~$3.
- Worst-case total: ~$5 + ~8 hours engineering.

## Invariants

- Seed=42/43/44 N=30 qids NEVER in training or validation data.
- v7 checkpoint stays untouched. v8 is a sibling, not a
  replacement in production. Production still ships v7 unless
  graduation criteria pass.
- brief_v1 prompt byte-for-byte identical to production.
- No corpus manipulation: AlwaysWrite as the sole "no-filter"
  baseline; v8 measured against that, not against a hand-tuned
  subset.
- Code review via code-reviewer subagent before ANY N=30 run
  that feeds RESULTS.md.

## Files to create next session

- `experiments/nanogpt/build_longmemeval_train.py` -- extract
  (turn_text, label) pairs from the 414 holdout questions.
- `experiments/nanogpt/train_v8_longmemeval.sh` -- invoke
  nanoGPT training with the new dataset.
- `experiments/retrieval/longmemeval/filter_recall_v8_smoke.py`
  -- port of filter_recall_smoke_n5.py but pointing at v8.
- `experiments/retrieval/longmemeval/pipeline_runner_v8.py` --
  variant of pipeline_runner.py with v8 as the write filter
  and NO briefs. Simpler than the brief variant.

## What this plan is NOT

- Not a brief_v1 improvement. Briefs are abandoned for
  LongMemEval per Step 3 rejection.
- Not a Judge post-hoc experiment. Judge (task #21) remains
  queued; the filter work is orthogonal and can run first.
- Not a production-filter replacement. v7 keeps shipping; v8
  is an experimental sibling until graduation gates pass on
  LongMemEval AND the existing work-log test suite doesn't
  regress.

## First session move

Step 0: inspect a sample of positive vs negative turns from
the 414-pool to verify label quality. If the chit-chat noise
inside answer sessions looks prohibitively bad, move straight
to Phase 2b. If it looks OK, proceed to Phase 1 training.

Cost: zero. Just reading text.

## Done-by milestones

- End of session 1 (planning + data prep): train/val JSONL
  shipped, Step 0 sample inspected, training script ready.
- End of session 2 (training): v8 ckpt trained, Phase 1 gate 2
  smoke run.
- End of session 3 (integration): Phase 2 N=30 seed=44 run,
  RESULTS.md updated.
- Session 4+ (conditional): Phase 2b or seed=42/43 replication
  depending on session 3 outcome.
