# Phase 3: NanoGPTMidloopDecider training plan

The Phase 1 pipeline (PR #20-23) produced a 1136-intervention
dataset over 385 cases drawn from 77 WHO/IMCI clinical protocols.
The Phase 2 primitive (PR #22) shipped the runtime scaffolding
(``StepObservation`` -> ``MidloopDecision``) with three reference
deciders, of which only the heuristic one is non-trivial.

This note plans Phase 3: a trained classifier that replaces the
heuristic. **It is design only -- no training run lives in this
note.** When the plan ships as code, it will follow the
NanoGPTWriteDecider precedent (PRs #1, #4, #5, #11) -- shadow
mode first, calibrated against fresh disagreements, graduated
only when the numeric bars are met.

## What the model needs to predict

Per the midloop spec section "NanoGPTMidloopDecider":

> Es decisión binaria intrínseca al pipeline de generación...
> token-level con sampling adaptativo.

The runtime question at inference time is:

> Given the user's prompt + the tokens the LLM has already emitted,
> SHOULD I intervene RIGHT NOW (at the next token boundary)?

So the prediction is:
- input: `prompt + response_tokens[:i]` (causal context)
- output: per-position binary `intervene` / `continue`

The training data shape must match this. Phase 1 produces aligned
divergence SPANS, not per-token labels. There are three reasonable
label-shape choices; each makes the model learn a different signal.

## Three label-shape options

### Option A. Span-inside (the obvious naive choice)

Label `1` at every token position that lies inside any intervention
span; `0` elsewhere.

```
truth:  REFER URGENTLY to a hospital with surgical capacity
model:  Administer IV fluids and obtain urgent abdominal ultrasound

response_tokens:    Administer IV fluids and obtain urgent abdominal ultrasound
intervene_labels:   1          1  1      1   1      1      1         1
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    one big span [0, 8) covers the whole response
```

**Properties measured on the 385-case dataset:**
- 8053 total response tokens
- 6965 positive (86.5%) -- the small model's response is wrong-most-of-the-time
- positive-per-case median is 16 of 21 tokens

**Why this is BAD for training:**
The class imbalance is the wrong way around. A trivial "always
predict 1" baseline gets 86.5% accuracy. The model would need to
learn the specific tokens that DON'T need intervention (typically
bridge words like "and", "to", "the" that overlap with the truth
sequence in the diff alignment). Those bridge words are not
clinically meaningful; learning them just memorizes diff artifacts.

This option is documented for completeness only. **DO NOT USE.**

### Option B. Span-boundary (recommended for v0)

Label `1` ONLY at the START of each intervention span; `0` elsewhere.

```
response_tokens:    Administer IV fluids and obtain urgent abdominal ultrasound
intervene_labels:   1          0  0      0   0      0      0         0
                    ^
                    only the start of [0, 8) gets labeled
```

For a case with 3 intervention spans, exactly 3 positive labels
(one per span start). Per-case positive count = number of
divergence regions = 1136 / 385 ~= 3 per case ~= 14% positive token
density.

**Properties (estimated):**
- 8053 total response tokens (same)
- ~1136 positive (~14.1%) -- one per span
- positive-per-case median ~3 of 21 tokens

**Why this is the right v0 signal:**
It teaches the model "where does an error START" -- the actionable
signal at inference time. The runtime can then flag intervention
RIGHT THERE, before the model commits to the whole bad span.
Detecting the span start is the production-relevant decision; the
fact that tokens 1-7 of the span are also wrong is downstream
context, not a separate prediction.

The 14% positive rate is still imbalanced but tractable. Class
weighting at ~6:1 (negative_weight=1, positive_weight=6) brings
loss contributions roughly even.

**This is what v0 should ship.**

### Option C. Case-level binary (the lazy fallback)

One label per case: "did the response need intervention at all?"

For our dataset every case has at least one intervention, so every
case is labeled 1. Useless as a training signal -- nothing to
discriminate. Documented to explicitly close it off as not worth
trying.

## Recommended dataset format (v0)

`format_for_training.py` emits Option A by default (it's the
straightforward span-inside computation). For Phase 3 v0, switch
to Option B by passing `--labeling boundary`. Output schema stays
the same; only the label vector changes.

```jsonl
{
  "case_id": "acute-abdomen-who__case_000",
  "prompt": "<CHW question>",
  "response_tokens": ["Administer", "IV", "fluids", ...],
  "intervene_labels": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],
  "intervention_spans": [...],   // for traceability + eval
  "metadata": {"protocol_id": "...", ...}
}
```

The training loader reads this directly. No tokenization
mismatch concern -- response_tokens already match
merken/training/midloop_dataset.py:tokenize.

## Architecture sketch

Same recipe as NanoGPTWriteDecider (the v7 graduated baseline) but
with a sequence-tagging head instead of a sequence-classification
head:

| | NanoGPTWriteDecider v7 | NanoGPTMidloopDecider v0 |
|---|---|---|
| arch | 4L/4H/128d, block=256, vocab=512 | same |
| param count | ~800K | ~800K (+ tiny tagging head) |
| loss | cross-entropy on a single label token after `<|label|>` | per-position BCE on intervene_labels |
| input | one event text | prompt `<sep>` response_tokens (causal) |
| output | DECISION / NOISE token logits at the last position | per-position {intervene: 0 or 1} |
| dataset | 3046 events (1026 transcript + 1922 synth + ...) | 385 cases (Phase 1 dataset) |
| training | next-token CE | per-position BCE w/ class weight ~6:1 |

The same nanoGPT codebase + tokenizer infrastructure ports. The
only real change is a `tagging_head: Linear(n_embd, 1)` applied to
EVERY position's hidden state instead of just the last.

## Data prep before training

1. Hold out a TEST set: 10% of cases stratified by protocol_id so
   no protocol leaks across train/test. ~38 test cases / 350
   train cases.
2. Train BPE tokenizer on prompt + response text only (NOT on
   labels -- those are post-hoc per-position).
3. Materialize a single sequence per case:
   `[BOS] <prompt tokens> [SEP] <response tokens> [EOS]`
   Labels: `0` for prompt + sep, `intervene_labels` for response
   tokens, `0` for EOS. Loss masked to response region only.

## Loss + training

- Per-position binary cross-entropy with `pos_weight = 6.0` (rough
  inverse of the 14% positive rate). Calibrate on a sweep
  [3, 5, 6, 8, 10] against held-out F1.
- Loss masked: only response-region positions contribute. Prompt
  + special tokens excluded from gradient.
- Optimizer: same as v7 (AdamW lr=1e-3, weight_decay=0.1, cosine
  schedule, warmup=100, max_iters TBD).
- Save best F1 (not loss) on held-out.

## Evaluation metrics

Asymmetric per the spec's clinical-use bias:

| metric | target | rationale |
|---|---|---|
| Precision (intervention-start detection) | >= 80% | False alarm = whisper that didn't need to fire. Annoying but reversible. |
| Recall (intervention-start detection) | >= 70% | Missed intervention = clinical error reaches the CHW. Costly. |
| F1 | >= 0.75 | Headline number for graduation gate. |
| Per-protocol breakdown | no protocol < 50% recall | Catches "model learned 5 protocols and ignored 72". |

**Baseline to beat:**
- "always predict 0" gets 0% recall = 0 F1.
- "always predict 1" gets 100% recall, ~14% precision = 0.25 F1.
- HeuristicMidloopDecider (NOT trained on this data) on the same
  held-out -- TBD; expect very low F1 because it operates at step
  level not token level.

## Graduation criteria (numeric bars per spec)

Per midloop-spec.md section "Phases of execution":

1. **Phase 0 -- pilot validates pipeline** (DONE in PR #21).
2. **Phase 1 -- data pipeline produces dataset** (DONE in PRs
   #20-23, 1136 labels).
3. **Phase 2 -- midloop primitive ships** (DONE in PR #22).
4. **Phase 3 -- v0 NanoGPTMidloopDecider trains:**
   - F1 >= 0.75 on held-out (38 cases)
   - No per-protocol recall < 50%
   - Acceptance training time <= 30 min on MPS
5. **Phase 4 -- shadow mode in production:**
   - Wrap as `ShadowMidloopDecider(NoopMidloopDecider(),
     NanoGPTMidloopDecider(ckpt))`
   - Accumulate >= 200 (decision, task_outcome) pairs
   - Calibrate threshold on real labels
6. **Phase 5 -- graduation:**
   - Promote to PRIMARY only if shadow disagreements show >= 70%
     "agree-with-oracle" rate on the >= 200 labels
   - Action initially clamped to WHISPER

## What NOT to do in v0

Per Silt's rule + the v7 saga lessons:
- Do NOT bump architecture above 4L/4H/128d before measuring v0
  (H8 confirmed capacity is rarely the bottleneck on small data).
- Do NOT add contrastive losses in v0 (the H2/H2b/H2c trio
  showed those saturate fast on small bucket pools).
- Do NOT use the full 8053 tokens × 385 cases as a flat training
  set -- the protocol-id dependency is real (similar phrasings
  across cases of the same protocol). Stratified split + held-out
  keeps generalization claims honest.
- Do NOT optimize threshold on the test set. Use a separate
  validation slice (10% of train).

## Open questions for v0 implementation

1. **Tokenizer choice**: BPE trained on prompt+response (~few MB
   of clinical text) vs reuse v7's BPE tokenizer. Reuse is easier
   but v7's vocab was tuned for write-filter content (decisions
   vs noise from Claude transcripts), not clinical advice. Probably
   train a fresh BPE; keep the architecture identical.
2. **Block size**: Phase 1 cases have prompt + response averaging
   ~50 + 21 = 71 tokens. block_size=128 is safe; 256 might be
   waste. Confirm distribution on the actual training data.
3. **Hidden dim of tagging head**: a single linear layer over
   n_embd=128 -> 1 gives 129 params. Tiny. Could add an MLP
   (128 -> 64 -> 1) for slightly more capacity. Bench both.

## Open questions blocking Phase 4 (shadow deployment)

These don't block Phase 3 implementation but should be answered
before the trained model goes live:

1. **What is a "step" in the LLM-runtime context?** The midloop
   primitive operates on `StepObservation` which assumes a
   discrete step. For token-level intervention, the "step" is
   one token. The runtime needs to call `Memory.observe_step` per
   token, which is performance-sensitive. Streaming-callback
   architecture vs batch-after-N-tokens is a downstream design.
2. **Whisper injection mechanism**: when `intervene=True,
   action=WHISPER` fires, HOW does the Builder consume that?
   For Anthropic streaming API: cannot inject mid-stream cleanly.
   For local-model setups (lmstudio, llama.cpp): can prepend a
   guidance token. For Reforge: depends on its execution model.
3. **Latency budget**: a token-level call to a small classifier
   per generated token at ~1ms each adds up. For 200-token
   responses, +200ms overhead. Within budget for non-realtime
   clinical advice; not for chat UX. Sampling adaptive (every
   N tokens) is the documented mitigation.

## Timeline estimate

- v0 dataset reformat to Option B: 30 min (extend
  format_for_training.py with --labeling flag).
- v0 nanoGPT training script: 1 day (fork v7 train, swap head +
  loss, add per-position masking).
- v0 first training run + eval: 1 hour (~30 min train + 30 min
  eval, MPS).
- Iteration to F1 >= 0.75: probably 1-3 weeks of small tweaks
  (class weight sweep, threshold calibration, maybe a wider
  hidden dim if tagging-head capacity is the bottleneck).
- Phase 4 shadow integration: 1 day (NanoGPTMidloopDecider class
  mirrors NanoGPTWriteDecider; same file shape).
- Phase 4 -> Phase 5 (real-data calibration): months of organic
  use to accumulate the 200 pairs.

Phase 3 v0 is realistically ~1-3 weeks of focused work for
a publication-grade result. Phase 5 graduation is calendar-bound
on real-world traffic, not engineering effort.
