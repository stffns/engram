# Mode C as the production shape, with a continuous streaming decider

Short memo distilling what clarified at end of 2026-04-21 while
running N=50 honest evals. We spent ~3 hours discovering that Mode
A is not the production shape and that Mode C is not simply "Mode
A but local". This note fixes the architecture so the next session
does not re-derive it.

## What the N=50 evals revealed (not what we planned to learn)

1. **Mode A v4 is dominated by naive RAG on answer correctness.**
   Three consecutive N=50 runs on LongMemEval_s, same seed:
   RAG baseline 71% / 70% / 78%; Mode A 71% / 66% / 64%. The
   Judge 235b does not earn its ~4x token cost in top-line
   correctness. It does buy +81% grounded evidence trail, which
   matters for audit but not for the oracle's correct/wrong
   judgment.

2. **Prompt engineering is a double-edged lever.** Adding 6
   targeted rules ("rag_specific": aggregation, recency,
   abstention, citation, verbatim, concise) **regressed by 20pp**
   vs the 3-rule RAG baseline. The minimal prompt survives on
   llama3.1-8b better than a precise prompt. Surgical, not
   additive.

3. **Temperature 0.3 was introducing cross-run variance.** Same
   seed, same question, flipped verdict 16-18% of the time
   between runs. Temperature 0.0 is necessary before any
   A/B claim survives scrutiny.

## The production shape that actually matters

Mode A (what we evaluated) is **retrieve-between-two-generations**.
Mode C is **retrieve-during-one-generation** via KV-splice. That
is the architectural difference that makes the cost collapse
viable.

### Biological analogy (useful intuition, NOT a thesis claim)

Jay framed the experiment informally as "asi como funciona la
memoria humana" -- the interleaving of retrieval and generation
feels analogous to how people fetch memories while speaking.
Treat this as an **intuition pump** for why continuous
interception is an interesting shape to test, not as a
cognitive-science claim about the architecture. We are not
modeling human memory; we are using the metaphor to motivate
a mid-stream retrieval loop that is worth measuring against
RAG and Mode A on real tasks.

Where the analogy is handy:
- Frequency of retrieval correlates with complexity of
  utterance -- "hi" consults nothing, "remember when we talked
  about X in 2019..." consults multiple times.
- Retrievals interleave with generation rather than preceding
  it.

Where it is only metaphor, not mechanism:
- K/V splice is an LLM-cache operation, not reconstruction-from-
  engram. Do not build on the human-memory framing as if it
  explained the attention math.
- No claim here about RAG or Mode A being "less like memory" --
  they are engineering shapes with their own tradeoffs. The
  analogy describes why Mode C is worth testing, not why the
  others are wrong.

### Continuous streaming decider -- the refined concept

The gate does NOT fire once at N=10 tokens and then wait. It is
a continuous monitor that runs across the generation stream at a
configurable cadence:

- Decide every N tokens where N in {10, 20, 100, variable} --
  the cadence is a hyperparameter, not a fixed cut.
- Windows may OVERLAP (sliding, e.g. evaluate the last 20 tokens
  every 10 tokens generated) or be DISJOINT (evaluate fresh
  chunks of 20 tokens as they arrive). Both are valid.
- Each firing may or may not trigger a vstash call. Simple
  responses may not hit the decider's threshold at all.
  Complex multi-claim responses trigger several retrievals +
  splices across one generation.
- Each retrieval is a potential K/V splice into the ongoing
  generation's cache. The Builder keeps running after each
  splice with the enriched context.

Concretely:

```
Builder streams tokens
  decider(window_1) -> maybe splice
  Builder continues
  decider(window_2) -> maybe splice
  Builder continues
  decider(window_3) -> no splice, just pass
  ...
  Builder emits EOS
```

### Implications for the claim_detector

Training data is **per-window**, not per-question:
- Features = `(question, partial_output_prefix, window_content)`
- Label = `needs_retrieval` (bool) plus `retrieval_query_hint`
  (what to search)

This is a classic streaming classifier problem, not a one-shot
gate. The audit rows we are already accumulating are close but
not directly usable -- they carry whole-question outcomes, not
per-window decisions. We need a per-window labeler (could be
derived from the claims[] array the Judge already emits, by
mapping sub-claims to the spans in the output where they first
appear).

### Cost profile (revised)

Average question (mix of simple and complex):
- 0-3 decider firings per question (simple: 0; complex: 2-3)
- Each firing: ~10ms for the decider + ~100ms vstash +
  ~100ms K/V splice
- One Builder generation end-to-end (no second LLM call)
- Total: ~500-800 LLM tokens, ~1-2s wall, $0 if local mlx

vs Mode A 8759 tok / ~5s / $0.008. An order of magnitude
cheaper once the decider works.

## RAG ceiling calibration (2026-04-21 grid)

Temperature + top_k sweep on RAG baseline, N=50 seed 42
longmemeval_s, same corpus the Mode A runs used
(`grids/rag_baseline_sweep.yml`):

**Temperature (top_k=5 fixed):**
- t=0.0: 72.0% correct, 2288 tok/q
- t=0.1: 74.0% correct, 2290 tok/q
- t=0.3: 74.0% correct, 2294 tok/q

**top_k (t=0.0 fixed):**
- k=1: 54.0%, 531 tok
- k=3: **70.0%, 1394 tok** (cost/correctness knee)
- k=5: 72.0%, 2288 tok
- k=10: 72.0%, 4436 tok

**Calibrated baseline: RAG-k3 at temp 0.0-0.3 = ~70-74%
correct at ~1400 tok/q.**

Important correction to the earlier "temperature noise"
hypothesis: the Run 1 vs Run 2 vs Run 3 flips (71% / 70% / 78%
for RAG) were NOT dominated by Cerebras sampling temperature.
The t=0.0 run on the same seed landed 72% -- in the middle of
that range. The variance was mostly **oracle (Gemini Flash)
noise** across runs. A real stability study would need
multiple oracle samples per answer; single-draw oracle scoring
is noisier than we were assuming.

Comparison anchor for Mode C:

| shape | correct | tok/q | API $/q |
|---|---|---|---|
| control (no context) | 8% | 290 | 0 |
| RAG-k3 (new calibrated baseline) | 70% | 1394 | ~$0.0006 |
| RAG-k5 (earlier baseline) | 72-74% | 2288 | ~$0.001 |
| Mode A v4 | 64-71% (avg ~67%) | 9048 | ~$0.008 |

**Mode A loses ~3pp to RAG-k3 at ~6.5x the cost.** Mode A is
not a production candidate on this corpus, at this scale,
with this Builder (llama3.1-8b). It remains useful as the
training-time labeler that will eventually distill a
claim_detector.

Mode C's target: match RAG-k3 on correctness while moving the
per-query retrieval call from "always" to "content-conditional"
via the continuous decider. If the decider correctly abstains
on ~30% of questions (the ones Builder answers correctly
without memory), Mode C expected cost ~70% of RAG-k3 = ~1000
tok. Running mlx-locally drops API cost to zero; Cerebras cost
remains below RAG.

## Measurement artifact lesson (2026-04-21, Mode C benchmark)

Running Mode C on LongMemEval N=30 exposed a nasty oracle-scoring
artifact. First pass reported 33% correct; detailed code review
caught three fixes; re-run reported 13% correct. Same pipeline,
same model, same questions. The 33% was **wrong**.

Root cause. gemma-4-E2B-it wraps its output as:

```
<|channel>thought
...150-400 tokens of internal reasoning / speculation...
<channel|>
...the actual user-facing answer...
<turn|>
```

The mode_a_eval oracle prompt truncates the candidate to the
FIRST 2000 chars. Mode C outputs clocked in at 3000-4000 chars.
That head-truncation delivered the THINKING PREAMBLE to the
oracle while discarding the real answer. The preamble often
speculates correctly on the ground truth (\"probably June 3rd
based on context\"), which the oracle then scored as `supports`.
The user-facing answer frequently disagreed (\"10th and 17th of
June\") but the oracle never saw it.

Fixes landed:

1. Strip the thinking preamble via \`rsplit(\"<channel|>\", 1)[-1]\`
   before passing to oracle.
2. Tail-truncate instead of head-truncate: \`candidate[-2000:]\`.
3. Don\'t inject splice_ids into answer_tokens -- they are
   prefilled KV-cache content, not model output. Including them
   made the oracle grade the retrieved chunk instead of the
   model\'s actual answer.
4. Use cerebras_midloop.retrieve(retrieval_mode=\"dual\") so Mode
   C hits the same retrieval substrate as the Mode A baselines
   (head-to-head fairness).

Post-fix numbers: Mode C at 13-20% correct on N=30. The honest
quality gap vs RAG-k3 (70%) is WIDER than the original measurement
implied.

Why this matters beyond Mode C: the pre-run code-review rule
(CLAUDE.md global) exists precisely to catch this class of silent
bias before it lands as a claim in RESULTS.md. The bots missed
the oracle-truncation issue; the code-reviewer subagent caught
it. Invoking code-reviewer before shipping a measured result is
not a nicety -- it is the thing that stops a fake win from
entering the project\'s record.

## What stays open

- **Long-splice probe**: Phase 0b proved 14-token splice produces
  coherent continuation. Needs 500-token probe with real vstash
  excerpts before the architecture is validated at production
  scale.
- **Chat-template interaction**: POC used raw continuation, not
  Gemma/Llama role markers. Splicing into an assistant turn
  mid-stream may or may not preserve the template boundaries.
- **claim_detector as streaming classifier**: current scaffolding
  (`merken/policies/claim_detector.py`) is one-shot. Needs a
  windowed wrapper.
- **vstash call budget policy**: how many splices per generation
  is too many? Is there a soft-cap? A cooldown?

## What is NOT the production shape

- Mode A with Judge 235b rewriting: validated as training-time
  teacher only. Kept in the stack for distillation labels.
- One-shot gate at N=10: superseded by continuous streaming
  decider per Jay 2026-04-21.
- Retrieve-everything RAG: the baseline we have to beat, not
  the shipping shape. Mode C's promise is "cheaper than RAG
  because the decider often says no".
