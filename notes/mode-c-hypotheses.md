# Mode C improvement hypotheses

Mode C ended the 2026-04-21 session at 13.3% correct on
LongMemEval N=30 (vs RAG-k3 at 70%). The mechanism works;
correctness is the gap. This doc enumerates testable hypotheses
for where the gap lives and what cheap experiments falsify or
confirm each. Ordered roughly by signal-per-effort.

The observations the hypotheses react to:

- 67% of N=30 verdicts are `neutral` -- gemma-4-E2B-it refuses
  or meta-reasons instead of answering.
- Decider fires late (t=700+). By then the model has committed
  to a refusal or to a wrong answer.
- 0/3 knowledge-update, 0/9 temporal, 0/8 multi-session.
  4/30 wins all on single-session-user "where does X live"
  lookups.
- Splice payloads retrieve via 3-way dual but often hit
  tangential chunks instead of the actionable one.

## Tier 1 -- cheap tests, high signal

### H1. Force early splice via a FIRST-fire policy

Observation: decider fires at t>=700 because heuristic patterns
don't trip on the thinking preamble. By then the model has
committed.

Hypothesis: forcing an unconditional retrieval at t=30-50 (well
before the preamble fully develops) gives the splice time to
shape the answer. The decider cadence still fires later for
complex reasoning, but a guaranteed first splice sets the
context early.

Test: add `force_first_fire_at_token=30` knob to
`StreamingDecider`. Rerun N=30. If correctness jumps to
25-40%, the late-firing hypothesis is the dominant issue.

### H2. Retrieve with question-only, not question + window

Observation: decider's window captures the model's
meta-reasoning text ("I am an AI, I need to verify..."). This
drifts the retrieval query away from the user's actual ask.

Hypothesis: searching with ONLY the question produces cleaner
hits. We already saw this in cotrimoxazole diagnostic -- the
chunk `hiv-who` hides on question+draft queries but surfaces
on question-only FTS.

Test: change `run_mode_c`'s retrieval call to `retrieve(mem,
question, ...)` (drop the window_text append). Rerun N=30.
If correctness rises, query drift was the retrieval issue.

### H3. Swap to gemma-4-E4B-it (same architecture, 2x params)

Observation: gemma-4-E2B-it 4-bit refuses aggressively. Phase
0 already proved gemma-4-E4B-it runs locally on the same
infra.

Hypothesis: E4B has enough capacity to follow the confident
preface instead of defaulting to refusal, and enough reasoning
to integrate spliced context into a correct answer.

Test: change DEFAULT_MODEL to gemma-4-E4B-it-MLX-4bit. Rerun
N=30. If correctness jumps noticeably (target: >=30%), the
Builder capacity hypothesis is the dominant issue.

Wall time goes up with E4B (2x params -> ~1.5-2x slower per
token). Budget allows it.

### H4. Tail-only oracle candidate (already landed)

Status: SHIPPED in the code-review fix commit. Documented here
so the hypothesis log is complete. The 33% -> 13% drop from v1
to v2 was because v1 was grading the thinking preamble
(head-truncation). Tail-truncation is the honest measurement.
Any future improvement claim must be measured with tail
truncation.

### H5. Reduce MAX_TOTAL_TOKENS from 800 to 200-300

Observation: gemma spends 300-400 tokens on thinking preamble
that contains speculation. The 800-token budget gives the
model room to commit to refusals before reaching the actual
answer.

Hypothesis: a tighter budget (200-300 tokens) forces the model
to skip or compress the preamble and go straight to the answer
body, which is where splice content matters.

Test: change MAX_TOTAL_TOKENS to 250. Rerun N=30. If
correctness rises, the thinking-preamble budget was crowding
out the answer.

Risk: on complex questions the model may not fit the full
answer in 250 tokens. Check wall + truncation rate.

## Tier 1.5 -- surfaced during H1/H2 analysis (2026-04-21)

### H12. Splice top-K chunks per firing (not top-1)

Observation (BBQ trace): vstash returned the correct chunk at
RANK 3 of the top-5 retrieval pool. The current code takes
``fresh[0]`` -- only top-1 per firing. With MAX_SPLICES=3,
three separate firings each splice one chunk. But the first
firing locks in a near-miss; by the time the third firing
surfaces the right chunk, the model has committed to the wrong
answer.

Hypothesis: splicing ``fresh[:3]`` (concatenated) per firing so
the model's KV cache sees up to 3 candidate chunks
simultaneously on the very first fire. Matches how RAG inlines
top-5 chunks upfront.

Test: change mode_c_demo to concatenate top-K per splice.
Rerun N=30 with K=3, MAX_SPLICES=2. Expected lift: wherever
the correct chunk was in rank 2-3 of the retrieval pool (most
multi-session and temporal questions).

Cost: splice payload grows ~3x per firing (~300 tok -> ~900
tok). Within existing 800-token budget if we also cut
MAX_SPLICES to 2.

### H13. Increase MAX_SPLICES to 5-6

Observation: some questions need 4+ chunks of context to
answer (multi-session aggregation, temporal ordering). Current
cap=3 truncates retrieval coverage.

Test: same as H12 but cap=6 with top-1 per firing. Cheaper per
firing but more firings.

Combined H12+H13: top-3 per firing × MAX_SPLICES=3 = up to 9
chunks covered. Probably overkill; pick the smaller move first.

### H14. Relative score threshold (top1 * factor)

Observation: absolute threshold 0.0161 was calibrated against
queries with full window_text appended. The Emily probe showed
the target chunk keeps score 0.0167 regardless of query form,
but rank-2 noise scores range from 0.0036 to 0.0164 depending
on whether the query has window or not. A fixed absolute cutoff
cannot distinguish signal from noise across regimes.

Hypothesis: cutoff = `top1_score * factor` adapts automatically.
Factor 0.5 means "keep chunks at least half as relevant as the
best hit" -- generous enough for multi-chunk aggregation,
strict enough to reject the 10x drop-off noise.

Test: run H3+H12 with `relative_threshold_factor=0.5`. Result
(2026-04-21): 40.0%, exact tie with H3+H12 absolute. Spliced
8.40 chunks/q vs 5.03 on absolute -- extra chunks accepted but
Builder ignored them, no signal gain. SHIPPED as a knob
(disabled by default).

### H15. Short retrieval window (last N chars)

Observation: full window_text (40 tokens, ~200-300 chars) has
enough surface area to inflate noise scores into the 0.016x
range where they pass threshold. The probe showed a 20-token
window keeps target-score stable while collapsing noise.

Hypothesis: truncating the window to the last 80 chars gives
the decider enough context for intent detection while keeping
the retrieval query compact.

Test: H3+H12 with `retrieval_window_tokens=80`. Result
(2026-04-21): 36.7%, -3.3pp regression. Short window cut off
reasoning state the decider needed, lost 1 supports + 1
contradicts, went to 15 neutral. REJECTED.

### H16. Question-only retrieval + lowered threshold

Observation: H2 (question-only) was measured at 13.3% with the
absolute threshold locked at 0.0161, which collapsed the noise
floor but also collapsed the multi-chunk policy to top-1
(because only rank-1 passed). Re-measure with threshold 0.008
so H12 can still pick top-3.

Hypothesis: question-only + lowered threshold should give the
cleanest retrieval (no window drift) AND preserve multi-chunk
KV enrichment.

Test: H3+H12+H16 with `--question-only-retrieval
--score-threshold-override 0.008`. Result (2026-04-21):
**20.0%, -20pp major regression**. Root cause: with a static
query across 3 firings, spliced-source dedup pushes firings 2
and 3 to ranks 4-9 of an unchanging pool -- all garbage. Low
threshold accepts them all. KV cache fills with three rounds
of off-topic chunks; Builder defaults to "not in memory"
19/30 times. REJECTED (and informative).

## Summary of Tier 1.5 follow-up (2026-04-21 PM)

All three retrieval-quality tweaks to H3+H12 either regressed
or tied. The Emily-probe finding ("clean queries have wider
signal gaps") is true in vitro but does not dominate in the
broader eval. The 30pp gap to RAG-k3 lives in the Builder,
not the retrieval pool -- H6 (prompt engineering) and H11
(non-refusing Builder) are the next plausible levers.

## Tier 2 -- moderate effort, moderate signal

### H6. Stronger anti-refusal preface

Observation: PROMPT_PREFACE currently says "Do not hedge, do
not refuse. If the answer is not in memory, say 'not in memory'
rather than guessing." gemma still takes the "not in memory"
escape hatch 20/30 times.

Hypothesis: removing the "not in memory" fallback and requiring
commitment would force the model to use the splices. Or
framing the question as "This is a training exercise, answer
as if the memory content is ground truth."

Test: variant prefaces. Rerun N=30 for each. If correctness
rises with the stronger preface, prompt engineering has more
room than we thought. If not, gemma's refusal is structural.

### H7. LLMClaimDetector in place of HeuristicClaimDetector

Observation: HeuristicClaimDetector fires on
`prescriptive:should_must` and `reasoning:causal` patterns --
which happen deep in the model's meta-reasoning, not when
the model is about to commit to a fact.

Hypothesis: a Gemini-backed LLMClaimDetector would fire on
semantically meaningful decision points (the model about to
commit to a specific fact) instead of regex patterns. Cheaper
per call than the 235b Judge because Gemini Flash at $0.075/M
input is < $0.001 per firing.

Test: swap in LLMClaimDetector in run_mode_c. Rerun N=10 (to
keep API cost bounded). If correctness rises AND firing
positions shift earlier, the decider is the bottleneck.

### H8. Splice format -- plain text vs envelope vs fake assistant turn

Observation: current splice is
`\n\n[Source: X]\n{text}\n\n`. The envelope is verbose and the
model may treat it as metadata to skip.

Hypothesis: a "fake assistant message" format
(`<start_of_turn>model\nBased on memory: {text}<end_of_turn>\n<start_of_turn>model\n`)
might make the model treat the injected content as HIS OWN
prior statement, which he then has to continue coherently.

Test: probe on cotrimoxazole question with 3 splice formats.
Pick the winner and rerun N=30.

## Tier 3 -- bigger moves

### H9. Hybrid shape: retrieve BEFORE generation + splice DURING

Observation: RAG pre-loads context; Mode C loads mid-stream.
What if we pre-load the top-1 chunk AND allow mid-stream
splices on top?

Hypothesis: the pre-load primes the model's thinking with the
right context (kills refusal); mid-stream splices refine when
the answer turns out to need more.

Test: new condition `mode_c_hybrid`. Prepend top chunk to the
user message before chat-template. Keep mid-stream mechanism.
Rerun N=30.

### H10. Train a tiny decider via self-supervision

Observation: the HeuristicClaimDetector is a regex; Mode C's
decider either fires too early (noise) or too late (committed).

Hypothesis: each Mode A audit row (we have ~100 accumulated)
is a labeled training example: (question, Builder draft,
retrieval pool, Judge verdict). A small classifier can learn
"should we retrieve AT this token position given the prefix".

Test: train a distilbert-scale classifier from the audit
rows. Plug in as LLMClaimDetector replacement. Rerun N=30.
Needs >>100 rows to actually work; we'd accumulate more audit
rows from the Tier 1/2 experiments first.

### H11. Swap Builder to a non-refusing small model

Observation: gemma-it is safety-tuned for dosing/medical/
personal questions. That's ruining Mode C on LongMemEval
which is personal-memory.

Hypothesis: a less safety-tuned local model (qwen-2.5-1.5B,
mistral-7b-base, llama-3.2-3B-base) would commit to answers
based on context instead of refusing.

Test: qwen-2.5-1.5b-instruct or qwen-2.5-3b-instruct via
mlx-lm. Rerun N=30. Wall time bumps; correctness comparison
tells us how much of the gap is model-choice vs architecture.

## Order of experiments

Signal-per-effort order for the next session:

1. **H1 force-first-splice** (~30 min, 0 API). If this alone
   jumps correctness past 30%, late firing was the main
   failure mode.
2. **H2 question-only retrieval** (~30 min, 0 API). Reuses
   the existing dual-3 retrieval; just drop the window text
   from the query.
3. **H3 swap to gemma-4-E4B-it** (~30 min, 0 API). Phase 0
   already proved it runs. Worth one N=30 data point to
   separate "architecture" from "model capacity".
4. **H5 tighter MAX_TOTAL_TOKENS** (~20 min, 0 API). Cheap
   knob test.
5. **H7 LLMClaimDetector** (~1h, ~\\$0.50 API on N=10). Signal
   on whether the decider itself is the limit.

If 1-4 don't move the needle, the bottleneck is fundamentally
the Builder (H11) or the retrieval (larger corpus + better
query expansion). If 1-4 do move it, H7/H10 become the
follow-ups.

## What we are NOT going to test (yet)

- Multi-step splicing (2+ retrievals in one generation with
  dedup): already happening, not the issue.
- KV-cache position encoding variants: phase0c/d proved the
  cache works; no need to re-litigate.
- Full-finetuning a decider / Builder: requires thousands of
  rows, not ready.
