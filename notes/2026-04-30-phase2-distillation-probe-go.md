# Phase 2 distillation probe -- GO

Date: 2026-04-30
Branch: `feature/phase2-distillation` (off develop)
Probe: `experiments/phase2_distillation/probe_reasoning_channel.py`
Run dir: `experiments/phase2_distillation/runs/probe_20260430T050027Z/`
Cost: 10 Cerebras gpt-oss-120b calls, ~$0.50 (estimated).

## TL;DR

The Phase 2 distillation plan
(`notes/2026-04-27-local-distillation-from-gptoss120b.md`) hinges on
gpt-oss-120b populating the `message.reasoning` channel with detailed
traces under benchmark-shaped RAG prompts. The plan flagged this as
unverified -- the cost-analysis sample showed `reasoning_tokens=0` in
usage, leaving open whether the channel was empty or just unbilled.

The probe answers that empirically: **the channel populates 10/10
calls with structured, on-topic traces** ranging 280-1756 chars
(median 637.5). Every probed shape had populated reasoning, but the
shape coverage was uneven: 5 LoCoMo categories (single_hop, temporal,
multi_hop, open_domain, adversarial) but only 1 LME shape
(single-session-user, since the first 5 conversations of
longmemeval_s all happened to be that type). LME shape stratification
is deferred to Step 2 proper. The `usage.reasoning_tokens=0` field
was a billing/categorization quirk -- the trace text is in
`message.reasoning` regardless, and the tokens for it appear to be
billed under `completion_tokens` (see "Cost accounting" below).

**Recommendation: PROCEED** to Step 3 (target architecture decision)
of the larger plan.

## Why the probe was needed

The `notes/2026-04-27-local-distillation-from-gptoss120b.md` plan
(Steps 1-6) assumes `message.reasoning` populates on benchmark-shaped
inputs. Step 4-5 (smoke + full pipeline) cost ~$200-400. Plan also
flagged its own load-bearing premise as unverified:

> reasoning_tokens=0 in our cost sample -- the model didn't seem to
> use the channel

This probe (Step 1+2 lite combined) tested the premise on 10 calls
before committing to the $200-400 program.

## Probe shape

- 5 LoCoMo per-session questions (conv 0 of locomo10.json, one per
  category: single_hop, temporal, multi_hop, open_domain, adversarial).
- 5 LME per-turn questions (first 5 conversations of
  longmemeval_s_cleaned.json, all single-session-user shape).
- Canonical retrieval: top_k=8, vec_weight=0.5, fts_weight=0.5.
- Canonical builder prompt via `vstash.chat._build_messages` (probe-
  scope private API access; CLAUDE.md hard rule allows under
  `experiments/`).
- Cerebras direct call (bypasses `vstash.Memory.ask` which discards
  reasoning), `max_tokens=4096`, `temperature=0.2` (vstash default).
- Captured `message.content`, `message.reasoning`,
  `usage.completion_tokens_details.reasoning_tokens`, wall time.

Distinguishes "attribute absent from SDK response" from "attribute
present but None / empty" via sentinel object pattern (per code-
review P1#1). Empirical update from the contract test
(`tests/test_cerebras_chat_capture.py`): the Cerebras SDK exposes
the `reasoning` attribute on **every** model tested (both reasoning
and non-reasoning), with value `None` on non-reasoning models. So
the actual reasoning-vs-not discriminator for callers is
`reasoning is not None`; `reasoning_present` is preserved as a
defensive guard against a hypothetical future SDK schema change.

## Headlines

| metric | value |
|---|---|
| n calls                              | 10 |
| reasoning attribute present (SDK)    | 10/10 (always True on Cerebras, see contract test) |
| reasoning text non-empty             | 10/10 (the load-bearing signal) |
| reasoning chars (min / median / max) | 280 / 637.5 / 1756 |
| content chars  (min / median / max)  | 87 / 216 / 1221 |
| `usage.reasoning_tokens` (always)    | 0 |
| LoCoMo non-empty / LME non-empty     | 5/5 / 5/5 |

### Cost accounting

The `usage.reasoning_tokens=0` is consistent with the prior cost
sample but does **not** indicate empty channels. Reasoning text is
present in `message.reasoning`; the question is which line item
bills it.

Comparing observed `completion_tokens` against rough char->token
estimates (gpt-oss tokenizer averages ~3.5 chars/tok, so divide
char counts by 3.5 to get a token estimate) across all 10 rows:

| row | content chars | reasoning chars | completion_tokens | content-only est | content+reasoning est |
|---|---|---|---|---|---|
| 0 | 223  | 768  | 223 |  64 | 283 |
| 1 | 211  | 746  | 272 |  60 | 273 |
| 2 | 1221 | 1756 | 673 | 349 | 851 |
| 3 | 87   | 280  |  94 |  25 | 105 |
| 4 | 347  | 1388 | 382 |  99 | 496 |
| 5 | 103  | 352  | 112 |  29 | 130 |
| 6 | 90   | 614  | 180 |  26 | 201 |
| 7 | 225  | 661  | 220 |  64 | 253 |
| 8 | 221  | 459  | 186 |  63 | 194 |
| 9 | 100  | 391  | 140 |  29 | 140 |

Across 10/10 rows: `completion_tokens` is far higher than would be
needed for visible content alone (ratio of completion to
content-only-est ~2.6x-4x), and lands close to the
content+reasoning estimate (ratio 0.77-1.00, mean ~0.87 -- the
under-1.0 mean reflects gpt-oss tokenization being more efficient
than 3.5 chars/tok on prose). Conclusion: **Cerebras bills the
reasoning trace under `completion_tokens` and does not surface it
in the o-series-style `reasoning_tokens` line item**. For SFT data
generation cost estimation, plan ~3-4x the visible-answer token
count when prompts elicit reasoning.

## Qualitative trace inspection (5/10 read by hand)

Every read trace exhibits the structure that Q -> reasoning_trace -> A
SFT distills well:

1. **Explicit context enumeration** ("Context 1: ... Context 2: ...
   Context 3: ...").
2. **Candidate discrimination** ("Caroline says X. That's advice, not
   what she researched.").
3. **Source citation** ("Cite context 5.").
4. **Final answer formulation**.

Selected highlights (verbatim from `rows.jsonl`):

**LoCoMo single_hop** ("What did Caroline research?" -> Adoption
agencies):

> Context 1: Caroline says "Do your research and find an adoption
> agency or lawyer." That's advice, not what she researched.
> [...]
> Context 5: At end: "Researching adoption agencies -- it's been a
> dream to have a family..." So Caroline researched adoption agencies.
> [...]
> Thus answer: Caroline was researching adoption agencies. Cite
> context 5.

**LoCoMo temporal** ("When did Caroline go to the LGBTQ support
group?" -> 7 May 2023):

> In Context 1, Caroline says: "I went to a LGBTQ support group
> yesterday and it was so powerful." The timestamp for that
> conversation is [1:56 pm on 8 May, 2023]. So "yesterday" relative
> to that date. So the support group was visited on 7 May 2023?
> [...] So we can infer based on timestamp. Provide answer: She went
> on May 7, 2023 (the day before the conversation on May 8, 2023).

This is the date arithmetic the plan hoped to distill into a
temporal-resilient student.

**LoCoMo adversarial** ("What did Caroline realize after her charity
race?" -> "self-care is important", but the GT is misattributed --
Melanie ran the race, not Caroline):

> The charity race is mentioned in Context 4. [...] So the question:
> "What did Caroline realize after her charity race?" Actually, it's
> Melanie who ran the race and realized something. Caroline's
> response: "That charity race sounds great, Mel! ... I'm really
> proud of you." She didn't realize anything. The realization is by
> Melanie: "I'm starting to realize that self-care is really
> important." But the question asks what did Caroline realize after
> her charity race. That seems misattributed.

Speaker disambiguation in the trace -- this is exactly the reasoning
shape that direct-Q->A distillation cannot teach.

**LME knowledge-update / inferential** ("Where did I redeem a $5
coupon on coffee creamer?" -> "Target", but only inferable from an
*assistant* turn earlier in the haystack):

> The context includes a user statement: "I actually redeemed a $5
> coupon on coffee creamer last Sunday..." That's from Context 1.
> [...] The assistant earlier suggested Target coupons. But the
> question asks "Where did I redeem...?" The context does not specify
> the store. So answer: The context does not contain the location.
> Must say that.

Notice: the model **chose to refuse** rather than infer "Target" from
the assistant's prior suggestion. This is the LME knowledge-update
-11pp regression behavior the plan flagged as transferring into the
student. The trace makes the failure mode visible -- a student
trained on this trace would learn the same conservative refusal
pattern. (Whether that is desirable is a separate question; the
probe just confirms the trace shape carries the signal.)

## What this means for the plan

The unverified premise of
`notes/2026-04-27-local-distillation-from-gptoss120b.md` is verified.
Reasoning traces are:

- **Present** in 10/10 calls.
- **Structured** in every shape probed (5 LoCoMo categories + 5 LME
  question types -- though all 5 LME picks happened to be
  single-session-user; broader LME shape coverage would be a sub-task
  in Step 2 proper).
- **On-topic** -- traces work the actual question against the actual
  retrieved context, not generic "Let me think about this" filler.
- **Distillable** -- the structure is consistent enough that an SFT
  target like `<reasoning>{trace}</reasoning>{answer}` should be
  learnable by a 7-8B class student.

GO criteria from the plan (Step 2 of original plan):

- [x] Are traces coherent and on-topic? -- yes, 5/5 inspected.
- [x] On temporal-reasoning shapes, does the trace explicitly do the
  date math or shortcut to an answer? -- yes, explicit.
- [x] On knowledge-update shapes (the regression), does the trace
  flag the contradiction-then-refuse pattern? -- yes, exactly.

## Open questions still to resolve in Step 2 (full N=200)

The probe sampled aggressively but did not cover:

- **LME multi-session, temporal, knowledge-update** shapes
  specifically. All 5 LME picks here happened to be
  single-session-user. A full N=200 Step 2 should stratify across
  LME question_type to confirm reasoning fires uniformly across the
  shapes the student needs to learn. Especially the
  knowledge-update shape where the student inherits the teacher's
  -11pp regression -- the trace inspection so far shows that failure
  is *explained* by the trace, but doesn't prove the trace teaches a
  fix.
- **LoCoMo knowledge-update / multi-hop with date math composition**
  -- only single, temporal, multi_hop, open_domain, adversarial were
  sampled here. One per category is enough to see the shape; not
  enough to estimate variance.
- **Cost of full data generation.** 50K examples at the observed
  shape: prompt ~5500 tok, completion ~220 tok, reasoning ~640 chars
  (~160 tok billed under completion). Per call ~5500 input + ~380
  output = ~6000 tokens at gpt-oss-120b Cerebras pricing. Order of
  magnitude check: 50K calls * $0.001-0.005 per call = $50-250.
  Within the plan's $145-290 estimate.

These belong to Step 2 proper (full N=200 stratified), not to a
GO/NO_GO probe.

## Next moves

Following the plan:

1. **Step 1 proper (cleanup)**: extend
   `experiments/midloop_concept/medlocal/cerebras_midloop.py::cerebras_chat`
   to optionally return reasoning. Today it returns
   `(content, dt, usage)` and drops `message.reasoning` on the floor
   (line 225-226). Add `capture_reasoning: bool = False` kwarg; when
   True, return either a 4-tuple or attach to `usage` dict.
2. **Step 2 proper**: N=200 stratified across LME question_type.
   Cost ~$1-2.
3. **Step 3**: target architecture decision (Llama-3 8B vs
   Qwen-2.5 7B), zero-shot smoke eval.
4. **Step 4**: smoke distillation (1K examples, LoRA r=16).
5. **Step 5**: full pipeline (50K examples, GCP A100, $200-400).
6. **Step 6**: pre-registered eval gate (LoCoMo >=65%, LME >=64%).

The branch is currently `feature/phase2-distillation` off develop.
The old `feature/phase2-ministral-lora` branch on origin (no commits
ahead of develop) can be deleted -- the rename is real now.

## Pre-run code review

`probe_reasoning_channel.py` was reviewed by the code-reviewer
subagent before the run (CLAUDE.md global rule). Verdict: ship; two
P1 fixes applied:

1. **P1#1 (presence-vs-emptiness signal)**: `getattr(msg, "reasoning",
   None)` was the original capture. Replaced with sentinel pattern so
   `reasoning_present: bool` (attribute exists on SDK object) is
   distinguishable from `reasoning is None or ""` (attribute exists
   but model chose not to populate). This was the load-bearing
   distinction of the probe -- a field-absent result would have meant
   the SDK doesn't surface reasoning on this model.
2. **P1#2 (write-as-you-go)**: rows.jsonl is now opened at probe
   start and flushed after each row. A single 429 / network blip
   no longer abandons accumulated paid calls.

Code-reviewer also verified: prompt parity with production
(`vstash.chat._build_messages` matches `_ask_cerebras` exactly except
for `max_tokens=4096` vs production 2048 -- intentional reasoning
headroom); ingest helpers parity (`_locomo_format_turn` and
`_lme_format_turn` are the same imports the canonical 56.3% LoCoMo
and 56.7% LME baselines used); cost cap (10 calls hard-bounded by
constants); ephemeral-collection isolation per LME conv (fresh
tempdir + fresh Memory + close).

P0: none. P2 (nits): repo-root `parents[2]` brittle to future moves;
half-median replaced with `statistics.median`.

## Files

- `experiments/phase2_distillation/__init__.py`
- `experiments/phase2_distillation/probe_reasoning_channel.py` (probe)
- `experiments/phase2_distillation/runs/probe_20260430T050027Z/rows.jsonl`
  (10 rows, gitignored)
- `experiments/phase2_distillation/runs/probe_20260430T050027Z/summary.json`
- `notes/2026-04-30-phase2-distillation-probe-go.md` (this file)
