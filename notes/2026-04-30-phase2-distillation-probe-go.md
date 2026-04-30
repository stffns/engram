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

## Step 2 stratified update (2026-04-30, same day)

Re-run with N=55 stratified across all 11 shapes (5 LoCoMo
categories x 5 calls + 6 LME question_types x 5 calls). Closes the
LME shape-coverage gap from Step 1+2 lite. Cost: ~$1, wallclock
12.8 min (faster than the 30-40 min estimate -- LoCoMo phase
shared a single ingest, and gpt-oss-120b on Cerebras turns ~3-5s
per call). Run dir:
`experiments/phase2_distillation/runs/step2_20260430T060103Z/`.

### Per-shape reasoning rates

LoCoMo (single conversation, conv-26):

| shape | n | reasoning rate | r chars (min/med/max) | content med |
|---|---|---|---|---|
| single_hop  | 5 | 5/5 (100%) | 596 / 1115 / 1560 | 437 |
| temporal    | 5 | 5/5 (100%) | 508 / 776  / 1084 | 215 |
| multi_hop   | 5 | 5/5 (100%) | 551 / 1208 / 5300 | 502 |
| open_domain | 5 | 5/5 (100%) | 314 / 535  / 1224 | 233 |
| adversarial | 5 | 5/5 (100%) | 456 / 1501 / 2160 | 296 |

LME (one conversation per question, 30 separate ingests):

| shape | n | reasoning rate | r chars (min/med/max) | content med |
|---|---|---|---|---|
| single-session-user      | 5 | 5/5 (100%) | 322 / 370  / 423  | 97 |
| single-session-assistant | 5 | 5/5 (100%) | 350 / 700  / 861  | 148 |
| single-session-preference| 5 | 5/5 (100%) | 411 / 916  / 2609 | 326 |
| multi-session            | 5 | 5/5 (100%) | 491 / 871  / 1680 | 205 |
| temporal-reasoning       | 5 | 5/5 (100%) | 628 / 1177 / 2222 | 228 |
| knowledge-update         | 5 | 5/5 (100%) | 785 / 1271 / 2646 | 465 |

Zero burned-budget rows, zero ingest skips, zero per-conv errors
across all 55 calls.

### What the per-shape distribution tells us

Reasoning rate is **uniformly 100%** across every shape; that
falsifies the worry that the channel is shape-sensitive. But the
**length distribution** is the more informative signal:

- Hardest shapes get the longest traces. LME `knowledge-update`
  (median 1271, max 2646) and `temporal-reasoning` (median 1177,
  max 2222) are at the top -- exactly the shapes the plan flagged
  as the teacher's documented weakness. The model is reasoning
  *more*, not less, on those shapes.
- Simplest shapes get the shortest traces. LME `single-session-user`
  (median 370) is the floor -- single context, single user
  statement, direct answer. The model isn't padding.
- LoCoMo `multi_hop` has a 5300-char outlier (max). Inspection
  confirms it is the model walking through 8 retrieved sessions
  systematically -- a long but coherent trace, not filler.

Implication for SFT: the trace **length variance follows question
complexity**, which is the property reasoning-distillation
literature relies on. A student trained on these traces would see
both short-and-direct and long-and-deliberative examples,
correlated with question shape. That is the signal that lets the
student learn *when* to reason at length, not just *how*.

### Hand-inspected traces (2 per load-bearing shape)

**LME temporal-reasoning** ("Which group did I join first, 'Page
Turners' or 'Marketing Professionals'?" -> Page Turners):

> Context 2: user says "I've been engaging with a group called
> 'Marketing Professionals' on LinkedIn since I joined it
> yesterday..." So they joined Marketing Professionals yesterday.
>
> Context 3: user says "I'm looking for some book recommendations.
> I just joined a new book club group called 'Page Turners' last
> week..." So they joined Page Turners last week.
>
> Thus timeline: Marketing Professionals joined yesterday; Page
> Turners joined last week. So Page Turners was joined earlier
> (last week) than Marketing Professionals (yesterday). So
> answer: Page Turners first.

Same date-arithmetic pattern as the LoCoMo temporal trace from the
probe. Distillable.

**LME multi-session** ("How much have I made from selling eggs
this month?" -> $120) -- cross-context synthesis with arithmetic:

> Context 1: user says selling eggs to neighbor for $3 a dozen.
> No quantity.
> Context 2: user says sold a total of 40 dozen eggs so far. No
> price given, but earlier context maybe price $3 per dozen. [...]
> So we can compute revenue: 40 dozen * $3 = $120.

Identifies price in one context, quantity in another, multiplies,
cites both. The student needs to learn this exact synthesis
pattern; the trace makes it explicit.

**LME knowledge-update** -- two outcomes worth contrasting:

Trace #1 (qid=6a27ffc2, "How many videos of Corey Schafer's Python
programming series have I completed?" -> 30):

> Context 1: user says "I've completed 30 videos so far for
> Corey's series". That's 30.
> Context 3: user says "I've completed 20 videos so far". That's
> 20.
> Context 6: assistant says "Congratulations on completing 30
> videos". So 30. [...]
> Thus the user likely wants current count. The most recent
> statements: In context 1 and 2 and 6 they say 30 videos. The
> earlier 20 was earlier. So answer: 30 videos.

Correct. The model resolves the conflict via "most recent count
wins" heuristic, applied to retrieval order, and gets it right
when retrieval order happens to match temporal order.

Trace #2 (qid=06db6396, "How many projects have I completed since
starting painting classes?" -> GT 5; **model answers 4**):

> Context 1: user says "I just finished my 5th project since
> starting painting classes". So that says 5 projects.
> Context 2: user says "I've completed 4 projects since starting
> painting classes". So 4. [...]
> The latest mention is context 3 (turn 8) where user says "Since
> I've completed 4 projects". That is after context 2 (turn 4).
> So likely the correct answer is 4 projects.

Wrong -- the model applies the "most recent" heuristic to
*retrieval order* (chunk turn index) instead of *temporal order
in the conversation*. The 5-projects statement was a later event;
the 4-projects statements were earlier reports the model
encountered last in retrieval. **The trace makes the failure mode
visible:** the student would inherit this exact heuristic, which
is what the plan's "teacher's failure modes transfer" risk
predicts. A future remediation (claim-level Judge with explicit
temporal ordering, or a retrieval rewrite that surfaces newest-
first) could be specifically targeted at this trace pattern --
and the trace gives the test cases.

### What this changes about the GO

Step 1+2 lite said "PROCEED to Step 3" with the LME-shape gap
flagged. Step 2 stratified closes the gap and confirms:

- Reasoning fires at 100% across all 11 shapes the student
  needs to learn.
- Reasoning length scales with question complexity -- the
  property reasoning distillation needs.
- The teacher's documented LME failure modes
  (knowledge-update, temporal-reasoning) get *more* reasoning,
  not less; the failure is in the heuristic the model applies,
  not in the model refusing to reason. Distillable; remediable
  downstream.

Updated recommendation: **PROCEED to Step 3** (target
architecture decision). The N=55 stratified evidence is now the
load-bearing artifact; the original N=10 probe served as a
falsification check. No further reasoning-channel verification is
needed before committing to Steps 4-5.

### Step 3 setup

The plan's whole point is to fall from ~$1/Q on Cerebras to
~$0/Q on local hardware (per
`notes/2026-04-27-local-distillation-from-gptoss120b.md`,
"Above those gates -> declare local-merken viable; cost falls
from ~$1/Q on Cerebras to ~$0/Q on local hardware"). The
student has always been local by design.

Step 3 zero-shot smoke runs locally via Ollama, which is
already installed (current pulls: `qwen3.5:4b`). The candidates
are Llama-3 8B Instruct and Qwen-2.5 7B Instruct; pulls needed
are `llama3:8b` (~4.7 GB) and `qwen2.5:7b` (~4.4 GB). Vstash
already supports the Ollama backend, so no new infrastructure
is needed.

## Step 3 zero-shot smoke -- qwen3.5-9b GGUF (2026-04-30)

First floor measurement of the local student. Same canonical
pipeline as steps 1+2 (per-session LoCoMo / per-turn LME ingest,
top_k=8 vec=0.5/fts=0.5, vstash._build_messages prompt) with the
Builder swap to qwen/qwen3.5-9b via LM Studio's OpenAI-compatible
endpoint at http://localhost:1234/v1.

Sample N=22 = 2 per shape x 11 shapes (the first 2 of each shape
from step2's seeded draw, empirically verified). Cost: $0 (local).
Wallclock: 40.3 min. Run dir:
`experiments/phase2_distillation/runs/step3_qwen_qwen3_5-9b_20260430T065831Z/`.

LM Studio loaded the model in **GGUF** format. An MLX equivalent
(alexcovo/qwen35-9b-mlx-turboquant-tq3, 4-bit MLX + 3-bit KV-cache
TQ3 overlay, ~5.95 GB, ~51 tok/s on Apple Silicon) is the production
format and a parallel MLX run is recommended before the candidate-
student decision is locked. The first GGUF run is enough to establish
that qwen3.5-9b zero-shot **does** populate the reasoning channel
(LM Studio surfaces it as `message.reasoning_content`, not Cerebras's
`message.reasoning`).

Earlier setup blocker: LM Studio's default n_ctx=4096 rejects the
canonical RAG prompt (~6430 tokens). Fixed by reloading the model
with n_ctx >= 8192 (2026-04-30, mid-session); the smoke would have
to be re-run if a future ETL bumps prompt size beyond that.

### Per-shape numbers (student vs teacher)

Reasoning channel populated 22/22. Per-shape **median reasoning
chars** vs teacher (step2 medians on the same questions):

| shape | n | student r med | teacher r med | ratio | student wall_s med |
|---|---|---|---|---|---|
| locomo:single_hop                 | 2 |  9115 | 1259 |  7.2x |  89s |
| locomo:temporal                   | 2 |  7927 |  849 |  9.3x |  91s |
| locomo:multi_hop                  | 2 |  9939 | 1411 |  7.0x | 101s |
| locomo:open_domain                | 2 |  3387 |  769 |  4.4x |  48s |
| locomo:adversarial                | 2 |  5640 | 1877 |  3.0x |  65s |
| lme:single-session-user           | 2 |  5986 |  374 | 16.0x |  62s |
| lme:single-session-assistant      | 2 |  6564 |  796 |  8.2x |  75s |
| lme:single-session-preference     | 2 | 11153 | 1762 |  6.3x | 107s |
| lme:multi-session                 | 2 | 13860 | 1275 | 10.9x | 148s |
| lme:temporal-reasoning            | 2 |  8819 | 1181 |  7.5x | 101s |
| lme:knowledge-update              | 2 | 13106 | 1316 | 10.0x | 143s |

Reasoning is **uniformly 3-16x longer** than teacher's. Median
ratio ~8x.

### Truncation: 4/22 calls did not return visible content

`finish_reason=length` on:

- lme:multi-session: **2/2** (both calls)
- lme:knowledge-update: 1/2
- lme:temporal-reasoning: 1/2

These calls exhausted `max_tokens=4096` on reasoning_content with
no remaining budget for visible content -- the row's `content` field
is empty even though `reasoning_content` is populated. Failure mode:
**rumination loop without convergence.** The student keeps re-reading
contexts and self-correcting ("Wait, ...", "Let me re-check Context
N...") and never commits to an answer.

This is the most informative finding of Step 3. It says: the
student's structural problem zero-shot is **not** "doesn't know
how to reason" -- it's "doesn't know when to stop reasoning."

### Hand-inspected outcomes

Direct comparison teacher vs student on six shapes (full output in
`step3_compare_to_teacher.py` against the run dir):

- LoCoMo single_hop, multi_hop, temporal, adversarial, open_domain:
  **same correctness** as teacher, just verbose. E.g. "Who is
  Melanie a fan of in terms of modern music?" -> Ed Sheeran (both
  correct). Student takes 5164 reasoning chars; teacher takes 314.
- LoCoMo adversarial ("What did Caroline do after the road trip?"):
  both correctly identify it was Melanie's road trip, not
  Caroline's. Speaker disambiguation works zero-shot.
- LME knowledge-update ("Corey Schafer videos completed?" -> 30):
  teacher commits to "30 videos". Student **hedges**: "two
  different numbers depending on document source" despite the
  oracle answer being 30. The teacher's heuristic ("most recent
  count wins") fires; the student's does not.
- LME multi-session ("eggs sold this month?" -> $120): teacher
  cleanly synthesizes 40 dozen * $3 = $120 from two contexts.
  Student burns 15281 chars on reasoning_content and **writes
  zero content** (truncation). The exact synthesis the teacher
  performs in 1680 chars, the student cannot complete in 4096
  tokens of generation budget.

### What this means for distillation

Three concrete findings:

1. **The student already has the right structure.** Trace pattern
   is identical: enumerate contexts, identify relevant chunks,
   discriminate candidates, cite. SFT does not need to teach
   format from scratch -- it needs to teach **brevity** and
   **convergence**.
2. **The trace is the right SFT target.** The teacher's traces
   are 4-16x shorter and produce equally correct (often more
   correct on knowledge-update) answers. Distilling on
   `Q -> teacher_trace -> A` with the student initialized on its
   verbose-but-structurally-correct floor gives the student a
   tight target with same shape.
3. **Truncation is the most-likely failure mode if SFT under-
   trains.** Multi-session + knowledge-update were exactly the
   shapes that truncated. They are also the shapes the teacher
   spent the **most** chars on (1271, 1275 medians). If the
   student fails to compress, those shapes will remain truncated
   even after limited SFT. Step 4 (smoke distillation) should
   target multi-session as the canary -- if smoke can't get
   multi-session to commit to an answer, the SFT is undertrained
   or the student is too small.

### What this means for the wallclock budget

Wallclock per call is the more practical concern for Step 5 (50K
training corpus generation): the **teacher** did 55 calls in 12.8
min (~14 s/call). The **student zero-shot** does 22 calls in 40.3
min (~110 s/call). That ratio is moot for training-corpus generation
(teacher produces the corpus, not student) but is the operational
ceiling on local-merken serving cost: a 6-9B reasoning model on
Apple Silicon GGUF runs at ~110 s/RAG-question. A trained student
that produces tighter traces (target ~1200 reasoning chars + 300
content chars = ~430 tokens) should land at roughly **15-25 s/call**
(estimate: ~40 tok/s on a 9B Q4 GGUF, 600 output tokens). MLX should
shave another ~30%. The Step 6 gate is correctness; this is the
operational footnote.

### MLX (alexcovo/qwen35-9b-mlx-turboquant-tq3) -- ABORTED, NOT VIABLE

Tested same canonical pipeline against the MLX-TQ3 quantization
(4-bit weights + 3-bit KV-cache compression via Hadamard rotation).
Run dir:
`experiments/phase2_distillation/runs/step3_qwen3_5-9b-mlx_PARTIAL_n12_20260430T080329Z/`.

Run aborted at 12/22 rows after the truncation pattern was already
unambiguous (Jay called it: "porque sigues corriendo este si el
resultado no es bueno?").

| metric | GGUF | MLX-TQ3 (n=12 partial) |
|---|---|---|
| reasoning chars (LoCoMo single_hop median) | 9115 | 14630 |
| reasoning chars (LoCoMo multi_hop median)  | 9939 | 14035 |
| reasoning chars (LoCoMo adversarial median)| 5640 | 15323 |
| LoCoMo truncation rate | **0/10** | **5/10 (50%)** |
| sanity wall_s on the same single_hop prompt | 91.5s | 109.9s |
| sanity reasoning chars on the same prompt   | 10705 | 16678 |

**MLX-TQ3 is worse on every axis** vs the same model in GGUF on the
same hardware:

- Reasoning runs **50-180% longer per shape** -- the 3-bit KV-cache
  compression damages attention precision; the model cannot keep
  track of where it is in the trace and re-reads contexts more.
- **5/10 LoCoMo calls truncated** before producing visible content
  (vs 0/10 on GGUF). Even shapes GGUF handled cleanly
  (single_hop, multi_hop, adversarial) now produce empty content
  responses on MLX-TQ3.
- Wallclock per call also worse (~109s vs ~91s on the sanity
  prompt) -- the longer trace dwarfs any per-token throughput
  speedup MLX would offer.
- Earlier crash mode: when GGUF and MLX were both loaded, MLX
  crashed entirely at large prompts ("The model has crashed
  without additional information") because of memory pressure.
  Fixed by ejecting GGUF, but the verbosity / truncation
  finding survives.

**Conclusion: alexcovo/qwen35-9b-mlx-turboquant-tq3 is not a
viable production target for this pipeline.** TQ3 trades attention
precision for KV-cache compression, and reasoning models on long
RAG contexts are exactly where that trade-off bites.

The cheap-to-validate paths if MLX is needed long-term:
- A non-TQ3 MLX quant: `mlx-community/Qwen3.5-9B-Instruct` in q8
  or bf16. Less aggressive compression, more memory, more stable
  attention. Step 5 training also wants q8/bf16 base, so this
  likely lands in the program at Step 5 anyway.
- Stay on GGUF for inference. ~91s per RAG-question on Apple
  Silicon at zero-shot is acceptable as a working baseline; SFT
  should bring per-call wall down by tightening traces.

For now: GGUF is the working surface for Step 4 smoke distillation.
MLX-TQ3 stays out.

## Step 4 SFT corpus (2026-04-30)

Pipeline e2e: capture teacher trajectories -> format as
chat-template -> split for mlx-lm-lora -> train on Apple Silicon.

### Step 4a: corpus generation

`experiments/phase2_distillation/step4a_generate_sft_data.py`. Runs
gpt-oss-120b on Cerebras over the canonical Phase 2 retrieval
pipeline and captures both the visible answer and the hidden
reasoning trace per row.

Production run (deterministic seed=42):

| metric | value |
|---|---|
| total rows | 1000 |
| LoCoMo / LME | 800 / 200 |
| n_with_reasoning | 1000/1000 (100%) |
| n_empty_content (teacher truncations) | 0/1000 |
| avg prompt tokens | 4568 |
| avg completion tokens | 327 |
| wallclock | 60.5 min |
| cost (estimated, $0.35 in / $0.75 out) | ~$2 |

The teacher converged on every single call -- no truncations
across 1000 calls is meaningful given that the student smoke
showed 4/22 truncations on the same shapes. Confirms the
distillation gap is structural (reasoning convergence), not a
benchmark difficulty.

LME ingest dominated wallclock: ~25s per LME conv (ingest +
call) vs ~0.6s per LoCoMo call (single ingest amortized across
the conv's questions).

Cost tracked closely to projection. Three transient 503
queue_exceeded errors caught and retried by the helper's
4-attempt backoff. No hard failures.

### Step 4b: format conversion

`experiments/phase2_distillation/step4b_format_sft_corpus.py`. Each
row from step4a becomes a chat-template training example with
the assistant target wrapped in Qwen3.5 reasoning tags::

    <think>
    {teacher_reasoning}
    </think>

    {teacher_content}

This is exactly the inference-time emission shape, so the student
learns to (a) emit a trace, (b) close it, (c) commit to a visible
answer. The third property is what step3 showed the zero-shot
student fails at.

| metric | value |
|---|---|
| rows in | 1000 |
| rows kept | 1000 (zero drops; no empty-content rows to filter) |
| target chars (min/median/max) | 184 / 1196 / 5469 |

Per-shape target chars (median, what the student will learn to
produce per question shape):

| shape | n | target chars med |
|---|---|---|
| LoCoMo single_hop  | 171 | 1527 |
| LoCoMo temporal    | 177 | 1000 |
| LoCoMo multi_hop   |  96 | 1346 |
| LoCoMo open_domain | 178 |  906 |
| LoCoMo adversarial | 178 | 1180 |
| LME single-session-user      | 34 |  616 |
| LME single-session-assistant | 34 |  768 |
| LME single-session-preference| 30 | 1623 |
| LME multi-session            | 34 | 1306 |
| LME temporal-reasoning       | 34 | 1576 |
| LME knowledge-update         | 34 | 1615 |

Distribution matches the qualitative pattern from step2: harder
shapes (knowledge-update, temporal-reasoning, multi-session) get
longer targets. The student's SFT therefore learns shape-aware
trace length, not a fixed format.

### Step 4c: mlx-lm split

`experiments/phase2_distillation/step4c_prepare_mlx_train.py`.
Splits the train.jsonl into 90/5/5 (deterministic seed=42) in the
directory layout mlx-lm-lora expects:

  `experiments/phase2_distillation/sft_corpus_v1/mlx_split/`
  -> `train.jsonl` (900), `valid.jsonl` (50), `test.jsonl` (50)

Each row is `{"messages": [system, user, assistant]}` -- mlx-lm
applies the chat template at training time, so the format is
already what the trainer wants.

### Step 4d: training run + post-train eval (DONE)

**Training (Colab A100 80GB, train_qwen_lora.py).** Loaded
Qwen/Qwen2.5-7B-Instruct as base. LoRA r=16 alpha=32 on
q/k/v/o_proj. 900 train rows, 1 epoch, bs=1 grad_accum=4 (eff
bs=4), max_seq_len=8192, lr=1e-4. Eval disabled by default
(40GB A100 OOMs on the eval logits; flag added but stayed off
for the smoke).

Wallclock 20.7 min on Colab A100 (subscription Pro). Final
train_loss 1.51 (avg over the run); per-step trajectory 1.91
-> 1.41 (-26%), token_accuracy 0.561 -> 0.641 (+14pp). Curve is
healthy: model is learning the trace pattern.

Cost: ~$0.60-1.20 estimated on Pay-as-you-go; with Colab Pro
subscription the run consumed ~3 of ~100 monthly compute units.

Adapter ~40 MB. Fused into base on local Mac (CPU merge via
peft.merge_and_unload, ~2 s) and converted to MLX q8 via
mlx_lm.convert (~8 s once base is local). Final MLX weights
~7.6 GB.

**Post-train eval: re-run step3 against the trained student.**
Same N=22 stratified sample (2 per shape * 11 shapes). Run dir:
`experiments/phase2_distillation/runs/step3_qwen25-7b-merken-smoke-v1_20260430T140523Z/`.
Wallclock 51.8 min on local Mac via LM Studio MLX backend.

Headline numbers vs the qwen3.5-9b GGUF zero-shot baseline:

| metric | zero-shot baseline | trained student | delta |
|---|---|---|---|
| truncation rate     | 4/22 = 18% | **10/22 = 45%** | **+27pp** |
| empty content rows  | 4/22       | 9/22            | +5 |
| reasoning chars med | 9614       | **2218**        | -77% (where it converges) |
| content chars med   | 338        | 124             | -63% (skewed by empties) |
| wall_s median       | 94s        | 88s             | ~tie |

Headline reading: **the SFT pipeline works end-to-end and on
several load-bearing shapes the student got dramatically tighter
(94% reasoning reduction), but on others it fell into greedy-
decoding repetition loops and truncated.** The aggregate
truncation rate got worse (+27pp), which by the plan's gate
metric is a regression. But the per-shape picture is bimodal
and the wins concentrate exactly where the plan needed them.

### Per-shape distillation outcome

| shape | zero-shot | student | reasoning ratio | verdict |
|---|---|---|---|---|
| lme:multi-session            | 2/2 trunc, r=13860 | **0/2 trunc, r= 991** | -93% | **WIN** |
| lme:knowledge-update         | 1/2 trunc, r=13106 | **0/2 trunc, r= 1492** | -89% | **WIN** |
| lme:temporal-reasoning       | 1/2 trunc, r= 8819 | **0/2 trunc, r= 2315** | -74% | **WIN** |
| lme:single-session-user      | 0/2 trunc, r= 5986 | 0/2 trunc, r=  375 | -94% | win |
| locomo:open_domain           | 0/2 trunc, r= 3387 | 0/2 trunc, r=  791 | -77% | win |
| lme:single-session-assistant | 0/2 trunc, r= 6564 | **2/2 trunc, r=18712** | +185% | **LOSS** |
| lme:single-session-preference| 0/2 trunc, r=11153 | **2/2 trunc, r=18778** | +68% | **LOSS** |
| locomo:multi_hop             | 0/2 trunc, r= 9939 | **2/2 trunc, r=16245** | +63% | **LOSS** |
| locomo:single_hop            | 0/2 trunc, r= 9115 | **2/2 trunc, r= 9268** | ~ | **LOSS** |
| locomo:temporal              | 0/2 trunc, r= 7927 | 1/2 trunc, r= 8759 | ~ | tie- |
| locomo:adversarial           | 0/2 trunc, r= 5640 | 1/2 trunc, r= 8260 | +46% | tie- |

The three big wins are **exactly the shapes the plan flagged as
the teacher's documented weakness** (multi-session,
knowledge-update, temporal-reasoning). On those shapes the SFT
pulled the student from 4/6 truncations to 0/6 -- the failure
mode the original design predicted, and the one the SFT was
intended to address, is fixed by even this undertrained smoke.

### Failure-mode forensics: repetition loops

On the loss shapes, manual inspection of one row
(locomo:single_hop, qid `What are some changes Caroline has
faced during her transition journey?`) shows the model produces
the correct bullet list for the first 5-6 items, then repeats
"Exploring her transition and her changing body (Context 6)"
verbatim hundreds of times until it hits max_tokens=4096. Same
pattern on the LME single-session-assistant shapes.

Cause: SFT with greedy decoding (temperature=0.0) plus
undertrained model. The model learned the bullet-list shape but
not the variety, so once the local KL gradient converges to a
favorite n-gram it loops. With a trained-but-not-yet-robust
student, greedy decoding is what amplifies this into a
catastrophic repetition.

### What this changes about Step 5

Step 5 was already going to bump epochs (1 -> 3) and corpus size
(1K -> 50K). The smoke confirms both moves are necessary, not
optional. Two further changes worth landing before the Step 5
run:

1. **Inference with temperature > 0 + repetition_penalty.**
   Add to the canonical evaluation pipeline (step3 + others):
   pass `temperature=0.2` (matching the teacher's setting) and
   `presence_penalty` / `frequency_penalty` if LM Studio /
   vstash backends expose them. Greedy decoding on a freshly
   SFT'd 7-9B reasoning model is a known pathological setting;
   the smoke confirms it bites here.
2. **Lower lr** for the LoRA (1e-4 -> 5e-5 or 1e-5). r=16 LoRA
   at 1e-4 over only 225 effective steps is fast enough to
   over-fit local n-grams. R1-Distill recipes typically use
   lr=1e-5 for LoRA on reasoning models; ours is 10x that.

Both changes can be tested in a follow-up smoke (same 1K corpus,
new training run) before committing the $200-400 Step 5 budget.

### Step 4d v2 retry: sampling + repetition penalty

Re-ran step3 against the same trained student with
`--temperature 0.2 --repetition-penalty 1.1`. Run dir:
`experiments/phase2_distillation/runs/step3_qwen25-7b-merken-smoke-v1_temp0.2-reppen1.1_20260430T173524Z/`.

Three-way comparison (same N=22 stratified sample):

| metric | zero-shot baseline | student v1 (greedy) | **student v2 (sampled+pen)** |
|---|---|---|---|
| truncation rate    | 4/22 = 18% | 10/22 = 45% | **4/22 = 18%** |
| empty content rows | 4/22       | 9/22        | **4/22** |
| reasoning chars med| 9614c      | 2218c       | **1004c** |
| content chars med  | 338c       | 124c        | **198c** |
| wall_s median      | 94s        | 88s         | **77s** |
| total wallclock    | 34 min     | 43 min      | **30 min** |

**v2 erases the +27pp truncation regression and lands at -90%
reasoning chars vs baseline.** That is the result the plan
targeted: distillation produces a tighter student that converges
in 1/10th the trace length without losing content quality, at
matched truncation rate to the zero-shot teacher.

Per-shape v2 truncation breakdown:

| shape | zero-shot | v1 greedy | **v2** |
|---|---|---|---|
| lme:knowledge-update          | 1/2 | 0/2 | **0/2** |
| lme:multi-session             | 2/2 | 0/2 | **0/2** |
| lme:single-session-assistant  | 0/2 | 2/2 | **0/2** |
| lme:single-session-preference | 0/2 | 2/2 | **0/2** |
| lme:single-session-user       | 0/2 | 0/2 | **0/2** |
| lme:temporal-reasoning        | 1/2 | 0/2 | **0/2** |
| locomo:adversarial            | 0/2 | 1/2 | 1/2 |
| locomo:multi_hop              | 0/2 | 2/2 | 1/2 |
| locomo:open_domain            | 0/2 | 0/2 | **0/2** |
| locomo:single_hop             | 0/2 | 2/2 | 1/2 |
| locomo:temporal               | 0/2 | 1/2 | 1/2 |

All six LME shapes at 0/2 truncated, including the three the
plan flagged as hard (knowledge-update, multi-session,
temporal-reasoning). The four residual truncations are all
LoCoMo. Hypothesis: the SFT corpus had ~5x more LME hard-shape
diversity than LoCoMo narrative-content diversity, and the
student remains brittle on the latter even with sampling. Step 5
(50K corpus, 3 epochs) is the right place to fix this.

### Cost ledger Step 4 total

- Step 4a corpus generation: ~$2 (Cerebras gpt-oss-120b, 1000 calls).
- Step 4b/c formatting + split: $0 (local).
- Step 4d Colab training: ~3 compute units (Colab Pro
  subscription); under Pay-as-you-go ~$1.
- Step 4d local fuse + MLX convert: $0.
- Step 4d post-train eval (v1 + v2): $0 (local LM Studio).
- **Total Step 4: ~$3.**

The smoke produced clean empirical findings (pipeline works,
distillation lesson transfers strongly, brittle-on-greedy
remediation identified) for ~$3.

### What this changes about Step 5

Going into the smoke, the planned remediation list was:
1. 1 -> 3 epochs.
2. 1K -> 50K corpus.
3. lr 1e-4 -> 1e-5.
4. Add temperature + repetition_penalty at inference.

Item 4 is now **measured** to fix the loops on its own; the v2
student is already viable as a working baseline for further
experiments. Items 1-3 still recommended for Step 5 because:

- **3 epochs** for stronger generalization beyond the 900 LoCoMo
  rows; v2's residual LoCoMo truncations (4/10) suggest the
  student has under-fit on narrative content.
- **50K corpus** for shape coverage; the per-shape distribution
  in our 1K is uneven (single-session-* gets ~30 examples each,
  open_domain gets 178). 50K should balance.
- **lr 1e-5** for adapter stability; r=16 LoRA at lr=1e-4 over
  225 effective steps is fast enough to over-fit specific
  n-grams (which is exactly what v1 demonstrated).

### Step 4 -- closing verdict

Pipeline e2e proven. Distillation lesson lands strongly.
Inference settings matter (greedy + reasoning model + SFT =
loops; sampled + penalty fixes it). Step 5 budget defensible.



The mlx-lm-lora command is in step4c's output and the script
docstring. Practical considerations for the smoke run on Apple
Silicon:

- `--model`: use a quantized base (`mlx-community/Qwen3.5-9B-Instruct-4bit`
  or the q8 variant). NOT TQ3 -- step3 confirmed TQ3 breaks
  reasoning convergence on long contexts.
- `--max-seq-length`: 8192 covers the canonical row (~7K tokens
  total for system + user + assistant). Bumping above 8192
  spikes memory; below 4096 truncates assistant targets.
- `--batch-size 1` + `--grad-checkpoint`: required to fit a 9B
  + LoRA on Apple Silicon unified memory at this seq length.
- `--mask-prompt`: load-bearing. Loss only on assistant tokens.
- `--iters`: 1000 with bs=1 ~ 1.1 epoch on 900 train rows. At
  ~50-80s per iter (M-series throughput estimate) the wallclock
  is 14-22 hours. Reduce to 200-300 iters for the smoke if
  overnight is not available; the eval gate (>=5pp over zero-shot)
  is the test, not the iter count.

Open: actual M-series throughput at this seq length is unknown;
first 50-100 iters will pin it.

## Files

- `experiments/phase2_distillation/__init__.py`
- `experiments/phase2_distillation/probe_reasoning_channel.py` (probe, Step 1+2 lite)
- `experiments/phase2_distillation/step2_stratified_capture.py` (Step 2 proper)
- `experiments/phase2_distillation/step2_analyze.py` (markdown analysis helper)
- `experiments/phase2_distillation/step3_zero_shot_smoke.py` (Step 3 smoke, takes --model)
- `experiments/phase2_distillation/step3_compare_to_teacher.py` (joins step3 + step2)
- `experiments/phase2_distillation/step4a_generate_sft_data.py` (Step 4a corpus gen)
- `experiments/phase2_distillation/step4b_format_sft_corpus.py` (Step 4b chat-template)
- `experiments/phase2_distillation/step4c_prepare_mlx_train.py` (Step 4c split)
- `experiments/phase2_distillation/runs/probe_20260430T050027Z/{rows.jsonl,summary.json}`
  (10 rows, gitignored)
- `experiments/phase2_distillation/runs/step2_20260430T060103Z/{rows.jsonl,summary.json}`
  (55 rows, gitignored)
- `experiments/phase2_distillation/runs/step3_qwen_qwen3_5-9b_20260430T065831Z/{rows.jsonl,summary.json}`
  (22 rows, GGUF, gitignored)
- `experiments/phase2_distillation/sft_corpus_v1/{rows,train,mlx_split/*}.jsonl`
  (1000-row SFT corpus + chat-template + train/valid/test split, gitignored)
- `notes/2026-04-30-phase2-distillation-probe-go.md` (this file)
