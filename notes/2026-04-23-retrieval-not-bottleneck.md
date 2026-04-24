# Retrieval is not the bottleneck on RAG-k3 (LongMemEval N=30)

Seed note for a future paper / blog post / memory entry. Captures
why the retrieval roadmap pivoted on 2026-04-24.

## Hypothesis (prior)

Improving retrieval quality improves end-to-end answer quality in
long-context memory systems. Specifically: if an encoder places the
answer-containing turn closer to the query (higher R@1), a downstream
RAG pipeline should answer more questions correctly.

This is the motivation behind:
- the entire filter/classifier track (v7 OOD, v8a/b/c nanoGPT filters),
- `notes/bge-small-lme-ft-plan.md` (v5-ft bge contrastive fine-tune),
- the investment budget we spent on encoder work instead of builder work.

## Experiment that falsified it

`experiments/retrieval/bge_lme_ft/` -- LoRA contrastive fine-tune of
`BAAI/bge-small-en-v1.5` on 1128 LongMemEval positive pairs with
same-session "no"-labeled hard negatives (4 per positive), 3 epochs
at lr=1e-4, q/k/v attention-only LoRA (rank 8).

Three-gate validation:

| gate            | observation                                                    |
|-----------------|----------------------------------------------------------------|
| BEIR-mini       | R@5 mean -0.97pp vs zero-shot (PASS; under 1.0pp threshold)    |
| LME 3-seed R@1  | +10.0pp (67.8% -> 77.8%, clean across seeds 42/43/44)          |
| LME 3-seed R@5  | +3.3pp (92.2% -> 94.4%)                                        |
| **Gate 3 RAG-k3** | **flat 60.0% vs 60.0% (18/30 supports each)**                |

Per-question analysis on Gate 3: 29 of 30 verdicts identical; 1 qid
flipped from `contradicts` to `neutral` (weaker failure mode, same
binary accuracy). The +10pp R@1 retrieval gain did not translate to
any supports-rate gain at RAG-k3 N=30.

## Why the gain didn't cross over

bge zero-shot already places the answer-session in its top-3 for
~96% of LongMemEval questions at N=30 (observed R@10 for both
encoders is 96.7%). When the baseline encoder and the fine-tuned
encoder both hand the same top-3 excerpts to the builder, the builder
produces the same answer, and the oracle assigns the same verdict.

The R@1 gain is real, but only matters for consumers that weight the
top-1 hit (rerankers, KV-splice / Mode C, citation footers). For a
top-k pooling pipeline like RAG-k3 where all k excerpts enter the
prompt equally, R@1 gains are invisible once R@k is near-ceiling.

## New hypothesis

**Builder composition is the bottleneck on RAG-k3.** In the 40% of
questions where the pipeline fails to produce a supports-grade
answer, the answer-bearing excerpt is (empirically) in the top-3
more often than not. What fails is one or more of:

- builder attending to the wrong excerpt in the k=3 pool,
- builder failing to compose facts across multiple excerpts,
- builder hedging when evidence conflicts or is imprecise,
- prompt shape making it hard for the builder to identify the
  authoritative excerpt.

This predicts:

- Improvements to builder prompt structure, context ordering, or
  post-hoc judging should move the number.
- Further retrieval work (rerankers, larger k, better encoders) on
  the same RAG-k3 substrate will be flat or near-flat.
- Consumers that dereference a single top hit (Mode C streaming,
  KV-splice, direct citation generation) should see encoder gains
  cross over where RAG-k3 did not.

## Stronger test: encoder swap on Mode C regressed -13pp

The third prediction turned out to be wrong in an instructive way.
Swapping v5-lora into Mode C (H18 preface, seed=44 N=30) produced
**11/30 = 36.7% vs bge baseline 15/30 = 50.0% -- a -13.3pp
regression**, not an improvement.

Post-mortem: Mode C's retrieval path has calibrated score
thresholds (e.g. `score_threshold=0.0161`, retrieval window cuts)
that were tuned against bge-small-en-v1.5's cosine distribution.
v5-lora produces different cosines for the same (query, chunk)
pairs. Many answer-bearing chunks that would have passed the
threshold under bge fell below it under v5-lora, triggering
Mode C's FTS-only fallback path (vstash log: `vec_w=0.00 fts_w=1.00`
observed on most queries after the first). FTS-only retrieval
on LongMemEval is weaker than hybrid -> Mode C's top-3 chunk
quality drops -> gemma generates worse answers.

The correct bookkeeping:

- v5-lora genuinely improves retrieval metrics (R@1 +10pp).
- A pipeline whose thresholds were calibrated for bge cannot
  absorb a different encoder without recalibration.
- "Drop in a better encoder" is not a safe move across pipelines.

This extends the hypothesis: **encoder-side gains require
matched recalibration of every downstream consumer to even show
up, let alone help.** Without recalibration, they can actively
regress.

## Roadmap implication

Retrieval work on this substrate is saturated AND potentially
harmful to uncalibrated downstream consumers. The next budget goes
to builder-side work, not retrieval-side work:

1. **Builder bottleneck attack** -- queued moves from memory:
   Judge post-hoc (task #21) and structured claim extraction (H-F).
   These operate on top-k output, not retrieval. If they pick up
   even a few of the 12 failing questions on RAG-k3 N=30, we've
   confirmed the hypothesis and found real gain.

2. **Mode C threshold recalibration** -- if someone ever wants to
   ship v5-lora in Mode C, the score thresholds (0.0161 absolute
   cutoff + distance filters) need a matched sweep on v5-lora's
   distribution before a head-to-head is even meaningful. Do not
   attempt until step 1 produces independent evidence.

## Update 2026-04-24 EOD: Mode C recalibration partial (-13.3 -> -6.7pp)

Two recalibration variants ran on Mode C H18 seed=44 N=30 with
v5-lora as encoder:

| variant                                     | strict supports | delta vs bge |
|---------------------------------------------|----------------:|-------------:|
| bge baseline                                |     15/30 = 50.0% |      --     |
| v5-lora default                             |     11/30 = 36.7% |    -13.3pp  |
| v5-lora `--relative-threshold-factor 0.5`   |     13/30 = 43.3% |     -6.7pp  |
| v5-lora `--bypass-score-threshold`          |     13/30 = 43.3% |     -6.7pp  |

Two independently-motivated knobs converge at the same plateau.
One knob recovers half the gap; removing the absolute cutoff
entirely does not recover any additional ground. The remaining
-6.7pp comes from a second layer: Mode C's adaptive-RRF vector-empty
fallback (logged as `vec_w=0.00 fts_w=1.00`) still triggers on
v5-lora cosines, so FTS-only retrieval still dominates later queries.
Recalibrating that fallback is a third knob not yet tried.

Net reading: "encoder ABI includes score distribution" is real,
with the addendum that Mode C has at least two correlated calibration
surfaces, not one. A full v5-lora ship into Mode C would need a
joint sweep (threshold + adaptive-RRF floor), not a one-line flag.

## Update 2026-04-24 EOD: Judge post-hoc over merken brief_v1 (net +2, marginal)

First builder-bottleneck attack. `experiments/retrieval/longmemeval/
judge_post_hoc.py` runs a Cerebras qwen-3-235b Judge over the
baseline row's (question, brief_hits, episodic_hits, builder_answer)
and either approves or rewrites grounded in the SAME context the
Builder saw (never ground truth). Gemini oracle re-grades rewrites.

Substrate: the available baseline jsonl is NOT RAG-k3 pure -- it's
the merken brief_v1 full pipeline (already rejected at -18.5pp vs
RAG-k3 in `project_merken_step3_rejected_2026_04_24.md`). That means
the headroom is larger than RAG-k3 and the experiment doesn't
speak directly to the RAG-k3 bottleneck hypothesis, only to "can a
second reasoning pass save a rejected substrate?"

Seed=44 N=30, 27 rows with baseline oracle verdict:

| metric             | value     |
|--------------------|----------:|
| Judge approves     |     15/27 |
| Judge rewrites     |     12/27 |
| rewrite still wrong|      8/12 |
| gain (fail->correct)|      3    |
| loss (correct->fail)|      1    |
| **net**            |    **+2** |
| cost               | ~$0.08 + 48s |

Interpretation: +2 is in the marginal band of the pre-registered
criteria (>=+3 = strong; +1..+2 = marginal; <=0 = rejected).
Not a green light to scale to 3 seeds.

Qualitative inspection of flips:
- All 3 gains are **multi-fact temporal reasoning** fixes the
  original Builder missed: "met jam-seller vs tourist first",
  "volleyball first vs 5K run", "tennis weekly vs 'table tennis'
  not mentioned". Judge succeeded by ordering events by date.
- The 1 loss is **over-literalism**: baseline inferred "poster
  presentation at Harvard" from "attended conference at Harvard,
  presented poster"; Judge refused to stitch the implicit
  inference and returned "not found in memory".

This is a clean signal about the class of failure the builder
has: when multiple facts must be combined with dates, the Builder
doesn't do the reasoning. When single facts require one obvious
inference, the Builder does it well enough that an LLM second pass
can hurt.

Implication: a generic LLM Judge is not the right intervention --
it partially helps the failure class but introduces a new one.
**Structured claim extraction (H-F)** is a better fit: extract
(entity, predicate, timestamp) tuples first, reason over them
with explicit temporal ordering, then render. That targets the
gain class directly and avoids the inference-refusal loss class
because the stitching happens in the extraction step, not in a
refusal-prone "is X supported by Y" gate.

Artifact: `experiments/retrieval/longmemeval/judge_post_hoc.py`.
Outputs: `*.judge_v2.jsonl` + `*.judge_v2.summary.json`.

## Update 2026-04-24 EOD+1: Structured claim extraction also net +2 (tied)

`experiments/retrieval/longmemeval/claim_extract_post_hoc.py` was
written to test the hypothesis above: does a question-shape-aware
extract-then-reason pass outperform the generic Judge on the
multi-fact temporal reasoning class? Same substrate (merken
brief_v1 pipeline seed=44 N=30, 27 graded rows), same Cerebras
qwen-3-235b model, same Gemini 2.5 Flash regrade.

Result: **net +2, identical to Judge, but with a different flip
distribution.**

| approach               | gain | loss | net | parse_fail |
|------------------------|-----:|-----:|----:|-----------:|
| Judge post-hoc (generic) |   3  |   1  | +2  |         0  |
| Claim extract (structured) | 4  |   2  | +2  |         4  |

Per-qid diff:
- Claim wins on 2 qids Judge didn't: `4f54b7c9` (counting antiques,
  GT=5 -- extractor enumerated all 5 items; baseline miscounted 4)
  and `86f00804` (current book -- extractor correctly identified
  which of 2 mentioned books is "currently reading"; baseline
  hedged). Both are enumeration tasks where the
  "list-then-reason" discipline shines.
- Claim loses on 2 qids Judge didn't: `gpt4_d31cdae3` (Europe trip
  first -- extractor over-literally said "Europe trip not in context",
  baseline correctly inferred order) and `gpt4_ec93e27f` (bus vs
  train most recent -- extractor reasoned confidently but wrong:
  "bus commute is ongoing, therefore bus is more recent" -- GT=train).
  The second loss is a new failure class: hallucinated reasoning
  over correctly-extracted facts.
- Both got `gpt4_213fd887` and `f685340e_abs`. Judge uniquely got
  `gpt4_0a05b494` (jam seller gender mismatch -- extractor
  refused the identity stitch when context said "he").

Reading: the union of gains is 5 distinct qids, the union of
losses is 3. A router that picks per question shape could in
theory reach +4 net (counting via extractor, ordering via
Judge), but the complexity cost of running two LLM passes +
shape-classifier is not justified by +2 over a single approach.

What the two experiments agree on:
- Builder composition is a real bottleneck on this substrate --
  both approaches move the number by the same small amount.
- The improvement is capped at ~+2 on N=27 (7pp) which is close
  to oracle noise; net wouldn't stay stable across seeds without
  a different intervention class.
- Multi-fact temporal reasoning is the only failure type that
  answer-time LLM passes consistently fix. Other failures
  (inference refusal, literal-match mismatches, over-confident
  reasoning) substitute one form of wrong for another.

Implication: builder-side answer-time interventions saturate
near +2 net on this substrate. Larger gains require either a
different substrate (RAG-k3 pure, different benchmark) or a
different intervention class (structured ingestion, better
prompt shape for the Builder itself, or a different model).

Artifacts:
- `experiments/retrieval/longmemeval/claim_extract_post_hoc.py`
- `*.claim_extract.jsonl` + `*.claim_extract.summary.json`

What we are NOT doing: a fourth encoder, a reranker on top of
v5-lora, a bigger training corpus for v5-lora, dropping v5-lora
into other pipelines without recalibration. The substrate saturates
before those moves would show up, and the recalibration cost for
each downstream consumer is nontrivial.

## Meta-lesson

"Upstream optimization only shows up at the bottleneck." A common
assumption in RAG systems is that embeddings are always the lever
because they are the most measurable component. The measurability
makes it easy to spend budget there. LongMemEval at RAG-k3 is a
counter-example: measurable encoder gains of +10pp R@1 translate to
0pp end-to-end at a k that already saturates recall. Before spending
the next retrieval budget, measure whether retrieval is on the
critical path for the consumer you care about.

Related references:
- `experiments/retrieval/bge_lme_ft/RESULTS.md` (full numbers)
- `notes/bge-small-lme-ft-plan.md` (original plan that this falsifies)
- Memory: `project_v5_ft_gate3_null.md`, `project_mode_c_production_target.md`
