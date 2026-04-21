# Midloop concept test -- Mode C, MedLocal target

Drafted 2026-04-20 after the architecture-v2 pivot. The generic
midloop (Mode A, step-boundary + LLM-as-detector) is the long-term
shape. This note is the **concept test** that validates the core
claim of the architecture -- that a small model embedded in the
generation loop can catch hallucinations BEFORE the user sees them
and inject verified corrections mid-stream.

MedLocal is the target because:

- The small model already exists (``v1c-6L``, 2.8M params,
  F1=0.461, 1 ms/step on MPS).
- The corpus is staged: 77 MedLocal protocols + 440 HF guidelines
  chunks + 101 WHO-PDF sections already chunked and labeled.
- The Builder (``gemma-4-e4b-it-mlx``) is already loaded via
  lmstudio and has produced 6k+ responses during v1c training.
- The offline / edge deployment constraint is exactly where a
  small-model-in-the-loop beats LLM-API calls -- no internet,
  no 300 ms API round-trip per check.
- Clinical correctness is the reason the concept matters: hallucination
  costs lives. A demoable correction of "amoxicillin 250 mg for
  severe dehydration" -> "Ringer's lactate 100 ml/kg PLAN C" is
  visceral evidence.

The concept test is NOT the MedLocal product. It is the smallest
runtime that proves mid-stream correction is mechanically possible
and measurably useful.

## Runtime architecture

```
            prompt (CHW question)
                 |
                 v
  +-----------------------------+
  | gemma-4-e4b-it-mlx          |
  | mlx-lm generation loop      |
  |                             |
  | token t0 -> t1 -> t2 -> ... |
  +-----------------------------+
           |  every K tokens (K=5 to 10 initial)
           v
  +------------------------------+
  | NanoGPTClaimDetector         |  <- v1c-6L ckpt loaded
  | input: prompt + response_so_far
  | output: per-position intervene prob
  |                              |
  | if max(prob[i:]) >= tau:     |
  |     FIRE at position i       |
  +------------------------------+
           |  FIRE
           v
  +------------------------------+
  | snapvec.search               |
  | query: response_so_far[i-L:i]|
  | k=5 from vstash of WHO/MSF   |
  +------------------------------+
           |
           v
  +------------------------------+
  | ClaimVerifier (rule-based v0)|
  | compares response_text to    |
  | retrieved top-1 by cosine    |
  | verdict: supports | contradicts | neutral
  +------------------------------+
           |
     CONTRADICTS
           |
           v
  +------------------------------+
  | whisper injection            |
  | option A: regenerate from i  |
  |   with system-prompt hint    |
  | option B: KV-cache rewrite   |
  |   (Phase 3 only)             |
  +------------------------------+
           |
           v
  resume generation from i with corrected context
```

Every step that does NOT fire continues at native throughput
(~60 tok/s on MPS for gemma-4-e4b). Midloop latency applies only on
fire events. Expected fire rate on clinical prompts: ~10-20% of
responses per the v1c-6L training distribution.

## Stack -- concrete components

| role | component | status |
|---|---|---|
| Builder | gemma-4-e4b-it-mlx | on disk, works via lmstudio |
| Builder loop control | mlx-lm direct (NOT lmstudio API) | verify in Phase 0 |
| ClaimDetector | v1c-6L ckpt at ``out-midloop-v1-6L/ckpt.pt`` | trained, F1=0.461 |
| Detector tokenizer | BPE vocab 512, block_size 320 | saved alongside ckpt |
| Retrieval | snapvec over vstash at ~/.merken/medlocal_concept.db | needs seeding |
| Corpus | 77 MedLocal + 440 HF + 101 WHO-PDF | in data/ staged |
| Verifier v0 | cosine drop > 0.15 vs retrieved top-1 | ~50 lines new code |
| Verifier v1 (stretch) | LLM judge (local or Cerebras) | if rule-based noisy |
| Eval harness | held-out 50 clinical Qs with expert-correct answers | needs curation |

## Phases

Five phases. Each ends with a measurable checkpoint. **Nothing ships
unless the phase below it has a green number.**

### Phase 0 -- feasibility probe (1 day)

Goal: verify the mechanical primitives are accessible before
writing pipeline code.

1. Load gemma-4-e4b-it-mlx via ``mlx-lm`` Python API directly.
   Confirm we can (a) generate tokens one at a time, (b) inspect
   the KV cache, (c) inject tokens without restarting generation.
   If mlx-lm only exposes a streaming generator, drop back to
   Mode B (abort-and-regenerate) with a note in RESULTS.md.
2. Load v1c-6L ckpt + tokenizer in the same process. Run a
   single ``(prompt, response)`` pair through it and confirm the
   per-position probabilities match what the training eval
   produced on the held-out set.
3. Open a vstash DB, embed the 618 protocol chunks using the
   project's default embedder, run a sanity ``search("severe
   dehydration treatment", k=5)`` and confirm the top result is
   PLAN C / Ringer's lactate.

Exit criterion: all three primitives work in a single Python
script. ~50-80 lines of glue.

### Phase 1 -- observation-only midloop (2-3 days)

Goal: run the full detector+retrieval+verifier pipeline on
gemma's output, log every fire event, do NOT inject anything.

1. Write ``experiments/midloop_concept/medlocal/run_observe.py``:
   wraps the mlx-lm generation loop, calls v1c-6L every K tokens
   (K=5 initial), logs (position, detector_prob, retrieved_ids,
   verifier_verdict) to a JSONL file. Generates ends-to-end
   responses with no intervention.
2. Seed vstash with the 618 protocol chunks. Tag with
   ``source:authoritative`` per the merken convention
   (CLAUDE.md, midloop-spec.md). This is a one-shot ingest.
3. Run on 20 clinical prompts. Manually inspect the fire log:
   does v1c-6L fire on the clinically wrong spans? Does snapvec
   retrieve the right protocol? Does the rule-based verifier
   agree with human judgement?

Exit criterion: on 20 prompts, v1c-6L fires on >= 60% of the
spans where gemma hallucinated AND snapvec retrieves the correct
protocol in top-3 for >= 70% of fire events. Fail the phase and
debug per-component before Phase 2.

### Phase 2 -- intervention via regeneration (3-5 days)

Goal: close the loop with a naive injection strategy and measure
end-to-end correctness improvement.

1. Extend ``run_observe.py`` -> ``run_intervene.py``. On fire +
   CONTRADICTS, (a) abort current generation at position i,
   (b) add a system-prompt suffix of the form ``"Note: per
   [source_id], the correct answer is: {retrieved_clause}"``,
   (c) regenerate from token 0 with the augmented system prompt.
2. Run on 50 curated clinical prompts (Phase 4 curates these).
   For each prompt, produce two responses: baseline (no midloop)
   and midloop (intervention enabled). Log which spans fired.
3. Score both responses against the ground-truth answer. Rubric:
   - clinical_correctness: 0/1/2 (wrong / partial / correct)
   - dose_accuracy: 0/1 (wrong / correct, if dose was asked)
   - contraindication_respected: 0/1
   - citation_present: 0/1

Exit criterion: midloop improves mean clinical_correctness by
>= 0.25 points over baseline (on the 0/1/2 scale) AND fires on
<= 30% of responses that were already correct (false-alarm cap).
If the false-alarm rate is too high, tune the detector threshold
tau before Phase 3.

### Phase 3 -- mid-stream injection (3-5 days, stretch)

Goal: eliminate the "restart from token 0" inefficiency. True
Mode C injection: continue generation from position i with the
correction spliced into the KV cache.

1. Implement ``inject_correction(kv_cache, response_tokens,
   correction_tokens, position_i)`` that (a) truncates the
   response to position i, (b) tokenizes the correction, (c)
   forward-passes correction tokens to update the KV cache, (d)
   returns a cache that gemma can continue sampling from.
2. Run A/B against Phase 2's regenerate-from-zero approach on
   the same 50 prompts. Compare latency AND correctness -- it
   is possible mid-stream injection hurts correctness because
   gemma has to coherently continue from a forced prefix it
   didn't choose.
3. If correctness holds, latency should drop by roughly 50%
   (no second full generation).

Exit criterion: injection matches regenerate correctness within
0.1 points AND reduces mean wall-time per fire by >= 40%.
If correctness drops, keep Phase 2's regenerate mode as the
shipping path and document WHY KV-splice hurt (coherence break,
distribution shift, etc).

### Phase 4 -- eval dataset + A/B report (2 days)

Goal: produce the numbers that decide whether the concept wins.

1. Curate 50-100 clinical questions with expert-verified
   correct answers. Mix: dose questions, plan-selection
   questions (PLAN A/B/C for dehydration), danger-sign triage,
   referral criteria. Source from WHO IMCI / IMAI / MedLocal's
   existing eval set if one exists.
2. Run baseline gemma and midloop-gemma on all questions.
3. Score by two raters (me + Gemini 2.5 Pro as second rater to
   guard against systematic bias). Report:
   - mean clinical_correctness baseline vs midloop
   - fire rate (% of responses that triggered)
   - correct-fire rate (% of fires where baseline WAS wrong)
   - false-fire rate (% of fires where baseline WAS right)
   - mean latency overhead per prompt
4. Ship ``experiments/midloop_concept/medlocal/RESULTS.md``
   with the full A/B table, per-question rows, and a short
   honest-caveats section.

Exit criterion: if midloop shows statistically meaningful
(bootstrap-CI 95%) improvement in clinical_correctness, the
concept is validated. If not, RESULTS.md documents why, and the
generic midloop (Mode A, LLM-based) becomes the primary track
since Mode C's small-model path did not clear the bar.

## Success criteria (the one number that matters)

On the 50-question held-out:
- **baseline clinical_correctness mean**: reference number
- **midloop clinical_correctness mean**: must beat baseline by
  >= 0.25 points on the 0/1/2 scale AND 95% bootstrap CI does not
  cross zero
- **latency overhead**: <= 2.5x baseline wall-time per prompt
- **false-fire rate**: <= 30%

Everything else is diagnostic. These four numbers decide ship or
no-ship for Mode C.

## Risks and mitigations

| risk | probability | mitigation |
|---|---|---|
| mlx-lm does not expose token-level loop control | medium | Phase 0 probes; fallback = Mode B (abort-and-regenerate via lmstudio streaming API) |
| v1c-6L F1=0.461 is too noisy to be useful mid-stream | medium-high | threshold-sweep in Phase 1 (try tau in {0.5, 0.7, 0.85, 0.9}); fail-open = no intervention if confidence low |
| tokenizer mismatch between v1c-6L BPE (vocab 512) and gemma | high | re-detokenize gemma output to text, re-tokenize through v1c-6L BPE every check. ~5 ms overhead per check; acceptable |
| snapvec retrieval too slow on MPS/CPU | low | vstash is already optimized; pre-warm the index at process start |
| KV-cache splicing breaks gemma coherence | medium | Phase 3 is explicitly stretch; Phase 2's regenerate path is the shipping fallback |
| 50-question eval set too small for CI to close | medium | expand to 100 if bootstrap CI is wide; document the N used in RESULTS.md |
| expert bias in correctness ratings | low | two-rater protocol (me + Gemini 2.5 Pro); disagreements flagged and reviewed |

## Out of scope for this concept test

Everything below is deferred to either the MedLocal product or
the generic midloop line. Do NOT let any of it creep into the
concept test scope.

- CHW UI / clinical workflow integration
- Real patient data (the 50 questions are curated, not live)
- Safety certification / regulatory approval
- Shadow-mode deployment collecting real traffic
- Generic (non-clinical) claim detection
- The fifth merken primitive (Memory.intervene_step)
- Cerebras or other cloud-API baselines (those are the Mode A
  track; including them would double scope)
- Continuous retraining / online learning

## Deliverables

At the end of the five phases:

1. ``experiments/midloop_concept/medlocal/run_observe.py``
2. ``experiments/midloop_concept/medlocal/run_intervene.py``
3. ``experiments/midloop_concept/medlocal/verifier.py``
4. ``experiments/midloop_concept/medlocal/eval.py``
5. ``experiments/midloop_concept/medlocal/eval_questions.jsonl``
   (50-100 curated clinical Qs with expert answers)
6. ``experiments/midloop_concept/medlocal/RESULTS.md``
7. ``merken/policies/claim_detector.py::NanoGPTClaimDetector``
   (wraps v1c-6L as a plug-in for the ClaimDetector Protocol;
   reuses the scaffolding already landed on develop)
8. A 30-60 second screen recording of gemma producing a wrong
   answer, the midloop firing, and the corrected answer appearing.
   This is the demo that communicates the concept in one go.

## Timeline estimate

- Phase 0: 1 day
- Phase 1: 2-3 days
- Phase 2: 3-5 days
- Phase 3: 3-5 days (stretch)
- Phase 4: 2 days

Total: 11-16 working days calendar (2-3 weeks) for a complete,
measured Mode C concept test on MedLocal. Phase 3 can be dropped
without invalidating the result -- Phase 2's regenerate path is
functionally sufficient to prove the concept.

## Why this plan is the right first step

The architecture-v2 doc lays out three Mode A paths (LLM detector,
LLM verifier, Cerebras concept test). Those answer "does retrieval-
grounded output beat baseline on a benchmark?" -- a real question.

But Jay's core thesis, from the v2 doc:

> "It is like reaching into my input while I am typing and
>  correcting me mid-keystroke, so my assertions end up as
>  verifiable facts grounded in verifiable sources."

Only Mode C delivers on that mental model. Mode A's step-boundary
correction is a RAG pipeline with extra steps. Mode C, if it works,
is a new primitive: a trained model sitting inside another model's
generation loop, silently redirecting outputs. That is the claim
the midloop makes. The concept test either proves it or forces the
v1c-6L artifact into the retrieval-baseline-competitor role it's
already been mostly repositioned to.

Either outcome is load-bearing for the research direction. Do this
first; decide Mode A priority after the numbers come back.
