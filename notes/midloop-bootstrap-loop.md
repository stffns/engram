# Midloop as self-bootstrapping label factory

Drafted 2026-04-20 during the Mode A concept test. Every run of
``experiments/midloop_concept/medlocal/cerebras_midloop.py``
emits a structured audit row. The same row that gives the user
provenance at runtime is also a labeled training example. The
pipeline is therefore not just a correction loop -- it is a
label-generation loop that amortises its own training cost.

## What each audit row contains

```json
{
  "audit_id": "...",
  "timestamp": "2026-04-20T...Z",
  "builder_model": "llama3.1-8b",
  "judge_model": "qwen-3-235b-a22b-instruct-2507",
  "question": "...",
  "draft": "...",                    // what the small model said
  "retrieved": [                     // what memory returned
    {"source_id": "who-hf-...", "text": "..."}
  ],
  "judgment": {
    "has_claim": true,               // binary label for detector
    "verdict": "contradicts",        // 4-class label for verifier
    "cited_excerpt_ids": [0, 2],     // retrieval-relevance signal
    "quoted_evidence": "...",        // verbatim grounding
    "corrected_text": "..."          // preferred response
  },
  "final": "..."                     // annotated output
}
```

## Three training sets derived from the audit

``extract_training_data.py`` flattens audit rows into three
training-ready JSONL files. One audit run produces all three
without any extra API calls.

### 1. detector_examples.jsonl

- input: ``draft`` (free text)
- label: ``has_claim`` (bool)
- hint: ``verdict`` (for weak supervision)

Trains a ``NanoGPTClaimDetector``-style per-response classifier
that answers "is this draft worth verifying?" up-front. Once
trained, the detector replaces the judge for the cheap
early-exit path: if ``has_claim == false``, skip retrieval and
the judge call entirely. For Mode A this saves 1 LLM call on
roughly 30-50% of traffic in expectation.

### 2. verifier_examples.jsonl

- input: ``(question, draft, excerpts)``
- labels: ``verdict``, ``cited_excerpt_ids``, ``quoted_evidence``

Trains a verifier that can run locally (no LLM API) in Mode C.
The verbatim-evidence rule carries over -- the training label
for ``quoted_evidence`` is a substring of the excerpts, so the
student learns extractive quoting rather than generative
paraphrase. That is how we transfer the anti-leakage property
from the teacher (Cerebras qwen 235B) to the student.

### 3. preference_pairs.jsonl

- prompt: ``question``
- chosen: ``judgment.corrected_text`` (grounded, judge-approved)
- rejected: ``draft`` (small-model, contradicted)
- grounding metadata: cited excerpts + quoted evidence

DPO / reward-model pairs. Every ``contradicts`` run produces
one pair. Training on these pairs teaches the Builder to emit
drafts that would NOT have fired the judge -- it internalises
the grounded pattern so the judge becomes redundant for cases
the Builder has seen enough of.

## The closed loop

```
+-------------------------------------------------+
|                                                 |
|      user question                              |
|            |                                    |
|            v                                    |
|      Builder draft  --\                         |
|            |          \                         |
|            v           \                        |
|      retrieve           `---> AUDIT ROW         |
|            |           /                        |
|            v          /                         |
|      Judge verify+   /                          |
|      rewrite        /                           |
|            |       /                            |
|            v      /                             |
|      annotated output (to user)                 |
|                                                 |
|                          AUDIT ROW              |
|                              |                  |
|                              v                  |
|                       extract_training_data.py  |
|                              |                  |
|                   +----------+----------+       |
|                   |          |          |       |
|                   v          v          v       |
|             detector    verifier   preference   |
|                   |          |          |       |
|                   \          |          /       |
|                    \         |         /        |
|                     v        v        v         |
|              finetune smaller models             |
|                        |                         |
|                        v                         |
|              Builder gets better                 |
|                                                  |
+-------------------------------------------------+
```

Critically: the user ALWAYS sees the annotated output with
provenance (what came from memory, what was generated). The
training path is invisible to the user. The same data that
delivers provenance at runtime delivers labels at training time.

## Why this is different from "log your prompts and fine-tune"

Generic chat logs give you ``(prompt, response)`` pairs with no
label on correctness. Human feedback (thumbs up/down) gives you
weak scalar signals. RLHF needs human raters at scale.

The midloop audit gives you:
- a per-response verdict grounded in verbatim excerpt evidence,
- explicit positive AND negative pairs (for contradicts),
- retrieval-relevance signal (``cited_excerpt_ids``),
- all of it without a single human rater in the loop,
- with anti-leakage enforced by the verbatim-quote rule.

That is the missing piece that makes agent memory train itself.
Production traffic becomes training data, the judge gets cheaper
over time as the Builder learns its lessons, and retrieval
quality is measurable (cited vs retrieved) rather than a guess.

## Scale path

- Each Mode A run: ~2400 tokens, ~1s, one audit row.
- At 100 queries/day: 100 rows/day = 36k rows/year, effectively
  free (comes out of the runtime spend, no extra labelling).
- Detector training needs ~1-5k examples to surpass regex
  baselines; achievable in 2-3 weeks of organic traffic.
- Verifier training needs ~5-20k examples for a useful local
  model; achievable in 2-6 months of organic traffic.
- Preference pairs accumulate only on ``contradicts``; with a
  20% contradict rate, 20 pairs/day. Enough for first DPO
  iteration in ~3 months.

At N >= 5k, the detector can replace the judge for the cheap
path. At N >= 20k, the verifier can run locally. At N >= 5k
preference pairs, the Builder itself starts internalising the
grounded pattern. The loop accelerates -- every training
iteration makes subsequent audit rows cheaper to produce
(fewer judge calls needed) and more informative (harder cases
survive).

## What this changes operationally

- Every future midloop session should check ``cerebras_smoke.jsonl``
  (or equivalent production audit log) first; the last N rows are
  new labels that did not exist before.
- ``extract_training_data.py`` is the bridge from audit log to
  any training pipeline; keep it stable so downstream consumers
  (nanoGPT training, HF TRL for DPO) plug in without friction.
- Nothing in this scheme requires cloud tokens to produce labels
  once Mode C is wired. Local Builder + local judge = offline
  label factory.

## Non-goals (do not creep)

- Real-time training. Labels accumulate offline; the Builder is
  updated on a schedule, not per-query.
- Cross-domain transfer without explicit validation. Clinical
  labels won't teach a general-purpose Builder without
  distribution-aware re-weighting.
- Replacing human review for high-stakes domains. The midloop
  catches provenance violations, not value judgements. Clinical
  deployment still needs expert sign-off before labels flow
  back into the Builder.
