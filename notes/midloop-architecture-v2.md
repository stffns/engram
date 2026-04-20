# Midloop architecture v2 -- claim detection + snapvec verification

Clarified 2026-04-20 after reviewing the v0-v1c training outcomes.
The domain-specific per-position tagger we trained (v1c-6L, VAL F1=0.461)
is NOT the generic midloop. It is a clinical-claim divergence detector
useful for MedLocal. This note re-specifies what the generic midloop
actually is.

## Core thesis (Jay, 2026-04-20)

The midloop exists to **improve every conversation output by grounding
the model's responses in real stored data, not fabricated facts**.

### The mental model

> "It is like reaching into my input while I am typing and correcting
> me mid-keystroke, so my assertions end up as verifiable facts
> grounded in verifiable sources." -- Jay, 2026-04-20

Think of it as a grammar-checker, but for factual claims. As the
Builder model streams tokens that will become a user-visible
response, the midloop looks at each emerging step, notices "this
is an assertion", checks the memory, and edits the draft with a
citation-backed correction BEFORE the Builder finishes the
sentence. The user sees a response that is factual-by-construction,
not one that is plausible-sounding-then-audited.

Contrast with what we actually trained:

- What we trained (v1c-6L): memorised the answers to the
  clinical questions, so when the Builder writes "amoxicillin
  250 mg for severe dehydration" the model recognises that
  string as misaligned and flags it. This only works for
  questions whose answers were in the training set.
- What the midloop should be: a typing-assistant that, when the
  Builder writes "for severe dehydration, administer...", pauses,
  asks memory "what does the WHO protocol for severe dehydration
  say about treatment?", reads back Ringer's lactate 100 ml/kg
  PLAN C, and whispers "use this, cite \`who-hf-plan-c\`".

The generic midloop requires zero content in its own weights.
Content lives in vstash, gets updated when new guidelines ship,
gets cited when injected. The midloop only needs the skill of
"notice this is a claim" + "formulate a retrieval query" + "relay
the retrieved content back to the Builder."

### What the analogy does NOT imply (Jay, 2026-04-20)

The autocomplete framing can mislead into thinking the midloop
pastes retrieved snippets directly into the Builder's output.
That is NOT the design.

The Builder is still the one **composing** the final response,
coherently, in its own voice. The midloop's WHISPER delivers:

- retrieved facts,
- their source citations,
- optional correction hints ("memory says X, you wrote Y"),

as context the Builder reads before continuing. The Builder then
integrates those facts into its next tokens in a grammatically
and rhetorically coherent way. What the user sees is ONE fluent
response, not a collage of retrieved chunks interleaved with
model text.

Consequence: the WHISPER is **hints, not copy-paste**. The LLM
remains the author of the turn. The midloop is the fact-checker
whispering in the author's ear mid-paragraph. The writer at the
end of the turn persists the AUTHOR'S final prose (with metadata
about which retrieved events were referenced), not the retrieved
events themselves.

Implication for Builder API: the WHISPER channel needs a format
that signals "integrate this factually, re-author the in-flight
response" rather than "splice this text literally". The spec's
WHISPER InterventionAction is intentionally vague on the payload
so this decision lives with the Builder integration (Anthropic
streaming, lmstudio, Reforge, etc.) rather than the midloop core.

This is the opposite of what a trained-on-domain-protocols detector
does. A tagger that memorises WHO dosing tables has re-encoded the
same facts in weights that also live in text in the authoritative
corpus. The midloop's job is to **connect the model to the data that
already exists**, not to re-emit it from a compressed representation.

The shape follows from that thesis:

- Detection must be content-agnostic (classifying "this is a claim"
  rather than "this claim is wrong").
- Verification comes from a retrieval call into persistent memory
  (vstash + snapvec, sub-second).
- Reinforcement or correction is produced by comparing the step's
  claim against the retrieved events, then WHISPERed back to the
  Builder with citations.
- The writer persists the final turn so subsequent turns in the
  same conversation see the verified prior output without
  re-retrieving.

Nothing in this path requires domain knowledge in the midloop
model. The knowledge lives in vstash, where it can be updated,
dated, versioned, and cited. Model weights are not a database.

## What the generic midloop does

The midloop watches the Builder's output in flight and decides, at
each step, whether what the model is about to emit needs to be
cross-checked against memory. It is a **meta-policy**: it does not
know the content of any domain, only the shape of trajectories that
benefit from verification.

Four decisions per step, ordered:

1. **Anomaly check** (reuses `HeuristicMidloopDecider` signals,
   domain-agnostic): is the trajectory LOOPING / DRIFTING / STUCK?
2. **Claim check** (NEW): does this step make a factual assertion
   or express uncertainty that is verifiable against memory?
3. **Retrieval** (snapvec, sub-second top-N): if (1) or (2) fires,
   query memory with the step's salient terms. vstash's snapvec
   backend returns top-3 or top-5 at < 500 ms.
4. **Reinjection**: if retrieved content contradicts or
   supplements the step, inject a WHISPER: "memory says X; consider
   revising/reinforcing your response." The Builder continues with
   this guidance in context.

After the step completes, the **writer** persists the final
response and metadata so the next turn in the conversation can
reuse the context (without re-retrieving if nothing changed).

## Why this is different from what we trained

`NanoGPTMidloopDecider` (v0 -> v1c-6L, this session) did steps 2+3
merged into a single per-position classifier trained on clinical
protocols:

- Its input was `prompt + response_tokens[:i]` (raw token context).
- Its output was `intervene yes/no` per subword position.
- Its training data was clinical protocols (WHO/ICRC) -> it
  memorised dosing tables, antimalarial names, PLAN C for
  dehydration, etc.

Consequence: the model's "intervene" prediction is entangled with
whether it has seen the specific protocol text during training. It
cannot generalize to non-clinical domains without retraining on
those domains' protocols. It is a closed-world claim checker, not
the retrieval-enabled midloop we want.

## The correct factoring

```
step N produced by Builder
  |
  v
HeuristicMidloopDecider.decide(observation, ctx, trajectory)
  |
  |-- ON_TRACK + not a claim  -> no action, continue
  |-- LOOPING / STUCK         -> WHISPER with repetition hint
  |-- DRIFTING                -> trigger retrieval
  |-- claim_detector fires    -> trigger retrieval
  |
  v
snapvec.search(step_text, k=5)  # sub-second against vstash
  |
  v
ClaimVerifier.compare(step_text, retrieved_top_k)
  |
  |-- supports step        -> WHISPER "memory backs this: [cite]"
  |-- contradicts step     -> WHISPER "memory says: [alternative]"
  |-- no useful match      -> no intervention
  |
  v
Writer.persist(final_response, ctx)  # for next-turn continuity
```

Each box is a replaceable component. The generic midloop owns the
trajectory/anomaly detection and the dispatch. Domain-specific
models (like v1c-6L for MedLocal) plug in as ClaimVerifiers, not
as the midloop itself.

## What the ClaimDetector needs to be

Input: `(step_text, prev_steps, task_description)` -- agnostic of
domain.
Output: `(has_claim: bool, claim_span: (start, end), claim_type:
enum)`.

`claim_type` categories (from the Cognitive Companion literature +
Jay's synthesis):

- `factual_assertion` -- "X is Y", "the dose is Z mg", "patient
  has condition W"
- `uncertainty_marker` -- "I think...", "possibly...", "could be..."
- `prescriptive` -- "administer X", "avoid Y", "escalate to Z"
- `reasoning` -- "because X, therefore Y" -- these claim a causal
  link that may itself be a factual assertion

A 2-3M param tagger (same scale as v1c-6L) is more than enough for
this task because it has no vocabulary to memorise. Most of the
capacity can go into learning the LINGUISTIC signature of claims
vs chatter vs planning, which is domain-agnostic.

Training data: a small (2-5k) LLM-labeled corpus of step texts
annotated with `has_claim` / `claim_type`. Much cheaper to produce
than the Gemini-generated clinical cases. Can be synthesised from
any task logs (Claude Code transcripts, Reforge task traces, etc.).

## What the ClaimVerifier looks like

Input: `(step_text, retrieved: list[MemoryEvent])`.
Output: `VerifierDecision(supports / contradicts / neutral,
confidence, cited_event_ids)`.

Two implementations to consider:

1. **LLM-based** (first impl, least plumbing). Call Gemini 2.5
   Flash with step_text + retrieved top-5, ask "does the memory
   support, contradict, or not address the step?" Sub-second with
   structured output.
2. **Embedding-cosine + rules** (if LLM latency too high): compute
   cosine between step and each retrieved event, plus a
   rule-based NLI check for negation patterns ("not",
   "contraindicated", "instead of", etc.).

The v1c-6L model we trained IS a third option for specific
domains: if the Builder is MedLocal and the retrieved events are
WHO protocols, v1c-6L already encodes the specific dosing
knowledge. For that domain pair, it works without a separate
ClaimVerifier call.

## The Writer

Mirror of `merken.Memory.remember()`. Already exists. The
addition is a per-turn conversation-level cache so the next
Builder step sees the verified previous response without re-
retrieving.

## Roadmap

The generic midloop (Phases 4/5 of the plan) is unblocked once
three pieces land:

1. `ClaimDetector` (2-3M param, to be trained on LLM-labeled step
   traces). Small dataset, ~$10-20 Gemini.
2. `ClaimVerifier` wrapper (LLM-first). A Python class that takes
   a step_text + retrieved events and returns a verifier decision.
3. `snapvec.search(...)` wrapper around vstash, already exists
   (vstash 0.32.0 + snapvec 0.7.1 both installed per the spec).

Task #8 (lower gate to F1>=0.50 shadow-only) was always about the
v1c-6L clinical tagger, which is now repositioned as a domain-
specific tool. Shadow deployment for the GENERIC midloop uses the
Heuristic primary + ClaimDetector shadow pattern once
ClaimDetector lands.

## Archive: v1c-6L as MedLocal prototype

v1c-6L is NOT wasted. It is the end-to-end prototype of:

- HF-guidelines filtering + chunking by clinical density
- Gemini structured-output case generation at scale (zero
  parse failures over 8k+ calls this session)
- lmstudio gemma response generation (zero failures, 6k+ calls)
- Aligner + semantic filter + boundary-label training pipeline
- nanoGPT backbone + per-position tagging head

The artifact itself is a clinical-claim divergence detector
tuned for MedLocal's CHW assistant scenario. It plugs into the
generic midloop as a domain-specific ClaimVerifier when the
retrieval path happens to surface WHO / ICRC protocols.

See `notes/medlocal-clinical-claim-detector.md` for the
repositioning.
