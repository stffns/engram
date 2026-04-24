# Next session: merken full pipeline vs LongMemEval

Fresh plan as of 2026-04-24 morning. Reset context: a prior
session (2026-04-23) tried three substrate-agnostic interventions
to close the seed=44 gap on LongMemEval and all three were
rejected. This plan pivots to the question that those rejections
surfaced.

## The pivot

Three rejections in the 2026-04-23 session (commits `1382c26`,
`4358612`, `6f37820`):

1. **Retrieval v2** (pure-vec pool + sharegpt_ filter +
   --retrieval-pool 50): +1 flip on seed=44 N=30, 50.0 -> 53.3%.
   Below gate.
2. **H31 splice-awareness preface**: -1 regression on 4-qid
   smoke (2b8f3739 aggregation went supports -> contradicts).
3. **Pre-inject scratchpad k=2**: -2 regressions on same smoke
   (caf03d32 meta-cognitive spiral, 2b8f3739 lock-in on incomplete
   sums).

Each attempted to feed the Builder more / better / earlier
context over a substrate of **raw conversation turns**. None
worked. The failure modes (incomplete aggregation, meta-cognitive
over-analysis) are consistent with asking the Builder to
recompose information that was never curated.

Jay's load-bearing insight (2026-04-23 EOD): **the substrate is
the problem**. We are benchmarking Mode C (and RAG-k3) against
raw-turn chunking, which bypasses merken's entire curation
pipeline -- the nanoGPT write filter, the brief_v1 consolidation
layer, the layered recall. The right question is not "Mode C vs
RAG on raw chunks" but "merken full pipeline vs RAG-k3 on raw
chunks". That is the paper-relevant comparison.

## Load-bearing prior results

Before designing the experiment, keep these numbers in mind.
They are the foundation the pipeline claim rests on.

- **nanoGPT write filter v7** (`experiments/nanogpt/RESULTS.md`):
  100% precision / 100% recall on organic work-log validation,
  100/100 on easy novel, **90% on moderate novel, 19% on subtle
  noise**. Trained on work-log domain (sprint/ticket/meeting/
  code vocabulary). **LongMemEval haystack is out of
  distribution** -- personal conversations about recipes, sales,
  health, preferences. Transfer is NOT guaranteed.
- **brief_v1 consolidation** (`experiments/consolidation/RESULTS.md`):
  86% vs 40% retrieval-only at 50 topics / 1100 events (+46pp),
  100% vs 45% at 20 topics / 440 events (+55pp). Cost ~300
  tokens prepended per query. Strong signal but measured on
  organic work-log topics, not LongMemEval question-answer
  pairs.

## Architectural claim to test

| axis | raw-chunk ingest (today's benchmark) | full pipeline |
|---|---|---|
| ingestion | every turn -> vstash | turn -> `should_remember` filter -> vstash (if keep) |
| consolidation | none | periodic `consolidate(method="brief_v1")` over kept turns |
| retrieval | `mem.search(query)` over raw turns | `mem.recall_with_briefs(query)` over brief layer + filtered episodic |
| Mode C on top | splices from raw-turn pool | splices from curated pool |

Hypothesis: on LongMemEval seed=44 N=30, merken full pipeline
beats raw-chunk RAG-k3 (63.3% today) by >= 10pp because
aggregation and multi-session questions become near-trivial
when a brief has already materialized "user sold 3 items for
$495 total across 3 market dates".

Risk: the nanoGPT write filter drops answer-carrying sessions
(the needles in the LongMemEval haystack). If it does, the
benchmark is broken before the pipeline gets a chance to prove
itself.

## Experiment sequence (gate-driven)

### Step 1 -- write-filter recall smoke (cheapest, highest risk)

**Do this before anything else.** Load nanoGPT v7 on a single
seed=44 question's haystack. Count how many of the
`answer_*` session turns the filter KEEPS. Target: **>= 95%
recall on answer-session turns**.

If recall is lower, the filter will drop needles and the
pipeline experiment is compromised before it starts. Two
fallbacks: (a) retrain the filter on LongMemEval-like data, (b)
run the pipeline with `should_remember = AlwaysWrite` so only
consolidation + layered recall do work, isolating the brief_v1
contribution.

Cost: 1 question's haystack = ~500-2000 turns. No LLM spend,
~30 seconds local inference. Read
`merken/classifiers/nanogpt.py` for load shape; the classifier
is already wired into `HeuristicWriteDecider` via shadow mode.

Gate decision:
- `>= 95%` answer-session recall -> proceed to Step 2.
- `75-94%` -> proceed with explicit caveat in RESULTS.md that
  pipeline numbers are subject to needle-drop rate.
- `< 75%` -> STOP. Either retrain the filter on LongMemEval or
  fall back to AlwaysWrite + brief_v1 only.

### Step 2 -- brief_v1 materialization smoke

One seed=44 question's haystack -> ingest (through filter from
step 1) -> run `Memory.consolidate(method="brief_v1",
synthesize_fn=<local gemma or cerebras>)`. Inspect the briefs
produced. Look for:

- Does a brief materialize the aggregation fact for 2b8f3739
  ("user sold 15 jars for $225 + 12 bunches for $120 + [third
  sale] for [Y]")? If yes, the hypothesis is on track.
- Do temporal-reasoning questions get briefs that capture date
  deltas as fields, not as free-text noise?

Cost: one haystack of ingested events, one consolidation pass.
Synthesis LLM spend is the main variable -- local gemma is
free but slow; Cerebras is fast but ~$0.0006 per brief. Budget
~$1-2 for the smoke.

Gate decision:
- Briefs capture the multi-chunk facts that raw retrieval
  missed -> proceed to Step 3.
- Briefs are vague / don't compose -> tune `synthesize_fn`
  prompt or abandon.

### Step 3 -- pipeline N=30 seed=44

Full run: ingest through filter, consolidate into briefs, then
answer each of 30 questions via `mem.recall_with_briefs` ->
Mode C (or plain RAG for the head-to-head baseline). Same
gemma-4-E4B Builder, same Gemini oracle for scoring.

Conditions to measure:
- **pipeline + Mode C**: filter + briefs + recall_with_briefs
  + Mode C streaming splice.
- **pipeline + RAG-k3**: filter + briefs + recall_with_briefs
  -> prepend top-3 -> Builder (no splice). This isolates the
  pipeline's contribution from Mode C's.
- **baseline RAG-k3** (already measured, 63.3%). Just reuse
  the existing JSONL.

Headline we are chasing:
- pipeline + Mode C > pipeline + RAG-k3 > baseline RAG-k3: all
  three levers contribute, Mode C on curated substrate is the
  winner.
- pipeline + Mode C ~= pipeline + RAG-k3 >> baseline RAG-k3:
  the substrate is what matters; Mode C's splicing is redundant
  once context is curated.
- pipeline + * ~= baseline RAG-k3: pipeline does not help on
  LongMemEval (unlikely given brief_v1's +46pp in the prior
  study, but cannot be ruled out).

Cost: ~15-20 min Mode C N=30, ~10 min RAG-k3 N=30, + ingestion
+ consolidation for each of the 30 questions (each has a fresh
haystack so consolidation runs 30 times). Budget $5-10 in
synthesis + $0.30 oracle.

### Step 4 -- document + commit

RESULTS.md section in `experiments/retrieval/longmemeval/`.
Commit the pipeline runner as a new file (not by modifying
`mode_c_benchmark.py` -- that is for substrate-less Mode C
runs; the pipeline needs its own entry point).

## Invariants (carry from prior sessions)

- **No corpus manipulation for benchmark gain.** Filtering at
  ingestion via `should_remember` is legitimate because it is
  the production path; so is `consolidate(brief_v1)` because
  it is also production. Hand-tuning briefs to pass questions
  is NOT -- the `synthesize_fn` must be the same one merken
  ships with.
- **3-seed minimum before claiming a ceiling.** If pipeline
  seed=44 lands > RAG-k3 by >= 10pp, run seeds 42 and 43
  before updating the headline.
- **Code review via `code-reviewer` subagent** before any
  N=30 run that feeds RESULTS.md. CLAUDE.md mandate.
- **Honest oracle noise disclaimer.** N=30 single-seed carries
  ~+-2pp Gemini oracle variance.
- **No bespoke compression dialect.** The brief_v1 prompts
  already live in `merken/_briefs.py` (or similar); reuse them,
  do not author new ones for the benchmark.

## Files to read on resume

- `experiments/consolidation/RESULTS.md` -- brief_v1 numbers
  and architecture. Source of truth for the +46pp claim.
- `experiments/nanogpt/RESULTS.md` -- write filter version
  history, v7 as the graduated baseline.
- `merken/classifiers/nanogpt.py` -- classifier loader, BPE /
  char-level dispatch, verb-marker prefix behavior.
- `merken/memory.py` around line 451 (`recall_with_briefs`)
  and line 500 (`consolidate`) -- the production API surfaces.
- `experiments/retrieval/longmemeval/mode_a_eval.py` `_ingest`
  and `_ingest_turn_pairs` -- where the LongMemEval haystack
  gets ingested today. The pipeline runner replaces this.
- `experiments/retrieval/longmemeval/RESULTS.md` tail (from
  line ~1380) -- the 2026-04-23 three-rejections section that
  motivated this pivot.

## What this plan is NOT

- Not a Judge post-hoc experiment (task #21 in the prior plan).
  If the pipeline experiment is inconclusive, Judge is still
  queued, but it attacks a different axis (post-Builder
  correction, not substrate curation) and should be a separate
  session.
- Not a retraining of the nanoGPT filter. Step 1 measures
  transfer; if transfer fails, the fallback is AlwaysWrite +
  brief_v1, not a retrain. Retraining is its own project
  (LongMemEval-domain data collection, labels, etc.) and does
  not belong in this experiment.
- Not a change to vstash. The public API already exposes
  `consolidate` and `recall_with_briefs`.

## First prompt for the fresh session

> Load `notes/merken-full-pipeline-longmemeval.md` and execute
> Step 1 only: measure nanoGPT v7 write-filter recall on
> answer-session turns from one LongMemEval seed=44 question's
> haystack. Report the number + qid used. Stop before ingesting
> or consolidating anything.
