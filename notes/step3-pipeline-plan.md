# Step 3: merken full-pipeline N=30 on LongMemEval seed=44

Plan of record for 2026-04-24 afternoon. Step 1 and Step 2 closed cleanly
(see memory: `project_merken_step12_2026_04_24.md`). This doc captures
the primary experiment, the gate, and the alternative hypotheses that
stay live if the primary doesn't clear the gate.

## Primary experiment

**Hypothesis**: on LongMemEval seed=44 N=30, merken full pipeline
(AlwaysWrite ingest + per-session brief_v1 + `recall_with_briefs`)
wrapped around RAG-k3 retrieval beats baseline RAG-k3-over-raw-turns
by >= 10pp in answer correctness, because multi-session aggregation
and temporal questions become near-trivial when a brief already
materializes the composite fact.

**Prior evidence**:
- qid=099778bb (multi-session, 466 events) per-session brief_v1 produced
  two briefs that together encode the aggregation fact "20 women / 100
  leadership = 20%" (Step 2 PASS, 2026-04-24).
- brief_v1 +46pp vs retrieval-only at 50 topics / 1100 events on
  work-log consolidation scenarios (2026-04-16).

**Arms, in serial order**:

1. `pipeline+RAG-k3` -- AlwaysWrite ingest, per-session briefs written
   to the semantic layer, `recall_with_briefs(query, top_k=3, brief_k=3)`,
   brief prefix prepended to Builder context, then top-3 episodic. No
   splice.
2. Baseline `rag_k3` -- already measured in `mode_c_runs_v10/` at
   63.3%. Reuse that JSONL directly; do NOT re-run.
3. (Conditional on arm 1 passing the gate) `pipeline+Mode C` --
   same substrate, Mode C streaming splice instead of prompt-prefix
   RAG-k3.

Arms are serial: if arm 1 flat-lines vs baseline, firing arm 3 is
waste. Only fire arm 3 if arm 1 >= baseline + 5pp.

**Gate (per `notes/merken-full-pipeline-longmemeval.md` Step 3)**:

Arm 1 vs baseline delta:
- `+10pp or more` -> substrate claim validated; run arm 3 for the
  Mode-C-on-curated-substrate comparison.
- `+5pp to +10pp` -> suggestive, underwhelming. Run arm 3 to see if
  Mode C pushes it over the line. Do NOT update headlines without
  3-seed replication.
- `0pp to +5pp` -> null result. Substrate curation does not help on
  this benchmark as shipped. Do NOT run arm 3. Pivot to alternative
  hypotheses below.
- negative -> pipeline HURTS. Diagnose (likely retrieval collapse on
  semantic-layer briefs). Pivot.

If the primary arm passes >= +10pp, replicate on seeds 42 and 43
BEFORE updating any RESULTS.md headline. 3-seed rule from
`CLAUDE.md`.

## Budget, time, risk

- Per-question cost: ~46 Cerebras calls for per-session briefs
  (~$0.003 each), 1 Builder call, 1 oracle call.
- 30 questions x ~50 Cerebras calls = ~1500 Cerebras calls,
  estimated $4-5 at qwen-3-235b rates.
- Oracle (Gemini): ~$0.30 for N=30.
- Builder: if gemma-4-E4B-it local, free (~3-5 min / 30 Q). If via
  Cerebras, +~$1.
- Wall time: ~25-35 min for arm 1. Plus ~15-20 min for arm 3 if we
  fire it.
- Risk: Cerebras rate-limits on 4-worker concurrency. Step 2 ran
  clean, but 30x the load in a single run could hit 429s. Retry
  with backoff already wired in the Step 2 synthesize_fn.

## Invariants

- No corpus manipulation. brief_v1 prompt text stays byte-for-byte
  identical to production `merken/consolidation.py`. The per-session
  wrapper is a LongMemEval-shape adapter, NOT a prompt change.
- `AlwaysWrite` is the fallback-filter from Step 1 (recall 86% median,
  caveat logged).
- Code review via `code-reviewer` subagent BEFORE the N=30 fire
  (CLAUDE.md mandate for runs that feed RESULTS.md).
- Honest variance disclaimer in RESULTS.md: N=30 single-seed carries
  ~+-2pp Gemini oracle variance.

## Alternative hypotheses (keep the pipeline full)

These stay queued in case the primary result is null or negative.
Ordered by signal-per-effort, highest first.

### H-A: brief retrieval fails, not brief content

If arm 1 flat-lines vs baseline, the first diagnosis is whether the
briefs are being retrieved at all. Quick probe:
- Log brief hits per question: did the 099778bb-style briefs surface
  in `recall_with_briefs` for the leadership question?
- If briefs exist but don't retrieve, the semantic-layer embedding
  for "What percentage of leadership positions do women hold"
  may not match the brief text. Action: query-aware reranking with
  BM25 over brief text, or hybrid semantic+BM25.
- Cost: quick analysis pass on arm 1's JSONL, no new runs.

### H-B: briefs retrieved but Builder can't compose

If briefs ARE retrieved but answers stay wrong, the Builder (gemma-4-E4B)
can't combine "women hold 20 leadership positions" + "team comprises 100
leadership positions" into "20%". Two candidate fixes:
- Chain-of-thought prefix: "Think step by step. If two briefs give
  numerator and denominator, compute the ratio."
- Judge post-hoc: add a second-pass Gemini Judge that re-reads the
  briefs + Builder output and emits a corrected answer.
- Cost: +1 Gemini call per question, ~$0.30 extra.

### H-C: Mode C on curated substrate beats RAG on curated substrate

If arm 1 passes +5-10pp, arm 3 directly tests whether streaming splice
on curated briefs further closes the gap. Already in the primary
sequencing.

### H-D: writer retrain via Cerebras labels (Jay's queued idea)

Out-of-distribution filter recall (86% median) is a known ceiling. Jay
proposed a midloop-style Cerebras-labels loop to train a v8 filter on
LongMemEval-domain vocabulary. This is its own project, 2-4 hours, ~$2-5
spend. Deferred to a separate session -- not a path-blocker for the
pipeline test since we use AlwaysWrite fallback.

### H-E: per-session brief wrapper changes `should_consolidate` semantics

The monolithic brief_v1 was designed with "identify topics across a
heterogeneous event stream". Per-session wrapping is a shape adapter,
but if Step 3 validates it, we should consider whether this becomes
the production default when the input has explicit session boundaries.
That is a `CONSTITUTION.md` level question; reserve for after the
benchmark is read.

### H-F: brief_v1 replaced by claim_detector / structured fact extraction

If brief_v1 shape is still too coarse (e.g., briefs materialize the
topic but not the specific numeric fact), the next rung up is
`merken/policies/claim_detector.py`-style structured claims. Briefs
become a bag of typed claims, retrieved by claim type + subject. Much
bigger change, logged here only for completeness.

## Files to create

- `experiments/retrieval/longmemeval/pipeline_runner.py` -- orchestrates
  arm 1 and arm 3. Per-question: fresh Memory, AlwaysWrite ingest, per-
  session brief wrapper, `recall_with_briefs`, Builder call, oracle
  grade, JSONL row. Reuses Builder + oracle shape from `mode_c_benchmark.py`.
- `experiments/retrieval/longmemeval/RESULTS.md` -- Step 3 section
  appended after the run, with the full arm table, oracle
  variance caveat, gate verdict.

## What NOT to do this session

- Do NOT touch `merken/consolidation.py` or the brief_v1 prompt.
- Do NOT run arm 3 unless arm 1 clears the +5pp threshold.
- Do NOT update the RESULTS.md headline without seed=42 + seed=43
  replication if arm 1 passes the +10pp gate.
- Do NOT start the Cerebras-loop writer retrain in this session.
