# LongMemEval results

## Scope

This file records absolute R@5 numbers on LongMemEval. **It does not
measure whether merken's loop adds value** — LongMemEval is chat-replay
with no duplicates in the haystack, so the decision primitives collapse
to "ingest everything." For the benchmark that tries to catch loop
value, see `../../loop_quality/`.

## How to read this file

Every row records:

- **Date** — when the run was made.
- **Commit** — the merken commit SHA the run was made against.
- **Baseline** — `vstash` (substrate only) / `merken-always` (no
  filtering) / `merken-heuristic` (Phase 1 default decider).
- **Subset** — `longmemeval_s` (full distractor haystack) /
  `longmemeval_oracle` (oracle context only — sanity, not signal).
- **n** — number of questions evaluated. **Anything below ~50 is
  sanity, not signal.**
- **R@5** — recall @ 5 with 95% bootstrap CI (1000 iters, seed 0).
- **API/q** — API calls per query. Engram is local-first; always 0.
- **Notes** — what was different about this run, what we learned.

## Results

| Date | Commit | Baseline | Subset | n | R@5 (95% CI) | API/q | Notes |
|------|--------|----------|--------|---|--------------|-------|-------|
| 2026-04-08 | `2bcf502` | `merken-always` | `longmemeval_s` | 3 | 1.000 [1.000, 1.000] | 0 | First real-data run. **Sanity only — n=3 produces a degenerate CI.** |
| 2026-04-08 | `2bcf502` | `merken-heuristic` | `longmemeval_s` | 3 | 1.000 [1.000, 1.000] | 0 | Same caveat. With no exact duplicates in the haystack, behaves identically to `merken-always`. |
| 2026-04-08 | `2bcf502` | `vstash` | `longmemeval_s` | 3 | 1.000 [1.000, 1.000] | 0 | Substrate-only baseline. Same caveat. |
| 2026-04-08 | `e18d7d4` | `merken-heuristic` | `longmemeval_s` | 10 | **0.900** [0.700, 1.000] | 0 | First non-degenerate CI. 9/10 hits. seed=42. |
| 2026-04-08 | `e18d7d4` | `vstash` | `longmemeval_s` | 10 | **0.900** [0.700, 1.000] | 0 | Identical hit set to merken-heuristic — confirms the heuristic decider is a no-op on this dataset. seed=42. |
| 2026-04-13 | `5a6c820` | `vstash` | `longmemeval_s` | **500** | **0.964** [0.948, 0.978] | 0 | **Phase A complete.** Full n=500 run, seed=42. Positions merken's substrate at parity with mempalace's claimed 96.6% raw (CIs overlap). |
| 2026-04-13 | `5a6c820` | `merken-heuristic` | `longmemeval_s` | **500** | **0.964** [0.948, 0.980] | 0 | Identical R@5 to vstash raw. Budget redistribution fix (commit `42d40ef`) closed the gap that existed at n=10. Heuristic decider is a no-op on this dataset (no duplicates). |
| 2026-04-14 | `482025e` | `merken-heuristic` | `longmemeval_s` | 100 | **0.980** [0.950, 1.000] | 0 | **Embedder swap probe.** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` instead of the default `BAAI/bge-small-en-v1.5`. n=100 (not 500) because the question is "does multilingual regress on English" — a no-worse check that tolerates a wider CI. CI [0.950, 1.000] overlaps the bge row's [0.948, 0.980] → no regression. Colab CPU, 40 min. Combined with the bilingual scenario in `../../loop_quality/RESULTS_multilingual.md`, this clears multilingual as the recommended embedder for bilingual users. |

### Wall-clock cost (informational, not part of the metric)

| Baseline | n | Elapsed | Per-question |
|---|---|---|---|
| `merken-always` | 3 | 92.0 s | ~30.7 s |
| `merken-heuristic` | 3 | 170.2 s | ~56.7 s |
| `vstash` | 3 | 129.2 s | ~43.1 s |
| `merken-heuristic` | 10 | 405.5 s | ~40.6 s |
| `vstash` | 10 | 355.8 s | ~35.6 s |
| `vstash` | **500** | 8923.0 s (2.5h) | ~17.8 s |
| `merken-heuristic` | **500** | 10166.6 s (2.8h) | ~20.3 s |

The n=500 run used vstash 0.28.0 batch ingest for the vstash baseline
(single-transaction writes), cutting per-question cost from ~35s to
~18s. merken-heuristic still ingests sequentially (events pass through
the decider one by one) at ~20s/question — 14% overhead from the audit
log, consistent with the n=10 measurement.

Run on Mac (MLX backend on Apple Silicon, vstash 0.28.0).
~490 turns ingested per question median.

**Full 500-question extrapolation, this hardware:** ~6 hours per
baseline. CONSTITUTION §9's "under 5 minutes on a laptop" target is
**not currently met** for the full LongMemEval_s split. Two options:
(a) use a faster local embedder profile in vstash, (b) accept that
LongMemEval needs an overnight run and amend §9's language. Tracked
in the chat history; not yet in an issue.

## What the n=500 run says (Phase A complete)

- **merken-heuristic and vstash are tied at 0.964 (R@5).** CIs
  overlap completely: [0.948, 0.978] vs [0.948, 0.980]. The budget
  redistribution fix (commit `42d40ef`) closed the gap that existed
  at earlier runs where the empty semantic layer was stealing slots.
- **96.4% positions merken at parity with mempalace's 96.6% raw
  claim.** The 0.2pp difference is well within the CI. Engram's
  substrate (vstash with bge-small-en-v1.5) is competitive with the
  strongest verified raw-mode claim in the space.
- **18 questions missed out of 500.** These are the questions worth
  investigating — each represents a retrieval failure where the
  correct session's turns were in the haystack but didn't land in
  top-5. A `--dump json` flag would help identify patterns.
- **Heuristic exact-dedup still contributes zero on this dataset.**
  Same R@5 as vstash raw. Confirmed at scale what n=10 showed.

## What the n=10 run said (historical)

- merken-heuristic and vstash tied at 9/10 (R@5 = 0.900). CI was
  [0.700, 1.000] — too wide for claims. The n=500 run narrowed this
  to [0.948, 0.980].

## Findings from the first real-data runs

1. **The pipeline works.** End-to-end ingest → recall →
   session-attribution → R@k → bootstrap CI matches the design. No
   schema surprises against the real LongMemEval-cleaned dataset.
2. **Heuristic exact-dedup adds nothing on LongMemEval.** Real
   haystacks have no exact duplicates, so `HeuristicWriteDecider`'s
   `dup_exact` rule never fires and merken-heuristic collapses to
   merken-always behaviorally. This rule's value can only show in
   live agent loops where the same content gets re-ingested. **The
   benchmark in this directory cannot validate this rule.** It needs
   a different benchmark — the one in `../../loop_quality/`.
3. **Recall-based dedup was prohibitive.** The original
   `HeuristicWriteDecider` ran a vstash hybrid search on every write
   to detect duplicates. That made ingest O(N²) per haystack and the
   3-question run was killed mid-flight at ~25 minutes per baseline.
   Replaced with an in-process `set[str]` of normalized text. Same
   semantics, hash-fast. The recall callable stays in `WriteContext`
   for future similarity-based deciders that genuinely need it.
4. **Audit log overhead is not free.** Every decision (write or skip)
   triggers an extra `vstash.remember` call to the audit collection.
   On 3 questions × ~490 turns × 3 baselines, that's ~4400 extra
   writes beyond the user-facing ingest. Worth measuring on a bigger
   run before we decide whether to batch audit writes.
5. **Wall-clock variance between baselines is unexplained.** With one
   DB per (baseline, question), `merken-always` (92 s) < `vstash`
   (129 s) < `merken-heuristic` (170 s) is suspicious — the three
   baselines should be within ~10% of each other on identical
   hardware. Possibilities: model warmup spread across baselines,
   audit collection growth, or a per-DB cold-start cost.

## Competitive positioning (updated 2026-04-13)

The n=500 result positions merken against the published landscape:

| System | R@5 | Mode | Actually tests the system? |
|---|---|---|---|
| **merken** | **96.4%** [0.948, 0.980] | raw, full loop | **Yes** — decider, recaller, audit all active |
| mempalace "raw" | 96.6% | ChromaDB only | **No** — issue #214 showed the benchmark only calls ChromaDB, no mempalace code |
| mempalace rooms | 89.4% | with palace features | Yes — 7pp below merken |
| mempalace AAAK | 84.2% | with compression | Yes — 12pp below merken |
| Mem0 | ~85% | hybrid + GPT-4 | Yes — LLM in path, higher cost per query |

merken is the only system in this table that (a) publishes a CI,
(b) runs its actual decision loop during the benchmark, and (c)
matches the raw-retrieval ceiling without an LLM.

## Honesty discipline

If a number we publish here turns out to be wrong, the fix is to add
a new row with the corrected number *and* leave the old row in place
with a strikethrough and a link to the correction. We do not silently
edit history. See `notes/prior-art.md` for the cautionary tale that
pinned this rule down.

The n <= 10 numbers above are **absolute positioning only** -- they
cannot support any claim of the form "merken matches X" or "merken
beats Y." Claims like that require n >= 50 with a non-degenerate CI
on the same `longmemeval_s_cleaned` split against the same metric.
Until such a row exists in this table, the claim does not get made
anywhere in the repo.

---

## Mode A answer-quality eval (not R@k)

Different question. The table above measures whether retrieval
surfaces a chunk from the answer session (R@k). `mode_a_eval.py`
adds a layer: **did we actually give the user a correct answer?**
That means running a Builder on each question, optionally with
retrieved chunks, and scoring the final answer against the ground
truth with an LLM-as-judge. See
`experiments/retrieval/longmemeval/mode_a_eval.py`.

Three conditions per question:

- **control** -- Builder alone (llama3.1-8b), confident-mode system
  prompt, no retrieval.
- **rag** -- Builder with top-5 retrieved chunks (3-way dual
  retrieval, same as Mode A) inlined into the user prompt. No
  Judge.
- **mode_a** -- full Mode A v4 pipeline (draft + dual-3 retrieval
  + claim-level Judge + deterministic annotator + provenance
  footer).

Oracle: Gemini 2.5 Flash scores `(question, ground_truth, answer)`
triples as `supports | partial | contradicts | neutral`. Correct =
`supports + partial`. Different model family from the
Builder/Judge on purpose, to avoid intra-family bias.

### Run 1 (2026-04-21) -- N=49 / seed 42 / longmemeval_s

Commit: `ccf6ce2` base (Mode A v4, PR #33 merged) + this branch's
eval harness. One question hit an `APIConnectionError` from
Cerebras and was logged as an error row rather than breaking the
loop (per-question try/except working as intended).

#### Headline

| condition | correct | supports | partial | contradicts | neutral | avg_tok | avg_s |
|---|---|---|---|---|---|---|---|
| control | **8.2%** (4/49) | 1 | 3 | 3 | 42 | 288 | 0.7 |
| rag | **71.4%** (35/49) | 30 | 5 | 9 | 5 | 2210 | 1.4 |
| mode_a | **71.4%** (35/49) | 33 | 2 | 7 | 7 | 8759 | 5.3 |

**RAG and Mode A tie on top-line correctness.** Mode A is NOT
better than naive RAG on these 49 questions. Mode A costs ~4x
the tokens and ~4x the wall time for equal correctness.

Mode A telemetry:
- **Grounded (verbatim `quoted_evidence` present): 40/49 = 81.6%.**
  This is the value Mode A buys over RAG -- every answer with a
  cite-able source trail.
- Claim-level leak (>=1 unsupported sub-claim): 5/49 = 10.2%.
- Sub-claims total: 125, of which 14 = 11.2% are unsupported.
- Judge top-level verdict distribution: contradicts 40 / no_claim
  5 / neutral 4. Mode A overrode the Builder draft in 82% of
  cases.

#### By question_type

| type | n | control | rag | mode_a | delta (mode_a - rag) |
|---|---|---|---|---|---|
| knowledge-update | 7 | 0% | 57% | **71%** | **+14pp** |
| multi-session | 17 | 12% | 65% | 65% | 0 |
| single-session-assistant | 3 | 0% | 100% | 100% | 0 |
| single-session-preference | 2 | 100% | 50% | 50% | 0 (n=2, noise) |
| single-session-user | 9 | 0% | 100% | 100% | 0 |
| temporal-reasoning | 11 | 0% | 64% | 55% | **-9pp** |

**Mode A wins on `knowledge-update`** (facts that evolve across
sessions, where the later assertion corrects the earlier one).
The Judge / claim-level rigor picks the right chunk instead of
averaging across contradictory ones.

**Mode A loses on `temporal-reasoning`** (questions that need
arithmetic or time-window reasoning). The Judge refuses when it
should have answered, or the Builder's arithmetic fails and Mode
A doesn't catch it.

Trivial lookups (`single-session-*`) are 100% for both -- they
are solved at the retrieval layer.

#### RAG vs Mode A disagreements (4 wins each; cases visible in log)

Mode A wins the cases that need **aggregation across multiple
chunks**:
- "money raised for charity total" (GT $3,750): RAG listed items,
  Mode A summed them.
- "rollercoasters across events July-October" (GT 10): RAG
  enumerated without totaling, Mode A aggregated.
- "engineers I lead (before / now)" (GT 4 / 5): RAG got cut off
  mid-sentence, Mode A produced a clean both-numbers answer.
- "book discount %" (GT 20%): RAG said "not mentioned", Mode A
  retrieved it correctly.

RAG wins the cases where **Mode A over-refuses or mis-aggregates**:
- "how many projects led" (GT 2): RAG guessed "at least one",
  Mode A refused entirely ("I don't have information...").
- "total $ from markets" (GT $495): Mode A summed but got $595.
  The Judge's arithmetic produced a wrong corrected_text.
- "sports event 2 weeks ago": RAG got partial credit, Mode A
  refused. Temporal reasoning weakness.

#### Caveats (code-reviewer pass before the run flagged these)

1. **Same dedup bug on both conditions.** `retrieve()` in this
   run used a 120-char prefix as dedup key. Conversational turns
   with identical prefixes (`"user: "`, `"assistant: "`) alias and
   drop distinct excerpts. Both RAG and Mode A hit this
   identically, so the A/B is still fair, but the absolute
   numbers under-represent what a bug-free retrieval could
   deliver. Fix landed on the same branch for Run 2.
2. **Control uses `confident` builder mode.** The Builder is told
   not to hedge. Refusals still dominate (42/49 = 86% control
   neutral) because the model genuinely lacks the personal
   information. But on the 7 non-refusal controls, the confident
   prompt pushes toward confabulation -- which the oracle scores
   as `contradicts`. Mostly this makes control look worse.
3. **RAG uses question-only query; Mode A uses question + draft
   query.** Two retrieval pools differ. Mode A's draft expansion
   can fetch chunks RAG would miss (knowledge-update advantage)
   OR noisy chunks RAG would skip.
4. **RAG truncates each excerpt to 800 chars** before inlining;
   Mode A's Judge sees the full excerpt text. If an answer lives
   past char 800 in a long chunk, RAG misses it. At N=49 this did
   not dominate the comparison but is a source of residual bias.

#### What this supports (conservative)

- Mode A at v4 **does not improve top-line correctness** over a
  naive inlined-context RAG baseline on LongMemEval_s at N=49.
- Mode A **does improve auditability**: 81.6% of answers carry
  a verbatim source quote; all answers carry the source_id and
  claim-level decomposition in the audit row.
- Mode A **has a real edge on `knowledge-update` questions**
  (+14pp over RAG) and a real weakness on `temporal-reasoning`
  (-9pp).

#### What this does NOT support

- That Mode A is production-ready as a drop-in replacement for
  RAG. Same correctness at 4x cost is a negative trade unless
  the auditability / grounded-source property is worth the
  premium.
- That the 4x token cost buys nothing -- it buys the grounded
  evidence trail, which is invisible in "did the oracle say
  supports?" but central to the "verifiable truth with source"
  thesis.
- That temporal-reasoning regressions are unfixable -- the
  Judge's arithmetic behavior and over-refusal rate are both
  tunable.

### Next moves (surfaced by this run)

- **Rerun with dedup fix (Run 2)**, same seed, same N. Measure
  whether both conditions move together (expected) or whether
  Mode A recovers ground vs RAG (less expected but possible).
- **Tune Judge for temporal questions.** The `-9pp` on
  temporal-reasoning comes mostly from over-refusal. A prompt
  that lets the Judge emit `contradicts` when arithmetic can
  be inferred from excerpts (rather than demanding a verbatim
  quote for the computed answer) would likely recover several
  points.
- **Token-cost levers.** 4x is steep. Cost profiling on the v1
  audit rows: excerpts dominate the Judge input. Score-threshold
  cutoff + hash-dedup (the latter already shipped) should cut
  ~20-30%.
- **Graduate to N=100 or N=500** once the cost and Judge tuning
  stabilise. Current N=49 gives a direction but CIs are wide.
