# LongMemEval results

## Scope

This file records absolute R@5 numbers on LongMemEval. **It does not
measure whether engram's loop adds value** — LongMemEval is chat-replay
with no duplicates in the haystack, so the decision primitives collapse
to "ingest everything." For the benchmark that tries to catch loop
value, see `../../loop_quality/`.

## How to read this file

Every row records:

- **Date** — when the run was made.
- **Commit** — the engram commit SHA the run was made against.
- **Baseline** — `vstash` (substrate only) / `engram-always` (no
  filtering) / `engram-heuristic` (Phase 1 default decider).
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
| 2026-04-08 | `2bcf502` | `engram-always` | `longmemeval_s` | 3 | 1.000 [1.000, 1.000] | 0 | First real-data run. **Sanity only — n=3 produces a degenerate CI.** |
| 2026-04-08 | `2bcf502` | `engram-heuristic` | `longmemeval_s` | 3 | 1.000 [1.000, 1.000] | 0 | Same caveat. With no exact duplicates in the haystack, behaves identically to `engram-always`. |
| 2026-04-08 | `2bcf502` | `vstash` | `longmemeval_s` | 3 | 1.000 [1.000, 1.000] | 0 | Substrate-only baseline. Same caveat. |
| 2026-04-08 | `e18d7d4` | `engram-heuristic` | `longmemeval_s` | 10 | **0.900** [0.700, 1.000] | 0 | First non-degenerate CI. 9/10 hits. seed=42. |
| 2026-04-08 | `e18d7d4` | `vstash` | `longmemeval_s` | 10 | **0.900** [0.700, 1.000] | 0 | Identical hit set to engram-heuristic — confirms the heuristic decider is a no-op on this dataset. seed=42. |

### Wall-clock cost (informational, not part of the metric)

| Baseline | n | Elapsed | Per-question |
|---|---|---|---|
| `engram-always` | 3 | 92.0 s | ~30.7 s |
| `engram-heuristic` | 3 | 170.2 s | ~56.7 s |
| `vstash` | 3 | 129.2 s | ~43.1 s |
| `engram-heuristic` | 10 | 405.5 s | ~40.6 s |
| `vstash` | 10 | 355.8 s | ~35.6 s |

The 10-question pair stabilizes the wall-clock picture: engram costs
~14% more than raw vstash (40.6 vs 35.6 s/question). That overhead is
the audit log — every decision triggers an extra `vstash.remember`
into the `engram_audit` collection. Worth measuring whether batching
audit writes recovers most of it before §9 amends its targets.

Run on a single Mac (CPU only — vstash uses local sentence-transformers).
~490 turns ingested per question median. Cost is dominated by the
embedder on this hardware.

**Full 500-question extrapolation, this hardware:** ~6 hours per
baseline. CONSTITUTION §9's "under 5 minutes on a laptop" target is
**not currently met** for the full LongMemEval_s split. Two options:
(a) use a faster local embedder profile in vstash, (b) accept that
LongMemEval needs an overnight run and amend §9's language. Tracked
in the chat history; not yet in an issue.

## What the n=10 run actually says

- **engram-heuristic and vstash are tied at 9/10 (R@5 = 0.900).** The
  CI is [0.700, 1.000] — informative but wide.
- **Heuristic exact-dedup contributes zero on this dataset.** Same hit
  set as raw vstash. The decider is not adding signal here. It is
  also not subtracting — but on a benchmark that cannot validate the
  rule's intended use case (re-ingestion in a live loop), it's not
  earning its keep either. This is exactly why `loop_quality/` exists.
- **The 1-in-10 miss is interesting but not yet investigated.** The
  per-question results aren't dumped by the CLI — adding a
  `--dump json` flag is a small follow-up so we can look at *which*
  question we miss and *why*.

## Findings from the first real-data runs

1. **The pipeline works.** End-to-end ingest → recall →
   session-attribution → R@k → bootstrap CI matches the design. No
   schema surprises against the real LongMemEval-cleaned dataset.
2. **Heuristic exact-dedup adds nothing on LongMemEval.** Real
   haystacks have no exact duplicates, so `HeuristicWriteDecider`'s
   `dup_exact` rule never fires and engram-heuristic collapses to
   engram-always behaviorally. This rule's value can only show in
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
   DB per (baseline, question), `engram-always` (92 s) < `vstash`
   (129 s) < `engram-heuristic` (170 s) is suspicious — the three
   baselines should be within ~10% of each other on identical
   hardware. Possibilities: model warmup spread across baselines,
   audit collection growth, or a per-DB cold-start cost.

## Honesty discipline

If a number we publish here turns out to be wrong, the fix is to add
a new row with the corrected number *and* leave the old row in place
with a strikethrough and a link to the correction. We do not silently
edit history. See `notes/prior-art.md` for the cautionary tale that
pinned this rule down.

The n ≤ 10 numbers above are **absolute positioning only** — they
cannot support any claim of the form "engram matches X" or "engram
beats Y." Claims like that require n ≥ 50 with a non-degenerate CI
on the same `longmemeval_s_cleaned` split against the same metric.
Until such a row exists in this table, the claim does not get made
anywhere in the repo.
