# LongMemEval results

## How to read this file

Every row records:

- **Date** — when the run was made.
- **Commit** — the engram commit SHA the run was made against.
- **Baseline** — `vstash` (substrate only) / `engram-always` (no policies) /
  `engram-heuristic` (Phase 1 default decider) / `mempalace` / `mem0` / `zep`.
- **Subset** — `longmemeval_s` (full distractor haystack) /
  `longmemeval_oracle` (oracle context only — sanity, not signal).
- **n** — number of questions evaluated. **Anything below ~50 is sanity, not signal.**
- **R@5** — recall @ 5 with 95% bootstrap CI (1000 iters, seed 0).
- **API/q** — API calls per query. Engram is local-first; should always be 0.
- **Notes** — what was different about this run, what we learned, what we'd
  change next time.

## Results

| Date | Commit | Baseline | Subset | n | R@5 (95% CI) | API/q | Notes |
|------|--------|----------|--------|---|--------------|-------|-------|
| 2026-04-08 | `2bcf502` | `engram-always` | `longmemeval_s` | 3 | 1.000 [1.000, 1.000] | 0 | First real-data run. **Sanity only — n=3 produces a degenerate CI.** Not comparable to mempalace's 96.6% on n=500. |
| 2026-04-08 | `2bcf502` | `engram-heuristic` | `longmemeval_s` | 3 | 1.000 [1.000, 1.000] | 0 | Same caveat. With no exact duplicates in the haystack, behaves identically to `engram-always`. |
| 2026-04-08 | `2bcf502` | `vstash` | `longmemeval_s` | 3 | 1.000 [1.000, 1.000] | 0 | Substrate-only baseline. Same caveat. |

### Wall-clock cost (informational, not part of the metric)

| Baseline | n | Elapsed | Per-question |
|---|---|---|---|
| `engram-always` | 3 | 92.0 s | ~30.7 s |
| `engram-heuristic` | 3 | 170.2 s | ~56.7 s |
| `vstash` | 3 | 129.2 s | ~43.1 s |

Run on a single Mac (CPU only — vstash uses local sentence-transformers).
~490 turns ingested per question median. Cost is dominated by the embedder
on this hardware.

**Extrapolation to the full 500-question benchmark, this hardware:**
~6 hours per baseline. CONSTITUTION §9's "under 5 minutes on a laptop"
target is **not currently met** for the full LongMemEval_s split. Two
options before the bar can claim to be honored: (a) use a faster local
embedder profile in vstash, (b) accept that LongMemEval needs an
overnight run and adjust §9's target language. Tracked in the engram
chat history; not yet in an issue.

## Findings from the first real-data run

1. **The pipeline works.** End-to-end ingest → recall → session-attribution
   → R@k → bootstrap CI matches the design. No schema surprises against
   the real LongMemEval-cleaned dataset.
2. **Heuristic exact-dedup adds nothing on LongMemEval.** Real haystacks
   have no exact duplicates, so `HeuristicWriteDecider`'s `dup_exact` rule
   never fires and engram-heuristic collapses to engram-always
   behaviorally. This rule's value will only show in *live agent loops*
   where the same content gets re-ingested, not in benchmark replay.
   **The benchmark cannot validate this rule. We need a different
   benchmark for it, or we need to admit it earns nothing.**
3. **Recall-based dedup was prohibitive.** The original
   `HeuristicWriteDecider` ran a vstash hybrid search on every write to
   detect duplicates. That made ingest O(N²) per haystack and the
   3-question run was killed mid-flight at ~25 minutes per baseline. We
   replaced it with an in-process `set[str]` of normalized text. Same
   semantics, hash-fast. The recall callable stays in `WriteContext` for
   future similarity-based deciders that genuinely need it.
4. **Audit log overhead is not free.** Every decision (write or skip)
   triggers an extra `vstash.remember` call to the audit collection. On
   3 questions × ~490 turns × 3 baselines, that's ~4400 extra writes
   beyond the user-facing ingest. Worth measuring on a bigger run before
   we decide whether to batch audit writes.
5. **Wall-clock variance between baselines is unexplained.** With one DB
   per (baseline, question), `engram-always` (92 s) < `vstash` (129 s) <
   `engram-heuristic` (170 s) is suspicious — the three baselines should
   be within ~10% of each other on identical hardware. Possibilities:
   model warmup spread across baselines, audit collection growth, or a
   per-DB cold-start cost. Not blocking, but a 10-question run with the
   same seed should help triangulate.

## Honesty discipline

If a number we publish here turns out to be wrong, the fix is to add a
new row with the corrected number *and* leave the old row in place with
a strikethrough and a link to the correction. We do not silently edit
history. See mempalace's April 2026 correction note for the model.

The n=3 numbers above are explicitly not a comparison to mempalace's
96.6%. A claim like "engram matches mempalace" requires n ≥ 50 with a
non-degenerate CI, on the same `longmemeval_s_cleaned` split, against
the same metric (R@5). Until that row exists in this table, do not make
that claim anywhere.
