# Loop-quality results

## Scope

This file records scenario-level metrics for engram's decision loop.
Every row is a run of the scenario runner against a pinned engram
commit. Unlike `experiments/retrieval/`, these numbers *are* about
whether engram's loop is adding value — there is no "absolute
positioning" disclaimer to hide behind.

## Metrics per scenario

- **query_pass_rate** — fraction of queries whose top-k semantic hits
  include a fact whose ground-truth topic matches the query's
  expected topic. Primary metric.
- **cluster_purity** — of all multi-event facts written, the fraction
  whose derived events all share the same topic. A mixed-topic fact
  is impure even if one of its members matches the query.
- **topic_coverage** — of all topics with ≥ 2 events, the fraction
  that produced at least one multi-event topic-pure fact. Measures
  whether the consolidator is finding the clusters that exist.

## Results

| Date | Commit | Scenario | Method | Threshold | Linkage | n_events | facts | pass_rate | purity | coverage | Notes |
|------|--------|----------|--------|-----------|---------|----------|-------|-----------|--------|----------|-------|
| 2026-04-09 | `4e6c7e0` | `session_2026_04_09` | `embedding_v1` | 0.65 | single | 12 | 3 | **25.00%** (1/4) | 66.67% | 33.33% | First honest run. Single-link transitive cascade via two cross-topic edges (longmemeval_a~dedup_fix_a=0.663, vstash_bug_a~dedup_fix_a=0.652) contaminates a 4-event impure cluster. Three same-topic pairs sit just below 0.65 (longmemeval=0.649, dedup_fix=0.575, loop_philosophy=0.558). |
| 2026-04-09 | `HEAD` | `session_2026_04_09` | `embedding_v1` | 0.65 | **complete** | 12 | 4 | **50.00%** (2/4) | 75.00% | 50.00% | Complete-link refuses to merge `{vstash_bug_a,b}` with `{longmemeval_a, dedup_fix_a}` because the weakest cross-pair (`vstash_bug_a~longmemeval_a=0.616`) is below 0.65. Three pure clusters emerge (mempalace, vstash_bug, consolidation_design) plus one impure mini-cluster (longmemeval_a+dedup_fix_a, the one edge that did cross). pass_rate doubles without changing the threshold. |

## Delta log

**Row 2 (complete-link, 2026-04-09):** `pass_rate 25% → 50%`,
`purity 67% → 75%`, `coverage 33% → 50%`. Same embedder, same
threshold, only the linkage strategy changed. Complete-link turned
out to be one line of logic (require min-cross-pair ≥ threshold
instead of any-cross-pair) but it fixed the specific cascade this
scenario exposed.

The ceiling that row 2 reveals is more important than the
improvement: **no threshold-based approach on this embedder can
push pass rate above ~50% on this scenario**. The reason is
visible in the pairwise cosine distribution around 0.65:

```
0.778  vstash_bug (same-topic)      ← above, clusters
0.717  mempalace  (same-topic)      ← above, clusters
0.663  longmemeval_a ~ dedup_fix_a  (CROSS-TOPIC)  ← above, false positive
0.661  consolidation_design (same)  ← above, clusters
0.652  vstash_bug_a ~ dedup_fix_a   (CROSS-TOPIC)  ← above, false positive
0.649  longmemeval (same-topic)     ← BELOW by 0.001, missed
0.616  longmemeval_a ~ vstash_bug_a (cross)        ← below
...
0.575  dedup_fix  (same-topic)      ← below, missed
0.558  loop_philosophy (same-topic) ← below, missed
```

Same-topic and cross-topic edges are interleaved through the band
`0.55 – 0.70`. `bge-small-en-v1.5` at 384 dimensions cannot
distinguish "engram internals about dedup" from "engram internals
about longmemeval benchmark" by cosine alone, because both land in
roughly the same region of the vector space.

**Implication for the next commits:**

- Lowering the threshold to 0.60 would catch longmemeval (0.649)
  and bring dedup_fix/loop_philosophy closer, but it also drags in
  more cross-topic false positives. Coverage up, purity down.
  Trade-off has to be measured, not argued.
- Raising the threshold to 0.70 kills the consolidation_design pair
  (0.661) and the false positives. Purity up, coverage down.
- A stricter linkage (e.g. k-medoids, or requiring min cluster
  density) can squeeze a few more points but cannot cross the
  signal/noise boundary that this embedder imposes.
- **The only path above ~50% on this scenario is an LLM-based
  consolidator that reads the text and distinguishes topics
  semantically, not by vector geometry.** Every non-LLM alternative
  is rearranging deck chairs in the 25–50% range.

That last bullet is the finding we commit to memory. The next
scenario to add to `loop_quality/scenarios/` should be content
where the embedder *does* cleanly separate topics — so we have a
control for "the loop is broken" vs "this scenario is beyond the
embedder."

## What the first row said (pre-complete-link)

**The engram loop at commit `be33c51` is not good enough yet.** On a
12-event scenario derived from engram's own design session, only
25% of queries route correctly to a topic-pure fact. Two failure
modes were diagnosed:

1. **Cross-topic false positives cascade via single-link**. Two
   pairs crossed the 0.65 threshold on genuine semantic similarity
   ("both are about engram internals") despite having different
   ground-truth topics. Single-link union-find cascades the
   contamination: `vstash_bug_a + dedup_fix_a + longmemeval_a`
   ended up in one 4-event cluster because each edge individually
   exceeded the threshold.
2. **Borderline same-topic pairs sit just under 0.65**. On this
   content domain (meta-discussion about engram itself),
   paraphrases of the same topic frequently land at cosine
   0.55–0.65 — below the v1 default. Coverage is penalized.

**What this rules out (or shouldn't yet):**

- It does NOT yet say engram is worse than raw vstash on this
  scenario — we haven't run the raw-vstash baseline for
  loop_quality yet. Scheduled as a follow-up.
- It does NOT say embedding_v1 is wrong. It says embedding_v1 with
  single-link clustering and threshold 0.65 is wrong *on
  meta-discussion content with semantically related but
  topic-distinct pairs*. A different scenario might show a
  different picture.

**What to try next (each in its own commit, each with a RESULTS.md row):**

1. Raise threshold to 0.70 — kills false positives, kills
   consolidation_design (0.661), kills longmemeval (0.649). Trade
   precision for recall.
2. Switch from single-link to average-link or complete-link. Single
   cross-topic edge no longer cascades through a whole cluster.
3. Per-scenario threshold calibration (advisory only — default
   stays a single value).
4. LLM-based consolidator that can distinguish topics by reading
   the sentences, not just counting vector proximity.

Each option above is a hypothesis. The right way to resolve them is
to run the runner again with the change and land a new row here. Do
not tune the scenario to make the current numbers look better.

## Honesty discipline

Same as the rest of the repo:

- No silent edits. Corrections add a new row and strike through the
  old one with a link.
- Scenario text is frozen on commit. Changing event text or query
  wording to make numbers go up is cheating — tune the *engram*
  code, not the benchmark.
- A result worse than the previous one is not embarrassing, it is
  information. We keep both rows and figure out what regressed.
