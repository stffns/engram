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

| Date | Commit | Scenario | Method | Threshold | n_events | facts | pass_rate | purity | coverage | Notes |
|------|--------|----------|--------|-----------|----------|-------|-----------|--------|----------|-------|
| 2026-04-09 | `be33c51` | `session_2026_04_09` | `embedding_v1` | 0.65 | 12 | 3 | **25.00%** (1/4) | 66.67% | 33.33% | First honest run. Single-link transitive cascade via two cross-topic edges (longmemeval_a~dedup_fix_a=0.663, vstash_bug_a~dedup_fix_a=0.652) contaminates a 4-event impure cluster. Three same-topic pairs sit just below 0.65 and don't cluster (longmemeval=0.649, dedup_fix=0.575, loop_philosophy=0.558). |

## What the first row says

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
