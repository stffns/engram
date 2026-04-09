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
| 2026-04-09 | `f83115e` | `session_2026_04_09` | `embedding_v1` | 0.65 | complete | 12 | 4 | **50.00%** (2/4) | 75.00% | 50.00% | Complete-link refuses to merge `{vstash_bug_a,b}` with `{longmemeval_a, dedup_fix_a}` because the weakest cross-pair (`vstash_bug_a~longmemeval_a=0.616`) is below 0.65. Three pure clusters emerge (mempalace, vstash_bug, consolidation_design) plus one impure mini-cluster (longmemeval_a+dedup_fix_a, the one edge that did cross). pass_rate doubles without changing the threshold. |
| 2026-04-09 | `8ff6953` | `analytics_project` | `embedding_v1` | 0.65 | complete | 12 | 6 | **100.00%** (4/4) | 100.00% | 100.00% | **Control scenario.** Six lexically distinct topics (auth/database/deploy/frontend/monitoring/billing). Pre-measured pairwise cosines: same-topic pairs [0.778, 0.937] median 0.915; cross-topic pairs [0.320, 0.617] median 0.465. Clean 0.162 gap — any threshold in (0.617, 0.778) gives 100%. This is the "loop works when the embedder cooperates" reference point. Future consolidator changes must not drop this below 100% without naming the trade-off. Recall path: explicit `layer="semantic"`. |
| 2026-04-09 | `HEAD` | `session_2026_04_09` | `embedding_v1` | 0.65 | complete | 12 | 4 | **100.00%** (4/4) | 75.00% | 50.00% | **`should_recall` enabled.** Same consolidation as row 2 (purity/coverage unchanged). The lift to 100% comes from `LayeredRecaller` falling back to the episodic layer when semantic doesn't have a topic-pure fact. `dedup_fix` and `longmemeval` queries now match their raw episodic events directly instead of failing for lack of a pure fact. The loop is now robust to imperfect consolidation — an impure cluster no longer takes down the query. |
| 2026-04-09 | `7e85566` | `analytics_project` | `embedding_v1` | 0.65 | complete | 12 | 6 | **100.00%** (4/4) | 100.00% | 100.00% | `should_recall` enabled. No regression: the scenario that was already at 100% stays at 100%. This is the "did we break anything" check; it passed. |
| 2026-04-09 | `HEAD` | `jay_vstash_2026_04_09_snapshot` | `embedding_v1` | 0.65 | complete | **20** | 4 | **75.00%** (3/4) | 75.00% | 60.00% | **First real-content row.** 20 organic docs from Jay's vstash frozen as a fixture, topic labels assigned by honest reading of titles (6 topics). Three queries pass: `kafka_meeting` (singleton via interleave fix), `engram_design` (singleton episodic), `medlocal_clinical` (pure 4-event cluster). One fails: `vstash_notes` — the real failure mode is that Fact 4 mixes 3 vstash_notes events with 1 engram_design event (agent-memory-use-cases-2026-04-07 has overlapping vocabulary), so `_fact_path_to_topic` excludes it from the pure lookup. The fact's anchor text literally contains "vstash Upstream Improvement Ideas" — a human would call it a pass, but the strict purity check correctly rejects an impure cluster. This is the test answering "what happens on real content nobody curated." |

## Three-scenario picture (post real-snapshot)

As of the real-vstash snapshot commit, the metric table has three
rows that are all running under the same runner, in the same
`pytest tests/` invocation, on the same commit:

|                          | analytics_project | session_2026_04_09 | jay_vstash_2026_04_09_snapshot |
|---|---|---|---|
| Type                     | synthetic control | synthetic borderline | real organic content |
| Events                   | 12                | 12                  | 20                  |
| Topics                   | 6 distinct        | 6 overlapping       | 6 (real distribution) |
| Same-topic median cosine | 0.915             | 0.649               | (not measured yet)  |
| facts_written            | 6                 | 4                   | 4                   |
| **pass_rate**            | **100%**          | **100%**            | **75%**             |
| cluster_purity           | 100%              | 75%                 | 75%                 |
| topic_coverage           | 100%              | 50%                 | 60%                 |

The three scenarios stress the loop from three angles:

- **analytics_project** — the embedder cooperates. This is the
  "nothing is broken" control. Any regression here is a disaster.
- **session_2026_04_09** — the embedder disagrees with ground
  truth. Tests how well the loop survives overlapping vocabulary.
  cluster_purity and topic_coverage are stuck but query routing
  still succeeds via the interleave fallback.
- **jay_vstash_2026_04_09_snapshot** — the real thing. Content
  the user produced organically, topic labels assigned by honest
  title-reading. The 75% is the first number that measures
  engram's loop on content nobody tuned for it.

The gap between "curated scenarios pass rate" (100%) and
"real-content pass rate" (75%) is the number to watch over time.
If it closes, the loop is improving. If it widens, the curated
scenarios are drifting away from what real content looks like and
need new siblings.

## Real-content smoke (2026-04-09, post should_recall)

After the scenario runner hit 100% on both curated fixtures, we ran
a **qualitative** smoke test against the user's real vstash — 20
recent docs that neither I nor the user curated for engram. Output
in `experiments/loop_quality/smoke_real_vstash.py`; this is not a
scenario runner because we have no ground-truth topic labels for
organic content.

**Consolidation on real content was actually good.** 20 events
produced 4 coherent clusters plus 2 honest singletons:

```
Fact 1 (n=4):  MedLocal clinical demos (Meningococcemia, Neonato,
               Motrin, Organofosforados) — pure cluster
Fact 2 (n=7):  MedLocal architecture/strategy (CHT, Loop, Competitive,
               Decision Tables, Retrieval Engineering, Pipeline,
               Estado EOD) — pure cluster
Fact 3 (n=3):  Daily Reviews (Teams×2 + Mail) — pure cluster
Fact 4 (n=4):  vstash meta notes (Non-Obvious, Upstream, Debug,
               agent-memory use cases) — mostly pure, agent-memory
               is arguable
Singletons:    engram v0.1 decisions, Kafka Merchant Pipeline meeting
               — both honestly unique in this slice
```

**Recall surfaced a real bug in Memory.recall, caught the fix, and
now shows both strengths and limits:**

- BEFORE the fix, the query "what happened in the Kafka merchant
  pipeline meeting?" never returned the Kafka singleton. Memory.recall
  drained layers sequentially: semantic returned 4 facts, filled the
  top_k=3 budget, and episodic was never visited. The "fallback"
  was a fallback only in name.
- AFTER the fix (round-robin interleave), the Kafka note surfaces
  at rank 2, and the MedLocal benchmark query surfaces the "98/99"
  EOD number at rank 2 — real episodic evidence that was previously
  invisible.
- BUT broad thematic queries ("what vstash bugs were found?", "what
  are the engram architecture decisions?") still rank a big
  MedLocal cluster at the top because the fact's anchor text
  contains vstash vocabulary and everything in the MedLocal cluster
  mentions vstash or engram in passing. The semantic layer is
  dense enough that broad queries flood toward large clusters
  regardless of topic specificity.

**What the smoke says about engram as of this commit:**

- Consolidation works on real content. No fixture-lying by
  construction.
- Layered recall + interleave works for specific queries
  (singletons, entity names, specific numbers).
- Layered recall fails the discrimination test for broad thematic
  queries on dense content. `bge-small-en-v1.5` at 384 dimensions
  cannot separate "vstash bug" from "MedLocal case that mentions
  vstash" when both appear in clusters with high cosine density.
- The honest next improvement is **query-type awareness in
  `should_recall`** — route entity/specific queries episodic-first
  and theme queries semantic-first. Not a new decider primitive,
  just a smarter default.

The smoke script is idempotent and safe: it opens the user's vstash
read-only and writes to a throwaway tempdir. Re-run any time with
`python -m experiments.loop_quality.smoke_real_vstash --n-docs 20`.

## What the `should_recall` rows say

`session_2026_04_09` went from 50% → 100% **without any change to
consolidation**. The trick was not fixing the clusters — it was
accepting that consolidation is imperfect and routing around the
imperfection.

`LayeredRecaller` (semantic first, episodic fallback):

```
Query: "why was the dedup rule rewritten to use an in-process set?"
  → semantic search: top hit is the impure 2-event cluster
    {longmemeval_a, dedup_fix_a} (wrong topic label)
  → fallback: episodic search returns dedup_fix_a and dedup_fix_b
    directly — raw events with the correct topic tag
  → runner's unified lookup accepts the episodic match
  → pass ✓
```

The consolidation metrics on the session scenario are unchanged:

- `cluster_purity: 75%` (one of four facts still mixes topics)
- `topic_coverage: 50%` (only 3 of 6 eligible topics produced a
  topic-pure multi-event fact)

These are real failures of the v1 consolidator, and they stay
visible in the results table. But `query_pass_rate` — the metric
that matches what a user actually experiences — is now 100%
because the loop can use the raw stream as evidence when the
distilled layer is broken.

**Important framing:** this is not "we fixed the 50% ceiling." The
50% consolidation ceiling is still real; it just stopped being a
user-facing problem because the loop has a second path to the
answer. An LLM-based consolidator would lift cluster_purity and
topic_coverage, and that has its own value (cheaper recall,
higher-quality facts, compression of episodic over time). But for
query routing on this scenario, the fallback is enough.

The next change that touches the consolidator needs to justify its
cost against what the current loop already does: 100% pass rate on
both scenarios via fallback. Not just against the 50% purity
number in isolation.

## The two-scenario picture

As of row 3 the repo has two scenarios and they say very different things:

|                    | session_2026_04_09 | analytics_project |
|---|---|---|
| Content            | meta-discussion about engram itself | realistic agent stream, six distinct topics |
| Same-topic cosine (median) | 0.649                | 0.915                |
| Cross-topic cosine (max)   | 0.663                | 0.617                |
| Gap                | **-0.014** (overlap) | **+0.162** (clean)   |
| pass_rate          | 50%                  | 100%                 |

**What this rules in:**

- The engram consolidation pipeline is *not* broken. On content where
  the embedder can separate topics, it hits 100% on every metric.
- Complete-link clustering is the right default. It does not cost the
  control scenario anything (still 100%) and it fixes the session
  scenario cascade.

**What this rules out:**

- The 50% on `session_2026_04_09` is not an indictment of engram. It
  is a statement about `bge-small-en-v1.5` on meta-discussion
  content. Any change to the consolidator that "fixes" the 50%
  without also holding `analytics_project` at 100% is tuning to
  the failing test, not improving the loop.

**What this sets up:**

- Every future decider change gets measured against *both*
  scenarios. A change that moves session up and analytics down is a
  trade-off that must be named, not declared an improvement.
- When we decide whether to add an LLM consolidator or a
  cross-encoder reranker, the bar is clear: **move session up
  without moving analytics down**. Anything else is a sideways step.

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
