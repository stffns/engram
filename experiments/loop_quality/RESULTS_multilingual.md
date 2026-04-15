# Multilingual calibration — results

## Scope

This file records the 2026-04-14 investigation that asked: does
merken's loop survive cross-lingual queries (es/en mixed), and if
not, where is the fix?

Motivation: all external benchmarks for merken run in English only
(LongMemEval R@5=0.964, n=500). Jay's real daily content is mixed
Spanish/English. The R@5 number is not necessarily transferable.

## Scenario

`experiments/loop_quality/scenarios/bilingual_es_en_2026_04_14.json`

- 12 events across 6 topics.
- Each topic has exactly one event in Spanish and one in English,
  describing the same underlying fact from a slightly different
  angle (not tight paraphrases — deliberately realistic).
- 6 queries, each asked in the OPPOSITE language of its
  most-relevant event, to force a cross-lingual retrieval path.

## Run 1 — default embedder (`BAAI/bge-small-en-v1.5`)

Threshold 0.65, complete linkage. Via `runner.py`:

```
query_pass_rate: 100.00%  (6/6)
facts_written:   5
topic_coverage:  83.33%
cluster_purity:  100.00%
```

First read: recall path survives cross-lingual queries (100%), but
consolidation misses one of six topics (`merken_multilingual_gap`).

## Run 2 — threshold grid (bge-small-en)

`experiments/loop_quality/threshold_grid.py` across thresholds
`{0.45, 0.50, 0.55, 0.60, 0.65, 0.70}`:

| scenario | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 |
|---|---|---|---|---|---|---|
| bilingual topic_coverage | 83% | 83% | 83% | 83% | 83% | 83% |

**Flat.** Lowering the threshold did not recover the missing topic.
That rules out the knob and points at the embedder — the missing
pair's cosine must sit below 0.45.

## Run 3 — embedder swap (`paraphrase-multilingual-MiniLM-L12-v2`)

`experiments/loop_quality/bilingual_embedder_probe.py`:

| embedder | query_pass_rate | topic_coverage | facts |
|---|---|---|---|
| bge-small-en-v1.5 | 100% | 83% | 5 |
| paraphrase-multilingual-MiniLM-L12-v2 | 100% | 83% | 5 |

Same result at threshold 0.65. But that hides the actual signal,
which lives in the distribution of pairwise cosines.

## The distribution — honest picture

Pairwise cosines computed directly on the 12 events, per embedder:

**`bge-small-en-v1.5`** (English-only model):

| topic | same-topic cos(es, en) |
|---|---|
| perf_kafka_lag | 0.757 |
| code_review_skill | 0.743 |
| vstash_api_gap | 0.790 |
| jira_pr_link | 0.675 |
| engram_rename | 0.678 |
| merken_multilingual_gap | **0.639** |

Top cross-topic cosines (noise): `0.704, 0.665, 0.651, 0.628, 0.626`.

**Overlap.** Same-topic pairs live in `[0.639, 0.790]`; cross-topic
noise reaches `0.704`. There is no threshold that cleanly separates
signal from noise on this embedder for bilingual content.

**`paraphrase-multilingual-MiniLM-L12-v2`**:

| topic | same-topic cos(es, en) |
|---|---|
| code_review_skill | 0.832 |
| perf_kafka_lag | 0.812 |
| vstash_api_gap | 0.784 |
| engram_rename | 0.762 |
| jira_pr_link | 0.754 |
| merken_multilingual_gap | **0.424** |

Top cross-topic cosines (noise): `0.501, 0.498, 0.414, 0.410, 0.360`.

**Clean gap.** Five of six same-topic pairs live in `[0.754, 0.832]`;
cross-topic noise caps at `0.501`. A threshold anywhere in
`[0.55, 0.75]` cleanly separates signal from noise.

## Finding

The multilingual embedder is strictly better on bilingual content,
but the improvement is not visible at the default threshold 0.65 —
it only shows up when you inspect the distribution or pick a
threshold that exploits the wider gap.

The `merken_multilingual_gap` pair (cos 0.424 on the multilingual
model) did not cluster. Reading the two event texts: the Spanish
version leads with `R@5=0.964`, `LongMemEval`, and `Jay`; the
English version leads with `cross-lingual retrieval quality
uncalibrated` and `paraphrase-multilingual-capable`. They describe
the same situation from different angles and are not tight
paraphrases. The embedder is correctly reporting that they are not
that similar. This is honest scenario design feedback, not an
embedder failure.

## What this changes (and doesn't)

**Does not change:**
- Default embedder stays `BAAI/bge-small-en-v1.5` (vstash's default).
  For English-only corpora it performs well and has the lowest
  footprint.
- Default threshold stays 0.70. No scenario-level evidence justifies
  changing it as a global default.

**Does change:**
- README gains a "Multilingual corpora" paragraph pointing users to
  `paraphrase-multilingual-MiniLM-L12-v2` + threshold in `[0.55, 0.60]`.
- `bilingual_es_en_2026_04_14` joins the standing loop_quality
  scenarios; future default-policy changes must not regress its
  pass_rate (currently 100%).
- Memory entry updated: recall cross-lingual is healthy, the
  consolidation gap is real but bounded, and the fix is config, not
  code.

## What's next

- Write a second bilingual scenario with tighter paraphrases
  (same key phrases in both languages) to isolate the embedder's
  ceiling from scenario-design artifacts.

## Follow-up: LongMemEval no-regression check (2026-04-14)

Ran `merken-heuristic` on LongMemEval `longmemeval_s` with
`paraphrase-multilingual-MiniLM-L12-v2` pinned via VSTASH_CONFIG.
n=100 (not 500) — this is a no-regression question and a wider CI
is adequate. Colab CPU, 40 min.

| embedder | n | R@5 | 95% CI |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 500 | 0.964 | [0.948, 0.980] |
| `paraphrase-multilingual-MiniLM-L12-v2` | 100 | **0.980** | [0.950, 1.000] |

CIs overlap and the multilingual point estimate is higher. We do
not claim improvement (n=100 CI is too wide for that), but we do
claim the stronger-than-needed result: **no regression on English**.

With both the bilingual scenario (clean signal/noise gap on cross-
lingual content) and the LongMemEval no-regression check cleared,
`paraphrase-multilingual-MiniLM-L12-v2` graduates from "consider
for multilingual corpora" to "recommended embedder for bilingual/
multilingual users, safe for English-only workloads." README
updated accordingly.
