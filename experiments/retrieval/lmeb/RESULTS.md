# LMEB results

## Scope

LMEB (Long-horizon Memory Embedding Benchmark) evaluates embedding
models across 22 datasets and 4 memory types: episodic, dialogue,
semantic, procedural. merken uses it as a retrieval benchmark at
real conversational scale — something the synthetic loop_quality
scenarios can't provide.

## REALTALK full results (2026-04-15)

10 scenes × 3 tasks, n=679 queries, local CPU, ~12 min.

| Date | Commit | Baseline | Task | n | R@5 (95% CI) | Notes |
|------|--------|----------|------|---|--------------|-------|
| 2026-04-15 | `44371d5` | `vstash` | commonsense | 103 | **0.379** [0.282, 0.476] | |
| 2026-04-15 | `44371d5` | `merken-heuristic` | commonsense | 103 | **0.369** [0.282, 0.466] | -1.0pp, CIs overlap |
| 2026-04-15 | `44371d5` | `vstash` | multi_hop | 265 | **0.389** [0.328, 0.449] | |
| 2026-04-15 | `44371d5` | `merken-heuristic` | multi_hop | 265 | **0.404** [0.347, 0.464] | +1.5pp, CIs overlap |
| 2026-04-15 | `44371d5` | `vstash` | temporal_reasoning | 311 | **0.704** [0.653, 0.752] | |
| 2026-04-15 | `44371d5` | `merken-heuristic` | temporal_reasoning | 311 | **0.695** [0.643, 0.746] | -0.9pp, CIs overlap |

### Pattern

Unlike LoCoMo, REALTALK shows no significant gap between merken and
raw vstash — all three tasks have overlapping CIs. merken even edges
ahead on multi_hop. This is consistent with the LoCoMo diagnosis:
the LayeredRecaller penalty depends on how much the empty semantic
layer displaces good episodic hits, which varies by query
distribution. REALTALK queries appear to be a better fit.

## TMD full results (2026-04-15)

12 scenes × 12 tasks, n=2134 queries, local CPU, ~85 min.

| Date | Commit | Baseline | Task | n | R@5 (95% CI) | Notes |
|------|--------|----------|------|---|--------------|-------|
| 2026-04-15 | `44371d5` | `vstash` | content_time_qs | 177 | 0.305 [0.237, 0.373] | |
| 2026-04-15 | `44371d5` | `merken-heuristic` | content_time_qs | 177 | 0.305 [0.237, 0.373] | tie |
| 2026-04-15 | `44371d5` | `vstash` | date_span_time_qs | 180 | 0.444 [0.372, 0.511] | |
| 2026-04-15 | `44371d5` | `merken-heuristic` | date_span_time_qs | 180 | 0.428 [0.356, 0.494] | -1.6pp, CIs overlap |
| 2026-04-15 | `44371d5` | `vstash` | dates_time_qs | 330 | 0.170 [0.127, 0.212] | |
| 2026-04-15 | `44371d5` | `merken-heuristic` | dates_time_qs | 330 | 0.173 [0.133, 0.215] | +0.3pp |
| 2026-04-15 | `44371d5` | `vstash` | day_span_time_qs | 24 | 0.167 [0.042, 0.333] | |
| 2026-04-15 | `44371d5` | `merken-heuristic` | day_span_time_qs | 24 | 0.125 [0.000, 0.250] | n too small |
| 2026-04-15 | `44371d5` | `vstash` | earlier_today_time_qs | 12 | 0.167 [0.000, 0.417] | |
| 2026-04-15 | `44371d5` | `merken-heuristic` | earlier_today_time_qs | 12 | 0.083 [0.000, 0.250] | n too small |
| 2026-04-15 | `44371d5` | `vstash` | last_named_day_time_qs | 12 | 0.250 [0.083, 0.500] | |
| 2026-04-15 | `44371d5` | `merken-heuristic` | last_named_day_time_qs | 12 | 0.250 [0.083, 0.500] | tie |
| 2026-04-15 | `44371d5` | `vstash` | month_time_qs | 100 | 0.470 [0.380, 0.570] | |
| 2026-04-15 | `44371d5` | `merken-heuristic` | month_time_qs | 100 | 0.400 [0.300, 0.500] | -7.0pp, CIs overlap |
| 2026-04-15 | `44371d5` | `vstash` | rel_day_time_qs | 317 | 0.167 [0.129, 0.211] | |
| 2026-04-15 | `44371d5` | `merken-heuristic` | rel_day_time_qs | 317 | 0.164 [0.126, 0.208] | -0.3pp |
| 2026-04-15 | `44371d5` | `vstash` | rel_month_time_qs | 100 | 0.420 [0.320, 0.520] | |
| 2026-04-15 | `44371d5` | `merken-heuristic` | rel_month_time_qs | 100 | 0.420 [0.320, 0.520] | tie |
| 2026-04-15 | `44371d5` | `vstash` | rel_session_time_qs | 330 | 0.170 [0.130, 0.212] | |
| 2026-04-15 | `44371d5` | `merken-heuristic` | rel_session_time_qs | 330 | 0.161 [0.121, 0.203] | -0.9pp |
| 2026-04-15 | `44371d5` | `vstash` | session_span_time_qs | 258 | 0.391 [0.329, 0.453] | |
| 2026-04-15 | `44371d5` | `merken-heuristic` | session_span_time_qs | 258 | 0.380 [0.322, 0.438] | -1.1pp |
| 2026-04-15 | `44371d5` | `vstash` | session_time_qs | 294 | 0.180 [0.139, 0.224] | |
| 2026-04-15 | `44371d5` | `merken-heuristic` | session_time_qs | 294 | 0.170 [0.129, 0.214] | -1.0pp |

### Pattern

**Weighted mean across all TMD tasks:** vstash 0.258, merken 0.248
(Δ = -1.0pp). All task-level CIs overlap between baselines. TMD is a
low-ceiling dataset for dense retrieval (best R@5 ~0.47) because
temporal disambiguation is genuinely hard without structured
reasoning.

Combined with REALTALK (also statistical tie), this is **two
consecutive LMEB datasets where merken does not show the LoCoMo
regression**. The -7pp LoCoMo gap appears specific to LoCoMo query
distributions (single_hop/multi_hop with abundant displaceable
episodic hits), not a general pattern.

## DeepPlanning shopping_level1 (2026-04-15)

50 scenes, n=50 queries, local CPU, ~31 min.

| Date | Commit | Baseline | Task | n | R@5 (95% CI) | Notes |
|------|--------|----------|------|---|--------------|-------|
| 2026-04-15 | `44371d5` | `vstash` | shopping_level1 | 50 | **0.000** [0.000, 0.000] | |
| 2026-04-15 | `44371d5` | `merken-heuristic` | shopping_level1 | 50 | **0.000** [0.000, 0.000] | same |

### Diagnosis: ceiling, not regression

Both baselines score zero. Queries are multi-constraint natural-language
specs ("Nike in orange with fewer than 10 one-star reviews and more
than 300 four-star reviews"). Corpus docs are JSON product records
with structured fields (brand, color, rating distribution). Dense
embeddings cannot bridge this — the query's predicates don't lexically
or semantically align with the fields they reference.

This is a dataset/embedder mismatch, not a merken behavior finding.
DeepPlanning probably needs a structured-retrieval adapter (hybrid
filter + rerank) before it can yield signal. Skip levels 2/3 until
then.

### Runner fix (2026-04-15)

Added fallback to `_scene_from_query_id`: if `_q_` is absent and
`qid` matches a known scene_id, use it directly. Unblocks
DeepPlanning-style datasets where `query.id == scene_id`.

## LoCoMo full results (2026-04-10)

| Date | Commit | Baseline | Task | n | R@5 (95% CI) | Notes |
|------|--------|----------|------|---|--------------|-------|
| 2026-04-10 | `79303a9` | `vstash` | single_hop | 840 | **0.573** [0.540, 0.606] | |
| 2026-04-10 | `79303a9` | `merken-heuristic` | single_hop | 840 | **0.502** [0.470, 0.536] | -7.1pp |
| 2026-04-10 | `79303a9` | `vstash` | multi_hop | 280 | **0.464** [0.404, 0.521] | |
| 2026-04-10 | `79303a9` | `merken-heuristic` | multi_hop | 280 | **0.368** [0.314, 0.429] | -9.6pp |
| 2026-04-10 | `79303a9` | `vstash` | temporal_reasoning | 321 | **0.626** [0.576, 0.676] | |
| 2026-04-10 | `79303a9` | `merken-heuristic` | temporal_reasoning | 321 | **0.589** [0.536, 0.642] | -3.7pp, CIs overlap |
| 2026-04-10 | `79303a9` | `vstash` | adversarial | 446 | **0.439** [0.395, 0.484] | |
| 2026-04-10 | `79303a9` | `merken-heuristic` | adversarial | 446 | **0.350** [0.307, 0.395] | -8.9pp |
| 2026-04-10 | `79303a9` | `vstash` | open_domain | 89 | **0.326** [0.236, 0.427] | |
| 2026-04-10 | `79303a9` | `merken-heuristic` | open_domain | 89 | **0.281** [0.191, 0.371] | -4.5pp, CIs overlap |

### Pattern across categories

merken loses 4-10pp to vstash raw on every LoCoMo category. The gap
is largest on multi_hop (-9.6pp) and adversarial (-8.9pp) — the
hardest categories where each displaced hit matters most. temporal_
reasoning and open_domain have overlapping CIs, so the gap may not
be significant there.

### Diagnosis: why merken loses ~7pp

Scene 0 isolation test (70 queries):

| Method | R@5 | What it tests |
|---|---|---|
| vstash raw | 0.486 (34/70) | Substrate alone |
| merken layer=episodic | 0.471 (33/70) | Engram bypass, episodic only |
| merken recall (layered) | 0.400 (28/70) | Default LayeredRecaller |

**Root cause: empty semantic layer in round-robin interleave.**

The `LayeredRecaller` always queries both semantic (top_k=5) and
episodic (top_k=3). On a pure retrieval benchmark with no
consolidation, the semantic layer returns 0 or very few hits. But the
round-robin interleave still gives the semantic layer slots in the
final top_k=5, displacing good episodic hits.

When we bypass the recaller with `layer='episodic'`, merken is within
1 hit of raw vstash (0.471 vs 0.486). The gap shrinks from ~7pp to
~1.5pp — the residual is likely the audit collection slightly
perturbing the embedder's batch normalization.

**Fix (not yet implemented):** the LayeredRecaller should skip layers
that return 0 results, or the interleave should give empty layers
no slots. This is a behavior change that must pass all 4 loop_quality
scenarios before landing.

## Post-fix results: budget redistribution (2026-04-10)

Commit `42d40ef` — when a layer returns 0 hits, its top_k budget
carries forward to the next layer. Two lines changed in memory.py.

| Date | Commit | Baseline | Task | n | R@5 (95% CI) | Δ vs pre-fix |
|------|--------|----------|------|---|--------------|-------------|
| 2026-04-10 | `42d40ef` | `merken-heuristic` | single_hop | 840 | **0.568** [0.537, 0.601] | +0.066 |
| 2026-04-10 | `42d40ef` | `merken-heuristic` | multi_hop | 280 | **0.450** [0.393, 0.507] | +0.082 |
| 2026-04-10 | `42d40ef` | `merken-heuristic` | temporal_reasoning | 321 | **0.620** [0.570, 0.670] | +0.031 |
| 2026-04-10 | `42d40ef` | `merken-heuristic` | adversarial | 446 | **0.433** [0.388, 0.478] | +0.083 |
| 2026-04-10 | `42d40ef` | `merken-heuristic` | open_domain | 89 | **0.303** [0.213, 0.393] | +0.022 |

### Before/after summary

| Task | vstash | merken (before) | merken (after) | Gap closed |
|------|--------|-----------------|----------------|------------|
| single_hop | 0.573 | 0.502 (-7.1pp) | **0.568** (-0.5pp) | 93% |
| multi_hop | 0.464 | 0.368 (-9.6pp) | **0.450** (-1.4pp) | 85% |
| temporal | 0.626 | 0.589 (-3.7pp) | **0.620** (-0.6pp) | 84% |
| adversarial | 0.439 | 0.350 (-8.9pp) | **0.433** (-0.6pp) | 93% |
| open_domain | 0.326 | 0.281 (-4.5pp) | **0.303** (-2.3pp) | 49% |

All CIs now overlap with vstash. Average gap: ~6.8pp → ~1.1pp.
The fix recovered >80% of the lost recall on 4 of 5 categories.

## Honesty discipline

Same as the rest of the repo: no silent edits, corrections are
append-only, a result worse than expected is information not
embarrassment.
