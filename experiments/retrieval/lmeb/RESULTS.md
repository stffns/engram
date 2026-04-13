# LMEB results

## Scope

LMEB (Long-horizon Memory Embedding Benchmark) evaluates embedding
models across 22 datasets and 4 memory types: episodic, dialogue,
semantic, procedural. merken uses it as a retrieval benchmark at
real conversational scale — something the synthetic loop_quality
scenarios can't provide.

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
