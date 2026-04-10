# LMEB results

## Scope

LMEB (Long-horizon Memory Embedding Benchmark) evaluates embedding
models across 22 datasets and 4 memory types: episodic, dialogue,
semantic, procedural. engram uses it as a retrieval benchmark at
real conversational scale — something the synthetic loop_quality
scenarios can't provide.

## First results: LoCoMo single_hop (2026-04-10)

| Date | Commit | Baseline | Dataset | Task | n | R@5 (95% CI) | Notes |
|------|--------|----------|---------|------|---|--------------|-------|
| 2026-04-10 | `9e8b061` | `vstash` | LoCoMo | single_hop | 840 | **0.573** [0.540, 0.606] | Raw substrate, no engram overhead |
| 2026-04-10 | `9e8b061` | `engram-heuristic` | LoCoMo | single_hop | 840 | **0.502** [0.470, 0.536] | Default decider + LayeredRecaller |

### Diagnosis: why engram loses ~7pp

Scene 0 isolation test (70 queries):

| Method | R@5 | What it tests |
|---|---|---|
| vstash raw | 0.486 (34/70) | Substrate alone |
| engram layer=episodic | 0.471 (33/70) | Engram bypass, episodic only |
| engram recall (layered) | 0.400 (28/70) | Default LayeredRecaller |

**Root cause: empty semantic layer in round-robin interleave.**

The `LayeredRecaller` always queries both semantic (top_k=5) and
episodic (top_k=3). On a pure retrieval benchmark with no
consolidation, the semantic layer returns 0 or very few hits. But the
round-robin interleave still gives the semantic layer slots in the
final top_k=5, displacing good episodic hits.

When we bypass the recaller with `layer='episodic'`, engram is within
1 hit of raw vstash (0.471 vs 0.486). The gap shrinks from ~7pp to
~1.5pp — the residual is likely the audit collection slightly
perturbing the embedder's batch normalization.

**Fix (not yet implemented):** the LayeredRecaller should skip layers
that return 0 results, or the interleave should give empty layers
no slots. This is a behavior change that must pass all 4 loop_quality
scenarios before landing.

## Honesty discipline

Same as the rest of the repo: no silent edits, corrections are
append-only, a result worse than expected is information not
embarrassment.
