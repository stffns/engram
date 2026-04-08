# LongMemEval anchor benchmark

LongMemEval is the public benchmark mempalace uses to claim 96.6% R@5 in raw
mode. Engram uses the same dataset and metric so its numbers are directly
comparable to mempalace, Mem0, Zep, and any other system that publishes against
LongMemEval.

## Status

**Phase 0 — skeleton only.** No runner, no results. The directory exists so
that the empirical bar is a first-class citizen from commit 1, *before* any
default policy lands. This prevents the mempalace failure mode of shipping
features and benchmarking afterward.

The runner is built in **Phase 2** of `ULTRA_PLAN` (see chat history). Phase 2
is gated on Phase 1 (decision primitives) being in place, so the order is:

1. Phase 1 — `should_remember` and friends
2. Phase 2 — runner.py here, plus three baselines (B0 vstash, B1 engram raw,
   B2 mempalace if installable)
3. Publish numbers in `RESULTS.md` with confidence intervals

## Reproduction (planned)

```bash
# from the engram repo root
python -m experiments.longmemeval.runner \
    --baseline {vstash,engram,mempalace} \
    --questions 500 \
    --top-k 5
```

Expected runtime: under 5 minutes on an M-series laptop, zero API calls.

## Dataset

LongMemEval is publicly available; the runner downloads it on first use into a
gitignored cache directory. We do not vendor the dataset.

## See also

- `experiments/README.md` — empirical bar discipline
- `../../CONSTITUTION.md` §9 — the "no benchmark, no ship" rule
