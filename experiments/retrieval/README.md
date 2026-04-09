# retrieval/

Public retrieval benchmarks. These answer *"given a fixed haystack and
a fixed query, does the system surface the right chunk?"*

The answer lives mostly in vstash. Engram's loop barely participates,
because by the time a benchmark question is asked, everything has
already been ingested and the decision primitives (`should_remember`,
`should_consolidate`, etc.) have nothing more to do. That makes this
category useful for absolute positioning — *"we are in the same
ballpark as X"* — but a poor test of whether engram's loop adds value.

For the benchmark engram actually exists for, see
`../loop_quality/`.

## Subdirectories

- `longmemeval/` — public R@k benchmark, ~500 multi-session questions.
  Our first real-data runs live here.

## Adding a new benchmark

1. Create `experiments/retrieval/<name>/`.
2. Include a `README.md` with the exact reproduction command.
3. Include a `RESULTS.md` that starts empty. No row lands without a
   commit SHA and a CI.
4. If a result turns out to be wrong, strike through the row and add
   a corrected one below. Do not silently edit history.
