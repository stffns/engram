# experiments/

The empirical bar (CONSTITUTION §9). Every change to a default policy
or a default decider must cite a benchmark in this directory. "I think
it's better" does not ship.

## Two categories

```
experiments/
├── README.md             ← you are here
├── retrieval/            ← does the substrate find what's there?
│   └── longmemeval/      ← public retrieval benchmark, one of several
└── loop_quality/         ← does engram's loop add value over the substrate?
                            (the benchmark we actually care about)
```

The split exists because they answer different questions.

**`retrieval/`** asks: *given a fixed haystack and a fixed query, does
the system surface the right chunk?* Public benchmarks like LongMemEval
live here. The answer mostly depends on vstash's chunker, embedder, and
hybrid weights — engram's loop barely participates.

**`loop_quality/`** asks: *given a stream of agent events over days,
does engram's decision loop produce a memory that is more useful than
raw vstash for the next thing the agent has to do?* This is the
benchmark engram actually exists for. It is a live, scenario-driven
test, not chat-replay.

If a row in `loop_quality/` shows engram beating raw vstash on a
scenario you'd actually live with, the loop is earning its keep. If a
row in `loop_quality/` shows them tied, the policy that was on trial
moves to `engram.policies.experimental` until a different scenario
revives it.

## Discipline (same as vstash's `experiments/`)

1. Real datasets, not synthetic toys — except in `loop_quality/`,
   where carefully designed synthetic scenarios are exactly the point.
2. Reproducible. Each subdirectory has a `README.md` with the exact
   command to reproduce its `RESULTS.md`.
3. Numbers reported with confidence intervals, never point estimates.
4. The mode/policy/commit that produced the number is recorded next
   to it.
5. **No silent edits.** If a published number turns out to be wrong,
   the row stays with a strikethrough and a link to the correction.
   See `notes/prior-art.md` for the cautionary tale that taught us
   this rule.

## What gets benchmarked, and when

| Phase | Benchmark | Question it answers |
|---|---|---|
| 1 (now) | `retrieval/longmemeval` (sample) | does the runner work end-to-end on real data? |
| 2 | `loop_quality/scenario_basic` | does `should_remember` filter useful events on a real-shaped agent stream? |
| 3 | `loop_quality/scenario_consolidation` | does `consolidate` produce facts that are findable later? |
| 4 | `retrieval/longmemeval` (full) | how does engram-on-vstash sit on a public bench, in absolute terms? *Optional, gated on hardware allowing it.* |

The `retrieval/longmemeval` full run is *optional* on purpose. If we
publish it, we publish it honestly. If our hardware can't run it in a
sane window, we say so and skip it rather than cherry-picking samples.
