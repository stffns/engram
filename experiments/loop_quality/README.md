# loop_quality/

The benchmark engram actually exists for. Answers *"does the decision
loop add value over raw vstash on a scenario that looks like what an
agent actually lives through?"*

## Why this directory exists

Twice during engram's build we wrote tests that passed by
construction — contrived near-duplicate strings that let flawed
clustering pass. The "prueba del vaso" on real session snippets
destroyed the first `jaccard_v1` implementation in under a minute.
This directory codifies that lesson: **every new decider gets run
against a real-shaped scenario before it's declared working**.

A scenario is:

1. A list of events with text and a topic label (ground truth).
2. A list of queries with an expected topic (or a text pattern that
   must appear in the answer).

The runner ingests the events into a fresh `Memory`, runs
`consolidate()`, and then asks each query. A query passes if the
top-k hits from `recall(layer="semantic")` include a fact whose
provenance points back at an event with the expected topic.

The primary metric is **query pass rate**. Secondary metrics include
cluster purity (do facts only contain events from one topic?) and
topic coverage (did every topic with ≥ 2 events produce a fact?).

## Layout

```
loop_quality/
├── README.md               ← you are here
├── RESULTS.md              ← published numbers per scenario + commit
├── scenario.py             ← Scenario dataclass + JSON loader
├── runner.py               ← CLI: run all scenarios, print a report
└── scenarios/
    └── session_2026_04_09.json   ← 12 snippets from engram's own
                                    design session. Six topics, two
                                    events each, four real queries.
```

## Adding a scenario

1. Create `scenarios/<name>.json` matching the schema in
   `scenario.py` (fields: `name`, `description`, `events` with
   `id`/`text`/`topic`, `queries` with `question`/`expect_topic`).
2. Run `python -m experiments.loop_quality.runner` to see where it
   sits against the current engram state.
3. If the scenario reveals a regression, add a row to `RESULTS.md`
   with the commit SHA and a note on what broke.

## Honesty discipline

Same as retrieval/ and the repo-wide rule:

- Every row in `RESULTS.md` records the commit SHA it was run against.
- No silent edits. If a number turns out to be wrong, add a new row
  with the correction; leave the original with a strikethrough.
- A scenario that contains a known weakness (e.g. two events whose
  cosine is right on the threshold) should say so in its
  `description`. Tests should not lie by construction, but
  scenarios are allowed to contain borderline cases — we just call
  them out.
