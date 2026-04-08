# experiments/

The empirical bar (CONSTITUTION §9). Every change to a default policy must
cite a benchmark in this directory. "I think it's better" does not ship.

## Layout

```
experiments/
├── README.md                   ← you are here
└── longmemeval/                ← LongMemEval R@5 anchor benchmark
    ├── README.md               ← reproduction guide
    ├── RESULTS.md              ← published numbers, with CIs and the policy
    │                            commit they were measured against
    └── runner.py               ← (TODO Phase 2) ingest + query loop
```

## Discipline

Same as vstash's `experiments/`:

1. Real datasets, not synthetic toys (LongMemEval, LoCoMo).
2. Reproducible on a laptop in under 5 minutes.
3. Numbers reported with confidence intervals, never point estimates.
4. The mode/policy/commit that produced the number is recorded next to it.
5. Mempalace's correction note is the playbook for honesty when something
   we published turns out to be wrong.

## What gets benchmarked

- **Phase 2 (gating):** raw vstash baseline vs engram-on-vstash vs mempalace
  on LongMemEval R@5. The result decides whether engram needs a taxonomy
  classifier in Phase 3 or whether the thin wrapper is already enough.
- **Phase 3:** with vs without `should_remember` + classification.
- **Phase 5:** with vs without consolidation. Consolidation only ships if
  it beats the Phase-3 number.
- **Phase 6:** procedural memory (separate task suite, TBD).

The runners are stubs in Phase 0. They get filled in when we hit Phase 2.
