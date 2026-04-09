# CLAUDE.md — engram

This file is the entry point for any Claude session opened in the
engram repo. It should be cheap to read and keep a current Claude
session aligned with the state of the repo.

## Read these first, in order

1. [`CONSTITUTION.md`](CONSTITUTION.md) — what engram is, what it
   isn't, and the principles. Disagreements get edits to that file
   *before* any code.
2. The vstash repo's `CLAUDE.md` — engram is a strict consumer of
   vstash's public API. Understand the substrate before touching
   the loop.
3. [`experiments/BENCHMARK_STRATEGY.md`](experiments/BENCHMARK_STRATEGY.md)
   — how engram measures itself, which benchmarks are load-bearing
   and which are noise.
4. [`notes/silt.md`](notes/silt.md) — working notes on the patterns
   Silt caught that became hard rules.

## Current state (as of 2026-04-09)

All four decision primitives from CONSTITUTION §5.1 are implemented:

- **`should_remember`** — `HeuristicWriteDecider` (default) and
  `AlwaysWrite` (baseline). The heuristic decider now hydrates its
  dedup set from vstash on first use, so cross-invocation dedup
  works for CLI and MCP.
- **`should_consolidate`** — `PeriodicConsolidator` (default) and
  `NeverConsolidate`. Uses `cluster_by_embedding` with complete
  linkage and threshold 0.70 (picked via grid search on three
  loop_quality scenarios, 2026-04-09).
- **`should_recall`** — `LayeredRecaller` (default, semantic-first
  with episodic fallback) and `SemanticOnlyRecaller` baseline.
  `Memory.recall` does round-robin interleave across layers with
  dedup-by-path.
- **`should_forget`** — `NeverForget` (default, safe) and
  `ForgetConsolidated`. Tombstone-not-delete: full text preserved
  in `engram_tombstones`, reversible.

Deployed surfaces:

- **Python SDK** — `from engram import Memory`. Four primitives
  accessible as `Memory` methods.
- **CLI** — `engram` on `$PATH` after `pip install -e .`.
  Eight subcommands map 1:1 to `Memory` methods:
  `remember | recall | consolidate | forget | audit | tombstones | status | stats`.
  See `engram --help`.
- **MCP server** — `engram-mcp` on `$PATH`, `python -m engram.mcp_server`,
  or `claude mcp add engram -- python -m engram.mcp_server`. Eight
  tools, one per CLI subcommand. Default DB is `~/.engram/<project>.db`,
  deliberately isolated from `~/.vstash/memory.db`.
- **Claude Code hooks** — NOT YET. Future slice.

Safety net (`experiments/loop_quality/`):

- Three scenarios, all running under `pytest tests/test_loop_quality.py`
  via parametrized `test_runner_completes_on_every_scenario`:
    1. `analytics_project` (synthetic control, 100%/100%/100%)
    2. `session_2026_04_09` (synthetic borderline, 100%/100%/33%)
    3. `jay_vstash_2026_04_09_snapshot` (real organic, 100%/100%/80%)
- A new decider or policy change that drops any scenario below its
  current pass_rate is a regression. Investigate before merging.

## Hard rules

These survive across sessions. Breaking any of them requires an
explicit case in the PR description.

- **vstash is a hard dependency.** Never reach into
  `vstash._private`. Never read SQLite tables directly *in production
  code* (probes in `notes/` or one-off investigations are fine).
  If the public API is missing something, the fix is a vstash PR.
  (CONSTITUTION §4.4, §6.)
- **Glass box.** Every decision the loop makes writes an audit row
  to the `engram_audit` collection. No exceptions — even skipped
  writes and never-forgotten events produce audit trails. Tombstoned
  events additionally write to `engram_tombstones` so they're
  recoverable.
- **No new vector storage.** No ChromaDB, no second store, no FTS
  reimplementation. (CONSTITUTION §3, §8.)
- **No bespoke compression dialect.** Measure with a real tokenizer
  before claiming any compression result. See `notes/prior-art.md`
  for the cautionary tale.
- **Empirical first.** Every default-policy change cites a benchmark
  in `experiments/`. "I think it's better" does not ship.
  (CONSTITUTION §9, operationalized in `experiments/BENCHMARK_STRATEGY.md`.)
- **Silt's rule: "before proposing an algorithm, look at the
  distribution of the data."** Every time we reached for a new
  decider or a smarter policy without first measuring the
  distribution we were working against, we overengineered and had
  to retract later. When a number looks bad, the first move is to
  grid-search the knobs you already have against the bar you
  already built. Only after that exhausts the simple moves is a
  new algorithm justified. See `notes/silt.md` for the specific
  interventions this rule survives.
- **Test fixtures are not ground truth.** Two separate sessions,
  tests passed green while the real behavior on organic content
  was broken. Every new decider gets run against at least one
  real-content scenario (the `jay_vstash_*_snapshot` family) before
  landing. Synthetic tests are necessary but not sufficient.
- **Cross-model test independence.** Test text pairs used in
  consolidation tests must cluster above threshold in both
  `BAAI/bge-small-en-v1.5` AND
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
  vstash 0.27.0 can resolve to either depending on the environment.
  See `tests/test_mcp_server.py::test_consolidate_force_builds_fact`
  for the pattern.

## What's NOT next

- A fifth decision primitive. Four are enough.
- LLM-based consolidation in the hot path. Gated on a scenario
  where the non-LLM loop leaves real value on the table.
- A knowledge graph (CONSTITUTION §6, optional, gated).
- Shell completion, colors, a web UI. All noise for the scope.

## What IS next (approximately)

- Phase A of `experiments/BENCHMARK_STRATEGY.md` — the overnight
  LongMemEval full n=500 run, script at
  `experiments/retrieval/longmemeval/run_overnight.sh`.
- Claude Code hooks. Depends on MCP server being stable (it is)
  and on the user's `settings.json` design preference.
- Additional `loop_quality/` scenarios from Jay's real work:
  perf migration notes, MedLocal hackathon logs, Kafka meeting
  threads, daily reviews.
- LoCoMo runner under `experiments/retrieval/locomo/`. Needs a
  judge model choice first (open question in BENCHMARK_STRATEGY.md).

## Branching

`feature/*` → `develop` → `main` via release PR. Mirrors vstash.
