# CLAUDE.md — merken

This file is the entry point for any Claude session opened in the
merken repo. It should be cheap to read and keep a current Claude
session aligned with the state of the repo.

## Read these first, in order

1. [`CONSTITUTION.md`](CONSTITUTION.md) — what merken is, what it
   isn't, and the principles. Disagreements get edits to that file
   *before* any code.
2. The vstash repo's `CLAUDE.md` — merken is a strict consumer of
   vstash's public API. Understand the substrate before touching
   the loop.
3. [`experiments/BENCHMARK_STRATEGY.md`](experiments/BENCHMARK_STRATEGY.md)
   — how merken measures itself, which benchmarks are load-bearing
   and which are noise.
4. [`notes/silt.md`](notes/silt.md) — working notes on the patterns
   Silt caught that became hard rules.

## Current state (as of 2026-05-30)

**Heartbeat daemon (PR #50):** New `merken heartbeat` CLI subcommand and
`Heartbeat` class that runs `Memory.consolidate()` and `Memory.forget()`
on configurable schedules. Blocks until SIGINT/SIGTERM. Writes
`should_heartbeat` audit rows every tick (glass-box principle). Usage:
`merken heartbeat --consolidate-interval 300 --forget-interval 3600`.
Exported from `merken/heartbeat.py`.

**Phase 2 distillation (PR #49):** Steps 1-4 complete (1K SFT smoke,
distillation lesson transfers from GPT 120B → Qwen 3.5-9B). Key findings:
MLX-TQ3 not viable at 9B; zero-shot smoke results show qwen3.5-9B GGUF
with LM Studio works; step 4d LoRA trainer + Colab notebook shipped.
Training corpus: 1000 SFT examples. MLX fuse + convert pipeline works.

**Write-filter classifier status:** nanoGPT **v7** is the graduated
shadow baseline. First version to clear the markdown-tables blind
spot (FPR 66.7% v6 -> 0% v7) while keeping 100% recall on
organic_val and jay_vstash. 3/5 graduation criteria pass cleanly,
1 borderline (94.6% oracle agreement on 205-item subsample), 1 N/A.
`MERKEN_PRIMARY` flip still deferred; v7 runs as
`MERKEN_SHADOW=nanogpt`. Full version history and open-frontier
spec (v9 / v10) in `experiments/nanogpt/RESULTS.md`.

**Training-data pipeline:** 1026 real oracled labels in
`data/merken_labels_v7.jsonl` (gitignored), produced by the
bootstrap scripts under `experiments/` (PR #12). Pipeline is
idempotent -- re-runs are safe.

**Hook bug fixed (2026-04-17):** `~/.claude/hooks/merken-save.sh`
used to run `json.load` on JSONL transcripts and silently swallow
the exception. Result: zero shadow events accumulated. Fixed to
parse JSONL per-line + extract `message.content[].text`. Future
accumulation is now passive.

## Decision primitives (as of 2026-04-09)

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
  dedup-by-path. Optional temporal reranking via `temporal_weight`
  (default 0.0 = off). Grid search across 5 scenarios at 8 weights
  showed zero regressions.
- **`should_forget`** — `NeverForget` (default, safe) and
  `ForgetConsolidated`. Tombstone-not-delete: full text preserved
  in `merken_tombstones`, reversible.

Deployed surfaces:

- **Python SDK** — `from merken import Memory`. Four primitives
  accessible as `Memory` methods.
- **CLI** — `merken` on `$PATH` after `pip install -e .`.
  Nine subcommands map 1:1 to `Memory` methods:
  `remember | recall | consolidate | forget | audit | tombstones | status | stats | heartbeat`.
  See `merken --help`.
- **MCP server** — `merken-mcp` on `$PATH`, `python -m merken.mcp_server`,
  or `claude mcp add merken -- python -m merken.mcp_server`. Eight
  tools, one per CLI subcommand. Default DB is `~/.merken/<project>.db`,
  deliberately isolated from `~/.vstash/memory.db`.
- **Claude Code hooks** — live in `~/.claude/settings.json`.
  Three hooks: `SessionStart` (recall context), `PreCompact`
  (save to memory), `UserPromptSubmit` (search memory).
  Scripts at `~/.claude/hooks/merken-*.sh`.

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
  to the `merken_audit` collection. No exceptions — even skipped
  writes and never-forgotten events produce audit trails. Tombstoned
  events additionally write to `merken_tombstones` so they're
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

## Consolidation findings (2026-04-16)

**Embedding-based consolidation (embedding_v1) is structurally limited.**
Cosine(q, e(Ti)) >= cosine(q, e(T_consolidated)) for specific queries.
Geometric limit, confirmed across LoCoMo E2E + 3 knowledge_update
scenarios. LLM synthesis on top of embedding clustering does not help --
the bottleneck is clustering, not materialization.

**brief_v1 works.** LLM-generated temporal briefs with typed schemas
(DECISION/ENTITY/EVENT/FREE) and dedicated brief-layer search:
- 50 topics, 1100 events: 86% vs 40% retrieval-only (+46pp)
- 20 topics, 440 events: 100% vs 45% (direct inject, +55pp)
- Cost: ~300 tokens prepended per query (brief_k=3)

Architecture: `Memory.consolidate(method="brief_v1", synthesize_fn=fn)`
generates briefs. `Memory.recall_with_briefs()` searches brief layer
separately from episodic, prepends matched briefs to context.

See `experiments/consolidation/RESULTS.md` for full analysis.

## What's NOT next

- A fifth decision primitive. Four are enough.
- Embedding-based consolidation as a retrieval improvement -- proven
  structurally limited (see consolidation findings above).
- A knowledge graph (CONSTITUTION $6, optional, gated).
- Shell completion, colors, a web UI. All noise for the scope.

## What IS next (approximately)

- **Heartbeat CLI polish + PR #50 merge.** Address remaining code review
  feedback, merge to develop, then main.
- **brief_v1 integration into Claude Code hooks.** Generate briefs
  on PreCompact, prepend on SessionStart alongside recall results.
- **nanoGPT-as-connector.** Small model trained on vstash+merken
  data structure for topic identification and brief selection.
- **Phase 2 distillation — Step 5+.** Multi-turn training, eval on
  LongMemEval, production quantization (MLX).
- Scale brief_v1 to 100+ topics, diverse domains for paper.
- Additional `loop_quality/` scenarios from Jay's real work.
- vstash 0.29.0 validation (snapvec integration).
- **Issue #44 — Benchmark monoculture.** LongMemEval + LoCoMo are the
  only axes for retrieval / builder decisions. Need a third axis.

## Branching

`feature/*` → `develop` → `main` via release PR. Mirrors vstash.
