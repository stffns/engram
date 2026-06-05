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

## Current state (as of 2026-06-05)

**Test suite:** 398 passed, 3 pre-existing failures (all torch-import
failures in shadow classifier env tests — `test_env_primary_overrides_shadow`,
`test_nanogpt_env_missing_paths_falls_back_cleanly`,
`test_default_returns_plain_when_shadow_import_fails`), 1 skipped
(torch classifier), 3 warnings (fastembed/huggingface). Torch is a
soft dependency — the package is fully usable without it; the failing
tests only fire when `MERKEN_SHADOW=nanogpt` or `MERKEN_PRIMARY=nanogpt`
is set without torch installed.

**Write-filter classifier status (merged into develop):**

- **nanoGPT v7** is the graduated shadow baseline. Clears the
  markdown-tables blind spot (FPR 66.7% v6 → 0% v7) while keeping 100%
  recall on organic_val and jay_vstash. Runs as `MERKEN_SHADOW=nanogpt`;
  `MERKEN_PRIMARY` flip still deferred. Full version history in
  `experiments/nanogpt/RESULTS.md`.
- **LLM classifier backend** (`MERKEN_SHADOW=llm`, `MERKEN_PRIMARY=llm`)
  uses any HuggingFace model. Supports calibration via shipped
  `calibration_v7.json` or custom path. Backend-agnostic `_shadow.py`
  module handles both nanoGPT and LLM decider roles.
- **Calibration head** supports both backends: wraps predictions with
  a post-hoc Platt-calibrated `P_cal` via shipped
  `merken/classifiers/calibration_v7.json`. Load errors degrade
  gracefully (warning to stderr, raw P(D) shown).

**Role classifier (merged via PR #43/#48):**

- **`merken/role_classifier.py`** — generic role classifier that
  leverages a frozen encoder from any merken classifier and assigns
  events to a user-defined taxonomy of roles.
- **Production profile** (`role_prototypes_v2.json`) operationalises
  the #39b STATE_CHANGE_REPORT taxonomy — STRONG gate met with 4-role
  classification (DECISION, ENTITY, EVENT, FREE).
- **Decoupled from global ROLES** — supports arbitrary taxonomies.
- **Role markers** (#39 Phase 1-3): prefix-conditioning found
  ineffective; `STATE_CHANGE_REPORT` taxonomy (Phase 2) passed STRONG.

**Role geometry / directional residuals (merged via PR #41):**

- **Issue #38/#38b** — NO_GO verdict on EVR_1 (event-role-vector) as a
  write-gate. Validated on disjoint data; performed worse than plain
  heuristic gates.
- **Signal mining post-NO_GO** — partial recovery via directional
  residual features. #38c deferred.
- **Paper meta-issue** in `notes/` captures the geometric-methods
  argument for the merken paper (§7.3/§7.6/§7.7 expansion).

**vstash 0.35.0 drift fix (merged via PR #45):**

- Repaired 18 failing tests caused by upstream vstash API drift.
- Single commit: `bcc3375 fix(memory): vstash 0.35.0 API drift`.

**Training-data pipeline:** 1026 real oracled labels in
`data/merken_labels_v7.jsonl` (gitignored), produced by the
bootstrap scripts under `experiments/`. Pipeline is idempotent.

**Phase 2 distillation (experimental, PR #49 open with changes requested):**

- Steps 1-4 complete end-to-end: gpt-oss-120b teacher on Cerebras,
  Qwen2.5-7B-Instruct LoRA student.
- Key result: student v2 (temperature=0.2 + repetition_penalty=1.1)
  matches zero-shot truncation rate (18%) at **-90% reasoning chars**
  (1004c vs 9614c median).
- Step 5 planned: 1K → 50K corpus, 3 epochs, lr 1e-4 → 1e-5.

**Heartbeat daemon (PR #50 open with changes requested):**

- `merken heartbeat` subcommand — periodic background consolidation
  and forgetting on configurable intervals.
- Writes `should_heartbeat` audit rows on every tick.
- Graceful shutdown via SIGINT/SIGTERM.

**Training-data pipeline:** 1026 real oracled labels in
`data/merken_labels_v7.jsonl` (gitignored), produced by the
bootstrap scripts under `experiments/` (PR #12). Pipeline is
idempotent — re-runs are safe.

## Decision primitives

All four decision primitives from CONSTITUTION §5.1 are implemented:

- **`should_remember`** — `HeuristicWriteDecider` (default) and
  `AlwaysWrite` (baseline). Optional `ChainedWriteDecider` to compose
  with a classifier shadow/primary layer. The heuristic decider
  hydrates its dedup set from vstash on first use.
- **`should_consolidate`** — `PeriodicConsolidator` (default) and
  `NeverConsolidate`. Uses `cluster_by_embedding` with complete
  linkage and threshold 0.70.
- **`should_recall`** — `LayeredRecaller` (default, semantic-first
  with episodic fallback) and `SemanticOnlyRecaller` baseline.
  Round-robin interleave across layers with dedup-by-path.
  Optional temporal reranking via `temporal_weight` (default 0.0).
- **`should_forget`** — `NeverForget` (default, safe) and
  `ForgetConsolidated`. Tombstone-not-delete: full text preserved
  in `merken_tombstones`, reversible.

Deployed surfaces:

- **Python SDK** — `from merken import Memory`. Four primitives
  accessible as `Memory` methods.
- **CLI** — `merken` on `$PATH` after `pip install -e .`.
  Nine subcommands: `remember | recall | consolidate | forget |
  audit | tombstones | status | stats | heartbeat`.
  See `merken --help`.
- **MCP server** — `merken-mcp` on `$PATH`, `python -m merken.mcp_server`,
  or `claude mcp add merken -- python -m merken.mcp_server`. Eight
  tools, one per CLI subcommand (heartbeat excluded — daemon, not tool).
- **Claude Code hooks** — live in `~/.claude/settings.json`.
  Three hooks: `SessionStart` (recall context), `PreCompact`
  (save to memory), `UserPromptSubmit` (search memory).
  Scripts at `~/.claude/hooks/merken-*.sh`.

Safety net (`experiments/loop_quality/`):

- Three core scenarios, all running under `pytest tests/test_loop_quality.py`
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

## Consolidation findings

**Embedding-based consolidation (embedding_v1) is structurally limited.**
Cosine(q, e(Ti)) >= Cosine(q, e(T_consolidated)) for specific queries.
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

## Tracked concern: Benchmark monoculture (Issue #44)

All recent retrieval/builder decisions (k=8, brief_v1 reject, v5-ft
Gates, Mode C, gpt-oss-120b unstick) came from the same benchmark
axis: **LongMemEval + LoCoMo**. Risk of overfit to that signal pair.

**Protocol (from Issue #44):** Before the next round of retrieval/builder
tuning that would land a new default in production, evaluate on at least
one disjoint benchmark axis (candidates: MTEB retrieval subset, BEIR
subset, MS-MARCO). Pre-register the disjoint benchmark before any new
tuning experiment. Ship iff the change does not regress on the disjoint
benchmark by more than 1pp.

Reference this issue when planning any new retrieval/builder experiment.

## What's NOT next

- A fifth decision primitive. Four are enough.
- Embedding-based consolidation as a retrieval improvement -- proven
  structurally limited (see consolidation findings above).
- Knowledge graph (CONSTITUTION §6, optional, gated).
- Shell completion, colors, a web UI. All noise for the scope.
- Bespoke compression dialect. See `notes/prior-art.md` for why.
- Spatial vocabulary (wings/rooms/etc.). Use vstash's existing
  `project` / `collection` / `layer` / `tags` fields.

## What IS next (approximately)

- **Phase 2 distillation Step 5:** Scale to 50K corpus, 3 epochs,
  lr 1e-5. Estimated $200-400 (teacher generation + training + eval).
- **Heartbeat daemon (PR #50):** Address coderabbitai review comments
  and merge. Wires periodic maintenance into production usage.
- **brief_v1 integration into Claude Code hooks.** Generate briefs
  on PreCompact, prepend on SessionStart alongside recall results.
- **nanoGPT-as-connector.** Small model trained on vstash+merken
  data structure for topic identification and brief selection.
- **Scale brief_v1 to 100+ topics** for paper, diverse domains.
- **Additional `loop_quality/` scenarios** from real work (perf
  migration notes, MedLocal hackathon logs, Kafka meeting threads).
- **LoCoMo runner** under `experiments/retrieval/locomo/` (Phase B
  of BENCHMARK_STRATEGY.md).
- **Disjoint benchmark evaluation** (Issue #44 gate) before next
  retrieval/builder default change.

## Branching

`feature/*` → `develop` → `main` via release PR. Mirrors vstash.
