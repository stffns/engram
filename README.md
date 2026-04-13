# merken

> Agent-loop layer for persistent memory, built on top of
> [vstash](https://github.com/stffns/vstash).

**Status:** pre-v0.1, local-first, 154 tests green.
Four decision primitives, three deployment surfaces, three loop-quality
scenarios all at 100% pass rate and 100% cluster purity.

## In one paragraph

vstash is a **glass-box retrieval substrate** — SQLite + sqlite-vec +
FTS5 + reciprocal rank fusion, with observability and explicit limits.
merken is the **loop on top**: when to write a memory, when to recall,
when to distill raw events into semantic facts, when to tombstone the
ones that are redundant. vstash stores and searches; merken reasons
about *what* is worth storing and searching. Every decision is logged
to an audit collection you can query.

```
       ┌──────────────────────────────────────────┐
       │  your agent (Claude Code, your code, …)  │
       └────────────────────┬─────────────────────┘
                            │ remember / recall / consolidate / forget
                            ▼
       ┌──────────────────────────────────────────┐
       │                   merken                  │
       │                                           │
       │  ┌────────────┐  ┌─────────────────────┐ │
       │  │ Decision   │  │ Memory              │ │
       │  │ primitives │◄─┤  .remember()        │ │
       │  │            │  │  .recall()          │ │
       │  │ should_    │  │  .consolidate()     │ │
       │  │  remember  │  │  .forget()          │ │
       │  │  recall    │  │  .audit()           │ │
       │  │  consoli-  │  │  .tombstones()      │ │
       │  │  date      │  │                     │ │
       │  │  forget    │  │  Context manager    │ │
       │  └──────┬─────┘  └─────────┬───────────┘ │
       │         │                   │             │
       │         └─────────┬─────────┘             │
       │                   ▼                       │
       │          merken_audit collection          │
       │          merken_tombstones collection     │
       └────────────────────┬─────────────────────┘
                            │ every storage + search call
                            ▼
       ┌──────────────────────────────────────────┐
       │  vstash (substrate — glass box)           │
       │  sqlite-vec + FTS5 + RRF + MMR dedup     │
       │  metrics, limits, integrity, contracts    │
       └──────────────────────────────────────────┘
```

## The three deployment surfaces

merken ships as one library with three ways to call it. They all wrap
the same `Memory` class.

### 1. Python SDK

```python
from merken import Memory, ForgetConsolidated, PeriodicConsolidator

with Memory(
    project="my_agent",
    consolidate_decider=PeriodicConsolidator(min_events=10),
    forget_decider=ForgetConsolidated(),
) as mem:
    mem.remember("the user switched to Postgres on 2026-04-08")
    mem.remember("the analytics warehouse now runs on Postgres 16")

    result = mem.consolidate()
    print(f"{result.facts_written} fact(s) from {result.events_examined} events")

    hits = mem.recall("what database does the analytics warehouse use?")
    for h in hits[:3]:
        print(f"  • {h.text}")

    forget = mem.forget()
    print(f"{len(forget.tombstoned)} events tombstoned")
```

Deeper SDK docs: [`docs/primitives.md`](docs/primitives.md),
[`docs/extending.md`](docs/extending.md).

### 2. CLI (after `pip install -e .`)

```bash
merken remember "the user switched to Postgres on 2026-04-08"
merken recall "what database does the analytics warehouse use?"
merken consolidate
merken forget --decider consolidated
merken audit should_remember
merken tombstones
merken status
merken stats
```

Human output by default; pass `--json` anywhere for pipeable output.
Every command accepts `--project NAME` (or `ENGRAM_PROJECT` env var) and
`--db PATH` (default `~/.merken/<project>.db`, deliberately separate
from `~/.vstash/memory.db`).

Deeper CLI reference: [`docs/cli.md`](docs/cli.md).

### 3. MCP server — use merken from Claude Code

```bash
# attach merken to Claude Code as an MCP server
claude mcp add merken -- merken-mcp
```

Then, inside any Claude Code session:

> *"Claude, remember that we switched to Postgres on April 8th, 2026."*
> *"Claude, what did we decide about the analytics warehouse database?"*
> *"Claude, consolidate what we've discussed."*

Eight tools, one per CLI command: `merken_remember`, `merken_recall`,
`merken_consolidate`, `merken_forget`, `merken_audit`,
`merken_tombstones`, `merken_status`, `merken_stats`. Config via
environment: `ENGRAM_PROJECT`, `ENGRAM_DB`.

Deeper MCP reference: [`docs/mcp-server.md`](docs/mcp-server.md).

## The four decision primitives

Every memory system eventually has to answer four questions. merken
makes each one an explicit decision with inputs, outputs, and an audit
row.

| Primitive | What it decides | Default implementation |
|---|---|---|
| `should_remember` | Does this event merit a write? | `HeuristicWriteDecider` — skip empty / too short / too long / exact duplicate via an in-process set that hydrates lazily from vstash |
| `should_consolidate` | Is it time to distill episodic events into semantic facts? | `PeriodicConsolidator` — fires when ≥ `min_events` unconsolidated events accumulate |
| `should_recall` | Which layers to query and with what budget? | `LayeredRecaller` — semantic first, episodic fallback, round-robin interleave with dedup by path |
| `should_forget` | Is this event safe to tombstone? | `NeverForget` — safe default, only forgets on `force=True` or with `ForgetConsolidated` opt-in |

Full depth: [`docs/primitives.md`](docs/primitives.md).

Every decision writes a row to the `merken_audit` collection — you can
always query *why* something was kept or dropped:

```bash
merken audit should_remember
merken audit dup_exact
```

## Memory layers

- **episodic** — raw events, high volume, low information density
- **semantic** — consolidated facts derived from episodic, with
  `derived_from` provenance pointers to the source events
- **audit** — every decision the loop made, queryable via
  `mem.audit()` / `merken audit`
- **tombstones** — forgotten events with full text preserved for
  unforgetting, queryable via `mem.tombstones()` / `merken tombstones`

Deeper: [`docs/architecture.md`](docs/architecture.md).

## Quick start

```bash
# Clone and install
git clone https://github.com/stffns/engram && cd merken
pip install -e .

# Run the full test suite (~10s)
python3 -m pytest tests/ -q

# Try the CLI
merken remember "the user asked about postgres on 2026-04-08"
merken recall "postgres"
merken status

# Attach to Claude Code
claude mcp add merken -- merken-mcp
```

## Tests and scenarios

- **154 tests** across four decision primitives, three deployment
  surfaces, and three `loop_quality` scenarios.
- **Loop-quality scenarios** live in
  [`experiments/loop_quality/`](experiments/loop_quality/) and enforce
  that every decider change is validated against at least one
  real-content fixture before landing:
    1. `analytics_project` — synthetic control, 100%/100%/100%
    2. `session_2026_04_09` — synthetic borderline, 100%/100%/33%
    3. `jay_vstash_2026_04_09_snapshot` — real organic content from
       a live vstash, 100%/100%/80%
- **Public retrieval benchmarks** live in
  [`experiments/retrieval/`](experiments/retrieval/) for absolute
  positioning against published competitor claims. LongMemEval runner
  is implemented; full n=500 overnight run is Phase A of the roadmap.

Measurement doctrine:
[`experiments/BENCHMARK_STRATEGY.md`](experiments/BENCHMARK_STRATEGY.md).

## Repository layout

```
merken/
├── README.md                    ← you are here
├── CONSTITUTION.md              ← principles, non-negotiables
├── CLAUDE.md                    ← session entry-point for Claude sessions
├── pyproject.toml               ← package config, [project.scripts]
├── merken/                      ← the package itself
│   ├── __init__.py              ← public surface
│   ├── memory.py                ← Memory class, the glue
│   ├── consolidation.py         ← Fact, clustering, consolidate pipeline
│   ├── audit.py                 ← audit + tombstone row formats
│   ├── cli.py                   ← merken CLI entry point
│   ├── mcp_server.py            ← merken-mcp MCP server entry point
│   └── policies/
│       ├── should_remember.py
│       ├── should_recall.py
│       ├── should_consolidate.py
│       ├── should_forget.py
│       └── types.py             ← shared Event / Decision / Protocol
├── docs/                        ← user-facing documentation
│   ├── architecture.md          ← the memory model and the loop in depth
│   ├── primitives.md            ← each decision primitive's semantics
│   ├── cli.md                   ← every CLI command with examples
│   ├── mcp-server.md            ← MCP tools reference + Claude Code setup
│   └── extending.md             ← write your own decider
├── experiments/                 ← the empirical bar (CONSTITUTION §9)
│   ├── BENCHMARK_STRATEGY.md    ← measurement doctrine
│   ├── loop_quality/            ← merken's design bar (scenario runner)
│   │   ├── runner.py
│   │   ├── scenario.py
│   │   ├── RESULTS.md
│   │   └── scenarios/*.json
│   └── retrieval/               ← absolute positioning (public benches)
│       └── longmemeval/
│           ├── runner.py
│           ├── dataset.py
│           └── RESULTS.md
├── tests/                       ← pytest suite, every primitive + surface
└── notes/                       ← working notes, prior art, research
    ├── prior-art.md             ← mempalace retrospective
    ├── silt.md                  ← memorial + rule derivations
    ├── research-2026-04-09.md   ← 6 papers verified + findings
    └── vstash-issue-*.md        ← upstream issue drafts
```

## Configuration and defaults

| Knob | Default | Where to change |
|---|---|---|
| Project name | `"default"` (or `$ENGRAM_PROJECT`) | `Memory(project=...)` / `--project` / env |
| DB path | `~/.merken/<project>.db` | `Memory(db=...)` / `--db` / `$ENGRAM_DB` |
| Collection | `"default"` | `Memory(collection=...)` |
| Embedding model (consolidation) | read from vstash `store_meta` at runtime; fallback to `vstash.config.EmbeddingsConfig().model` | set on the vstash side |
| Consolidation method | `"embedding_v1"` | `mem.consolidate(method=...)` |
| Embedding threshold | `0.70` (complete linkage) | `mem.consolidate(embedding_threshold=...)` |
| Clustering linkage | `"complete"` | `mem.consolidate(embedding_linkage=...)` |
| `should_remember` decider | `HeuristicWriteDecider()` | `Memory(write_decider=...)` |
| `should_recall` decider | `LayeredRecaller()` (sem 5, epi 3) | `Memory(recall_decider=...)` |
| `should_consolidate` decider | `PeriodicConsolidator(min_events=10)` | `Memory(consolidate_decider=...)` |
| `should_forget` decider | `NeverForget()` (safe) | `Memory(forget_decider=...)` |

**Deliberately isolated by default:** merken's default DB is NOT your
`~/.vstash/memory.db`. It lives under `~/.merken/<project>.db` so a
buggy decider can't corrupt your main vstash store. To attach merken
to a live vstash, point at it explicitly via `--db` / `$ENGRAM_DB`.

## Design non-negotiables

From [`CONSTITUTION.md`](CONSTITUTION.md), enforced in
[`CLAUDE.md`](CLAUDE.md):

1. **Local-first.** No mandatory network calls. Default install runs
   offline against a local embedder and a local vstash.
2. **Glass box.** Every decision writes to the audit collection.
3. **Single process by default.** No daemons, no queues, no Redis.
4. **vstash is a hard dependency.** merken never reimplements retrieval
   or reaches into `vstash._private`.
5. **Empirical first.** Every default-policy change cites a benchmark
   in `experiments/`.
6. **Silt's rule.** Before proposing an algorithm, look at the
   distribution of the data. See [`notes/silt.md`](notes/silt.md).

## Development status

merken is pre-v0.1. The four decision primitives are implemented and
tested, the three deployment surfaces are working, and the loop-quality
safety net is in place. What's next is **use** — putting merken in
front of real agent workflows and watching what the loop does with
organic content.

### What's deliberately not here

- **Knowledge graph.** CONSTITUTION §5 keeps this optional and gated.
- **LLM-based consolidation in the hot path.** Gated on a scenario
  where the non-LLM loop leaves real value on the table.
- **Bespoke compression dialect.** See `notes/prior-art.md` for why.
- **Spatial vocabulary** (wings/rooms/etc.). Use vstash's existing
  `project` / `collection` / `layer` / `tags` fields.
- **MCP tool sprawl.** Eight tools, one per primitive. No more.
- **Public leaderboard.** Premature at pre-v0.1.

### What's coming

- Claude Code hooks (auto-save + precompact) — depends on the MCP
  server being stable, which it is.
- A second and third real-content scenario in `loop_quality/`.
- LoCoMo runner under `experiments/retrieval/locomo/` (Phase B of
  [`experiments/BENCHMARK_STRATEGY.md`](experiments/BENCHMARK_STRATEGY.md)).
- Overnight full n=500 LongMemEval run (Phase A).

## License

MIT.

---

*Read [`CONSTITUTION.md`](CONSTITUTION.md) for why merken exists.
Read [`CLAUDE.md`](CLAUDE.md) for how to work in the repo.
Read [`docs/`](docs/) for how to use merken.*
