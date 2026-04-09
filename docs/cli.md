# engram CLI reference

The `engram` CLI is a thin wrapper around the Python SDK. Every
command maps 1:1 to a `Memory` method. There is no business
logic in the CLI — if a command needs behavior the SDK doesn't
have, the method goes in `Memory` first and the CLI wraps it.

## Install

```bash
cd ~/Desktop/Personal/Projects/engram
pip install -e .
```

This puts `engram` on your `$PATH`. Verify:

```bash
which engram
# /Users/you/.pyenv/shims/engram   (or similar)

engram --help
```

## Global flags

All commands accept three global flags (pass them *before* the
subcommand):

| Flag | Default | What it does |
|---|---|---|
| `--project NAME` | `default` (or `$ENGRAM_PROJECT`) | Logical project name. Drives the default DB path and tags every write. |
| `--db PATH` | `~/.engram/<project>.db` | vstash DB to open. Deliberately isolated from `~/.vstash/memory.db`. |
| `--json` | off | Emit JSON instead of human-readable output. Pass it anywhere before the subcommand. |

**Environment variable:** `ENGRAM_PROJECT` sets the default
project name when `--project` is not given. Same priority as
git reading `~/.gitconfig`.

**Non-destructive default DB path:** the first time you run
`engram remember ...` without `--db`, engram creates
`~/.engram/default.db` and writes to it. Your main `~/.vstash/`
store is never touched. If you want engram to attach to your
real vstash:

```bash
engram --db ~/.vstash/memory.db remember "..."
```

## Commands

### `remember`

Write an event to memory.

```bash
# Positional text
engram remember "the user picked Postgres for the analytics warehouse"

# Via stdin (pipe-friendly)
cat note.txt | engram remember --stdin
echo "an event" | engram remember --stdin

# With metadata
engram remember "design meeting outcome" \
    --layer episodic \
    --title "meeting_2026_04_08" \
    --tags "type:decision,project:analytics"
```

**Flags:**

| Flag | Default | Meaning |
|---|---|---|
| `text` (positional) | — | Event text. Mutually exclusive with `--stdin`. |
| `--stdin` | off | Read text from standard input instead. |
| `--title TITLE` | auto | Document title. Affects the vstash path. |
| `--layer LAYER` | `episodic` | Layer tag. `episodic` or `semantic` in practice. |
| `--tags TAGS` | none | Comma-separated tags. |

**Output (human):**

```
✓ wrote  reason=novel  policy=HeuristicWriteDecider
⊘ skipped  reason=dup_exact  policy=HeuristicWriteDecider
⊘ skipped  reason=vstash_rejected:empty  policy=HeuristicWriteDecider
```

**Output (`--json`):**

```json
{
  "written": true,
  "decision": {
    "write": true,
    "reason": "novel",
    "policy": "HeuristicWriteDecider",
    "confidence": 0.8
  }
}
```

**Exit code:** 0 on success (including skipped writes — the
decider's "no" is a valid outcome). 2 if neither `text` nor
`--stdin` was provided.

**Note on short text:** vstash silently rejects text shorter
than ~20 characters with `status="empty"`. engram surfaces this
as `reason="vstash_rejected:empty"` and `written=False`. If you
need to store short events, tag them with context or wrap them
in longer strings.

### `recall`

Query memory and get ranked hits.

```bash
# Default: layered routing via should_recall
engram recall "what database did we pick for analytics?"

# Restrict to one layer (bypasses the decider)
engram recall "analytics" --layer semantic
engram recall "analytics" --layer episodic --top-k 10

# JSON for piping
engram --json recall "analytics" | jq '.[0].text'
```

**Flags:**

| Flag | Default | Meaning |
|---|---|---|
| `query` (positional) | required | Free-text query. |
| `--top-k N` | `5` | Max results in the final list. |
| `--layer LAYER` | none | Restrict to one layer. Bypasses the `should_recall` decider entirely. |

**Output (human):**

```
1. fact_a1b2c3d4e5f6
   [observed 3×] The team picked Postgres for the analytics warehouse...
2. the-team-picked-postgres-20260408-145533
   the user picked Postgres for the analytics warehouse after...
3. the-analytics-warehouse-runs-20260408-153012
   the analytics warehouse runs on Postgres 16...
```

**Output (`--json`):**

```json
[
  {
    "title": "fact_a1b2c3d4e5f6",
    "path": "text://fact_a1b2c3d4e5f6",
    "text": "[observed 3×] The team picked Postgres ...",
    "score": 0.0166,
    "chunk": 0
  },
  ...
]
```

**Empty result:** `(no hits)` (human) or `[]` (JSON). Exit code
stays 0 — no hits is a valid outcome.

### `consolidate`

Cluster episodic events into semantic facts.

```bash
# Default: runs if the decider says so (PeriodicConsolidator, min_events=10)
engram consolidate

# Force regardless of decider
engram consolidate --force

# Lower threshold (more aggressive clustering)
engram consolidate --force --threshold 0.65

# JSON for inspection
engram --json consolidate --force
```

**Flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--method METHOD` | `embedding_v1` | Clustering strategy. `embedding_v1` / `jaccard_v1` / `recall_v1`. |
| `--threshold FLOAT` | `0.70` | Cosine threshold for `embedding_v1`. Calibrated via grid search. |
| `--min-cluster N` | `2` | Minimum events to form a cluster. Singletons stay as episodic. |
| `--force` | off | Bypass the `should_consolidate` decider. |

**Output (human):**

```
✓ 3 fact(s) from 12 events  method=embedding_v1  decider=PeriodicConsolidator
  [1] size=4  [observed 4×] The team picked Postgres for the analytics warehouse
  [2] size=2  [observed 2×] OAuth 2.0 with PKCE is now the only supported auth flow
  [3] size=2  [observed 2×] Grafana dashboards now show p99 latency per endpoint
```

Or, if skipped:

```
⊘ skipped  reason=too_few_events:2<10  (2 events examined)
```

**Output (`--json`):**

```json
{
  "events_examined": 12,
  "facts_written": 3,
  "skipped": false,
  "reason": "clustered_embedding>=0.7_linkage=complete_mincluster=2",
  "decider": "PeriodicConsolidator",
  "method": "embedding_v1",
  "facts": [
    {
      "text": "[observed 4×] The team picked Postgres for ...",
      "cluster_size": 4,
      "method": "concat_v1",
      "derived_from": ["text://...", "text://...", "text://...", "text://..."]
    },
    ...
  ]
}
```

### `forget`

Tombstone episodic events. **Reversible** — the full text is
preserved in the `engram_tombstones` collection.

```bash
# Default: NeverForget (no-op unless --force)
engram forget

# Forget events already in a semantic fact
engram forget --decider consolidated

# Wipe all episodic events (still writes tombstones)
engram forget --force --verbose
```

**Flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--decider NAME` | `never` | `never` or `consolidated`. |
| `--min-facts N` | `1` | For `consolidated`: minimum facts citing the event. |
| `--force` | off | Tombstone every episodic event regardless of decider. |
| `-v`, `--verbose` | off | List tombstoned event paths. |

**Output (human):**

```
✓ tombstoned=3  skipped=5  events=8  decider=ForgetConsolidated
```

With `-v`:

```
✓ tombstoned=3  skipped=5  events=8  decider=ForgetConsolidated
  tombstoned:
    - text://the-team-picked-postgres-20260408-145533
    - text://user-chose-postgres-20260408-150117
    - text://team-adopted-postgres-20260408-152244
```

**Output (`--json`):**

```json
{
  "tombstoned": ["text://...", "text://...", "text://..."],
  "skipped": [
    ["text://...", "not_consolidated:0<1"],
    ...
  ],
  "events_examined": 8,
  "decider": "ForgetConsolidated"
}
```

### `audit`

Query the decision audit log. Every `should_remember`,
`should_recall`, `should_consolidate`, and `should_forget`
decision writes a row here.

```bash
# All recent decisions (default query matches "should_")
engram audit

# By specific decision type
engram audit should_remember
engram audit should_recall
engram audit should_consolidate

# By reason
engram audit dup_exact
engram audit novel
engram audit too_short

# JSON for piping
engram --json audit should_remember | jq '.[].text'
```

**Flags:**

| Flag | Default | Meaning |
|---|---|---|
| `query` (positional) | `should_` | Free-text query against the audit collection. |
| `--top-k N` | `20` | Max rows to return. |

**Output (human):**

```
• audit:should_remember:novel:2026-04-09T11:04:33+00:00
    timestamp: 2026-04-09T11:04:33+00:00
    decision: should_remember
    write: True
    reason: novel
    policy: HeuristicWriteDecider
    confidence: 0.8
• audit:should_remember:dup_exact:2026-04-09T11:04:47+00:00
    timestamp: 2026-04-09T11:04:47+00:00
    decision: should_remember
    write: False
    reason: dup_exact
    policy: HeuristicWriteDecider
```

### `tombstones`

Query forgotten events. The tombstone collection preserves the
full original text + provenance for every forgotten event.

```bash
# All tombstones
engram tombstones

# Search by content
engram tombstones "kafka meeting"
engram tombstones postgres
```

**Flags:**

| Flag | Default | Meaning |
|---|---|---|
| `query` (positional) | `tombstone` | Free-text query against the tombstones collection. |
| `--top-k N` | `20` | Max rows to return. |

**Output (human):**

```
• tombstone:text://the-team-picked-postgres-...
    The team picked Postgres 16 for the new analytics warehouse because of write concurrency...
• tombstone:text://user-chose-postgres-...
    User chose Postgres for the analytics project because of concurrency concerns and...
```

### `status`

Project summary: project name, DB path, total event count,
per-layer breakdown.

```bash
engram status
engram --project medlocal status
engram --db ~/.vstash/memory.db --json status
```

**Output (human):**

```
project:     default
db:          /Users/you/.engram/default.db
collection:  default
total:       47
  episodic              42
  semantic              5
```

**Output (`--json`):**

```json
{
  "project": "default",
  "db": "/Users/you/.engram/default.db",
  "collection": "default",
  "total_events": 47,
  "layers": {
    "episodic": 42,
    "semantic": 5
  }
}
```

### `stats`

Pass-through to `vstash.Memory.stats()`. Reports document count
across *all* collections in the DB (not just engram's default
collection), total chunks, DB size, etc.

```bash
engram stats
engram --json stats
```

**Output (human):**

```
documents=52 chunks=134 collections=3 db_size_mb=2.04 db_path='/Users/you/.engram/default.db'
```

**Output (`--json`):**

```json
{
  "documents": 52,
  "chunks": 134,
  "collections": 3,
  "db_size_mb": 2.04,
  "db_path": "/Users/you/.engram/default.db"
}
```

## Common workflows

### Ingest a file, line by line

```bash
while IFS= read -r line; do
    engram remember "$line"
done < notes.txt
```

Dedup is automatic — lines already in the store get
`reason=dup_exact`.

### Pipe a log into memory

```bash
tail -f /var/log/agent.log | while read line; do
    engram remember --stdin <<< "$line"
done
```

Or as a one-shot:

```bash
cat agent_session.log | engram remember --stdin
```

### Daily consolidation cron

```bash
# crontab: 0 23 * * *  /path/to/daily_consolidate.sh

#!/usr/bin/env bash
engram --project daily consolidate --force
engram --project daily forget --decider consolidated
```

### Recall into a Claude prompt

```bash
context=$(engram --json recall "$user_question" | jq -r '.[].text' | head -3)
echo "Context:\n$context\n\nQuestion: $user_question" | claude
```

### Inspect why something was dropped

```bash
engram audit too_short
engram audit dup_exact
engram audit vstash_rejected
```

## Multiple projects

Each project has its own DB file. Switch between them with
`--project`:

```bash
engram --project medlocal remember "new demo case: meningococcemia"
engram --project perf_migration remember "SPL BR ready for CI"
engram --project analytics remember "PG 16 chosen over SQLite"

engram --project medlocal status
engram --project perf_migration recall "SPL"
```

Or set a default for a session:

```bash
export ENGRAM_PROJECT=medlocal
engram remember "another note"
engram status   # now implicitly medlocal
```

## Troubleshooting

**`(no hits)` when you know the event is there.** You probably
wrote it with a layer other than `episodic` and are recalling
with the default layered routing. Try `--layer episodic` or
check `engram status`.

**`reason=vstash_rejected:empty` on text you know is not
empty.** Your text is shorter than ~20 characters. vstash's
ingest pipeline has a minimum-length guardrail. Add context or
wrap in a longer string.

**CLI seems to use a different DB than expected.** Check
`$ENGRAM_PROJECT` and `$ENGRAM_DB`. Run `engram status` to
print the resolved DB path.

**Consolidate runs but produces 0 facts.** Check
`engram audit should_consolidate` — the decider may be skipping
because of `too_few_events`. Or use `--force`. If it still
writes 0, the events' pairwise cosine is below 0.70 — try
lowering `--threshold` temporarily, or add a
`loop_quality` scenario that exposes the gap.

**Writes appear to succeed but recall returns nothing.** There's
a race between writes and vstash's FTS index. In practice
engram's tests have never hit this, but if you do, check that
`engram stats` shows the document count going up — that rules
out the write actually failing silently.

## Further reading

- [`primitives.md`](primitives.md) — what each decision primitive
  does under the hood
- [`mcp-server.md`](mcp-server.md) — the same commands as MCP tools
- [`extending.md`](extending.md) — write your own decider and wire
  it through the CLI
- `engram/cli.py` — the source of truth
