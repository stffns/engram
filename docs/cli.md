# merken CLI reference

The `merken` CLI is a thin wrapper around the Python SDK. Every
command maps 1:1 to a `Memory` method. There is no business
logic in the CLI — if a command needs behavior the SDK doesn't
have, the method goes in `Memory` first and the CLI wraps it.

## Install

```bash
cd ~/Desktop/Personal/Projects/merken
pip install -e .
```

This puts `merken` on your `$PATH`. Verify:

```bash
which merken
# /Users/you/.pyenv/shims/merken   (or similar)

merken --help
```

## Global flags

All commands accept three global flags (pass them *before* the
subcommand):

| Flag | Default | What it does |
|---|---|---|
| `--project NAME` | `default` (or `$ENGRAM_PROJECT`) | Logical project name. Drives the default DB path and tags every write. |
| `--db PATH` | `~/.merken/<project>.db` | vstash DB to open. Deliberately isolated from `~/.vstash/memory.db`. |
| `--json` | off | Emit JSON instead of human-readable output. Pass it anywhere before the subcommand. |

**Environment variable:** `ENGRAM_PROJECT` sets the default
project name when `--project` is not given. Same priority as
git reading `~/.gitconfig`.

**Non-destructive default DB path:** the first time you run
`merken remember ...` without `--db`, merken creates
`~/.merken/default.db` and writes to it. Your main `~/.vstash/`
store is never touched. If you want merken to attach to your
real vstash:

```bash
merken --db ~/.vstash/memory.db remember "..."
```

## Commands

### `remember`

Write an event to memory.

```bash
# Positional text
merken remember "the user picked Postgres for the analytics warehouse"

# Via stdin (pipe-friendly)
cat note.txt | merken remember --stdin
echo "an event" | merken remember --stdin

# With metadata
merken remember "design meeting outcome" \
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
| `--immutable` | off | Add `source:authoritative` unless another `source:` tag is already present. Consolidate/forget skip authoritative events. |

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
than ~20 characters with `status="empty"`. merken surfaces this
as `reason="vstash_rejected:empty"` and `written=False`. If you
need to store short events, tag them with context or wrap them
in longer strings.

### `recall`

Query memory and get ranked hits.

```bash
# Default: layered routing via should_recall
merken recall "what database did we pick for analytics?"

# Restrict to one layer (bypasses the decider)
merken recall "analytics" --layer semantic
merken recall "analytics" --layer episodic --top-k 10

# JSON for piping
merken --json recall "analytics" | jq '.[0].text'
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

### `recall-briefs`

Dual-channel recall for `brief_v1`: fetch current briefs from the
semantic layer and normal episodic hits separately.

```bash
# Human preview
merken recall-briefs "what is the current caching strategy?"

# JSON for hook/context assembly
merken --json recall-briefs "what is the current caching strategy?" \
    --brief-k 3 \
    --top-k 5
```

This command returns briefs and episodic hits in separate channels so
callers can prepend briefs above raw retrieval context. It is the CLI
wrapper around `Memory.recall_with_briefs()`.

**Flags:**

| Flag | Default | Meaning |
|---|---|---|
| `query` (positional) | required | Free-text query. |
| `--top-k N` | `5` | Max episodic hits. |
| `--brief-k N` | `3` | Max `method:brief_v1` semantic briefs. |
| `--max-brief-tokens N` | `8000` | Rough token budget for returned briefs. |

**Output (`--json`):**

```json
{
  "briefs": [
    "## Caching Strategy\n**As of:** 2026-04-19\n- **Current state:** ..."
  ],
  "episodic": [
    {
      "title": "cache-migration-note",
      "path": "text://cache-migration-note",
      "text": "Reverted the cache layer to Caffeine...",
      "score": 0.0166,
      "chunk": 0
    }
  ]
}
```

### `consolidate`

Cluster episodic events into semantic facts, or generate temporal
briefs when `--method brief_v1` is explicitly selected.

```bash
# Default: runs if the decider says so (PeriodicConsolidator, min_events=10)
merken consolidate

# Force regardless of decider
merken consolidate --force

# Lower threshold (more aggressive clustering)
merken consolidate --force --threshold 0.65

# Generate `brief_v1` briefs (requires GEMINI_API_KEY or GOOGLE_API_KEY)
merken consolidate --method brief_v1 --force

# JSON for inspection
merken --json consolidate --force
```

**Flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--method METHOD` | `embedding_v1` | Strategy. `embedding_v1` / `jaccard_v1` / `recall_v1` / `brief_v1`. |
| `--threshold FLOAT` | `0.70` | Cosine threshold for `embedding_v1`. Calibrated via grid search. |
| `--min-cluster N` | `2` | Minimum events to form a cluster. Singletons stay as episodic. |
| `--force` | off | Bypass the `should_consolidate` decider. |

`brief_v1` uses Gemini by default through `GEMINI_API_KEY` or
`GOOGLE_API_KEY`; set `MERKEN_BRIEF_MODEL` to override the model
name. It stores briefs with `method:brief_v1` tags so
`recall-briefs` can retrieve them separately from episodic events.

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
preserved in the `merken_tombstones` collection.

```bash
# Default: NeverForget (no-op unless --force)
merken forget

# Forget events already in a semantic fact
merken forget --decider consolidated

# Wipe all episodic events (still writes tombstones)
merken forget --force --verbose
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
merken audit

# By specific decision type
merken audit should_remember
merken audit should_recall
merken audit should_consolidate

# By reason
merken audit dup_exact
merken audit novel
merken audit too_short

# JSON for piping
merken --json audit should_remember | jq '.[].text'
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
merken tombstones

# Search by content
merken tombstones "kafka meeting"
merken tombstones postgres
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
merken status
merken --project medlocal status
merken --db ~/.vstash/memory.db --json status
```

**Output (human):**

```
project:     default
db:          /Users/you/.merken/default.db
collection:  default
total:       47
  episodic              42
  semantic              5
```

**Output (`--json`):**

```json
{
  "project": "default",
  "db": "/Users/you/.merken/default.db",
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
across *all* collections in the DB (not just merken's default
collection), total chunks, DB size, etc.

```bash
merken stats
merken --json stats
```

**Output (human):**

```
documents=52 chunks=134 collections=3 db_size_mb=2.04 db_path='/Users/you/.merken/default.db'
```

**Output (`--json`):**

```json
{
  "documents": 52,
  "chunks": 134,
  "collections": 3,
  "db_size_mb": 2.04,
  "db_path": "/Users/you/.merken/default.db"
}
```

## Common workflows

### Ingest a file, line by line

```bash
while IFS= read -r line; do
    merken remember "$line"
done < notes.txt
```

Dedup is automatic — lines already in the store get
`reason=dup_exact`.

### Pipe a log into memory

```bash
tail -f /var/log/agent.log | while read line; do
    merken remember --stdin <<< "$line"
done
```

Or as a one-shot:

```bash
cat agent_session.log | merken remember --stdin
```

### Daily consolidation cron

```bash
# crontab: 0 23 * * *  /path/to/daily_consolidate.sh

#!/usr/bin/env bash
merken --project daily consolidate --force
merken --project daily forget --decider consolidated
```

### Recall into a Claude prompt

```bash
context=$(merken --json recall "$user_question" | jq -r '.[].text' | head -3)
echo "Context:\n$context\n\nQuestion: $user_question" | claude
```

### Inspect why something was dropped

```bash
merken audit too_short
merken audit dup_exact
merken audit vstash_rejected
```

## Multiple projects

Each project has its own DB file. Switch between them with
`--project`:

```bash
merken --project medlocal remember "new demo case: meningococcemia"
merken --project perf_migration remember "SPL BR ready for CI"
merken --project analytics remember "PG 16 chosen over SQLite"

merken --project medlocal status
merken --project perf_migration recall "SPL"
```

Or set a default for a session:

```bash
export ENGRAM_PROJECT=medlocal
merken remember "another note"
merken status   # now implicitly medlocal
```

## Troubleshooting

**`(no hits)` when you know the event is there.** You probably
wrote it with a layer other than `episodic` and are recalling
with the default layered routing. Try `--layer episodic` or
check `merken status`.

**`reason=vstash_rejected:empty` on text you know is not
empty.** Your text is shorter than ~20 characters. vstash's
ingest pipeline has a minimum-length guardrail. Add context or
wrap in a longer string.

**CLI seems to use a different DB than expected.** Check
`$ENGRAM_PROJECT` and `$ENGRAM_DB`. Run `merken status` to
print the resolved DB path.

**Consolidate runs but produces 0 facts.** Check
`merken audit should_consolidate` — the decider may be skipping
because of `too_few_events`. Or use `--force`. If it still
writes 0, the events' pairwise cosine is below 0.70 — try
lowering `--threshold` temporarily, or add a
`loop_quality` scenario that exposes the gap.

**Writes appear to succeed but recall returns nothing.** There's
a race between writes and vstash's FTS index. In practice
merken's tests have never hit this, but if you do, check that
`merken stats` shows the document count going up — that rules
out the write actually failing silently.

## Further reading

- [`primitives.md`](primitives.md) — what each decision primitive
  does under the hood
- [`mcp-server.md`](mcp-server.md) — the same commands as MCP tools
- [`extending.md`](extending.md) — write your own decider and wire
  it through the CLI
- `merken/cli.py` — the source of truth
