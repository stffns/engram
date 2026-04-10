# Prior art — what we read before building, and what stuck

> A single place where engram tracks notes on existing memory systems
> we looked at while designing the loop. **Nothing here is a target.**
> Engram's job is to be useful for *the user's* loop, not to beat any
> system on any benchmark. If a finding from prior art changes a design
> decision, the change goes into `CONSTITUTION.md`; this file is just
> the working notes.

---

## mempalace (xiaowu0162 / Milla Jovovich + Ben Sigman, 2026-04-07)

**What it is:** an open-source AI memory system that bundles ChromaDB
storage, a spatial taxonomy (wings/rooms/halls/closets/drawers), an
optional lossy compression dialect (AAAK), a SQLite-backed temporal
knowledge graph, and an MCP server with 19 tools. Pitched as "the
highest-scoring AI memory system ever benchmarked."

**Why we looked:** it's the most public attempt at the same shape of
problem engram targets — an agent-loop layer over a retrieval substrate.
Reading it is faster than re-deriving the trade-offs from scratch.

### What aged well in 48 hours

- **Raw verbatim storage beats LLM-extracted facts on retrieval
  benchmarks.** Their 96.6% LongMemEval R@5 in raw mode vs ~85% for
  Mem0/Zep is the strongest available evidence that "let an LLM decide
  what's worth remembering" is the wrong default. **Implication for
  engram:** consolidation episodic→semantic is *additive*, not
  destructive. Episodic stays raw. Semantic is a derived index, not a
  replacement.
- **Metadata filtering is the largest single retrieval lever.** Their
  "+34% palace boost" is standard ChromaDB `where` clauses on
  `(wing, room)`. They admit it. **Implication for engram:** vstash
  already has `project`, `collection`, `layer`, `tags`. We don't need
  to invent a vocabulary on top.
- **MCP is the integration surface that matters.** Adoption story is
  `claude mcp add mempalace`, not `import mempalace`. **Implication
  for engram:** ship the SDK first to keep the loop honest, but don't
  defer MCP to v0.5 either.
- **Auto-save hooks change the user contract.** Without write triggers,
  no memory accumulates. **Implication:** engram needs a Claude Code
  hooks path before it claims to be "live."
- **Temporal knowledge graphs in SQLite are tractable.**
  `add_triple / query_entity(as_of=) / invalidate / timeline` in one
  file, no Neo4j. **Implication:** if engram ever wants a KG, it
  doesn't need new infrastructure.

### What blew up in 48 hours (the lessons we're stealing in negative)

The mempalace team posted a brutal correction note on launch day:

- **AAAK lossy compression was over-claimed.** "30x lossless" turned
  out to be a heuristic with no real tokenizer; the real benchmark
  showed AAAK *regresses* by 12.4 points (84.2% vs 96.6% raw mode).
  **Lesson for engram:** no bespoke compression dialect. If we ever
  need compression, measure with a real tokenizer first.
- **"+34% palace boost" was misleading framing.** Wings/rooms is
  metadata filtering, not a novel retrieval mechanism. **Lesson:**
  don't dress up standard substrate features as architectural
  innovations.
- **`fact_checker.py` existed but was not wired into the knowledge
  graph operations** the README implied. **Lesson:** if a feature is
  in the README, it's wired to the API and there's a test.
- **"100% with Haiku rerank"** was real but the rerank pipeline was
  not in the public benchmark scripts. **Lesson:** every published
  number has a runner in the repo at the same commit.

### What we deliberately do *not* take from mempalace

- **The spatial vocabulary** (wings, rooms, halls, closets, drawers,
  tunnels). Six nouns where two would do, and most of the magic is
  metadata filtering. Engram uses vstash's existing fields.
- **AAAK or any other bespoke compression dialect.**
- **Mining as the onboarding story** (`mempalace mine ~/chats/`).
  Engram is a *loop*, not a one-shot importer. Live capture via hooks
  comes first; retroactive import can come later if it earns its keep.
- **19 MCP tools.** The §7 principle in CONSTITUTION is
  "embarrassingly small at the top level." Engram targets ~6 tools.
- **`store everything` as the only policy.** It wins LongMemEval but
  it's not a memory *system* — it's a search index over chat logs.
  The decision loop (`should_remember`/`recall`/`consolidate`/`forget`)
  is the whole reason engram exists.

### How this maps to engram decisions

| mempalace finding | engram decision |
|---|---|
| Raw verbatim wins retrieval benchmarks | Episodic layer stays raw; consolidation produces an *additional* semantic layer, never replaces episodic |
| Metadata filtering is the main retrieval lever | Use vstash's `project`/`collection`/`layer`/`tags`; do not invent vocabulary |
| MCP is the integration surface | Ship MCP server in an early phase, not a late one |
| Auto-save hooks make adoption real | Phase 4 must include hooks |
| AAAK over-claim retro | Pre-write `notes/prior-art.md` and `RESULTS.md` honesty rules *before* publishing numbers |
| Spatial palace metaphor mostly cosmetic | Avoid coining vocabulary |
| 19 MCP tools is a smell | Cap engram MCP at ~6 |

---

## Mem0, Zep, LangChain memory

Mentioned in `CONSTITUTION.md` §1 as the prior art that bundles
substrate and loop into a single black box. We have not done a deep
dive on each — the framing decision is already in the constitution and
nothing in the mempalace launch changed it. If a specific feature from
one of these systems looks worth examining, it gets a section here.

---

## neo4j-labs/agent-memory — engram's cousin, not its competitor

> *Analyzed 2026-04-10, after engram's four decision primitives were
> complete. neo4j-labs/agent-memory is the closest published system
> to engram in **shape** — both are policy/loop layers sitting on
> top of a substrate they don't reimplement. The substrates are
> what differ.*

### The parallel

```
┌────────────┬──────────────────────────────────┬──────────────────────────────────┐
│    Capa    │       Dense-retrieval side       │          Graph side              │
├────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ Substrate  │ vstash (SQLite + sqlite-vec +    │ Neo4j (graph + vector + text     │
│            │ FTS5 + RRF)                      │ search)                          │
├────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ Agent-loop │ engram (4 decision primitives    │ neo4j-labs/agent-memory          │
│            │ + audit + tombstones)            │ (3 tiers + entity extraction     │
│            │                                  │ + POLE+O)                        │
└────────────┴──────────────────────────────────┴──────────────────────────────────┘
```

Both are policy layers on a substrate. Both decide what to
remember, what to recall, what to consolidate. The API shapes
are parallel:

- engram: `should_remember` / `should_recall` /
  `should_consolidate` / `should_forget` as explicit decision
  primitives with audit logs
- neo4j-labs: short-term/long-term/reasoning tiers with entity
  extraction pipeline and entity resolution

### The philosophical difference

Two theories of memory:

**Dense substrate + policy loop (engram/vstash):**

"Remember the text raw, fuse with RRF adaptivo, and the loop
decides what to promote to consolidated facts. The agent does
reasoning at runtime over chunks."

- Write cost: trivial (embed + store). Consolidation is cosine
  clustering, no NER, no LLM in the hot path.
- Read cost: `vstash.search` + interleave + the *caller's* LLM
  reasoning over the hits. engram itself does not reason — it
  returns strings.
- Failure mode: **retrieval miss**. The event IS in the DB,
  vstash just didn't surface it for this query. Recoverable by
  improving query / embedder / threshold. **No information loss.**

**Symbolic substrate + extraction pipeline (neo4j-labs):**

"Extract entities and relations at write time, and the graph
answers structured queries. Reasoning happens at the write
side."

- Write cost: NER + relation extraction per event (spaCy /
  GLiNER / LLM). Expensive, fragile if extractor fails.
- Read cost: graph traversal + optional vector similarity.
  Cheap for relational queries ("all entities connected to X").
- Failure mode: **extraction miss**. The NER didn't recognize
  the entity, the relation was never extracted, **no node
  represents that knowledge in the graph**. Potentially
  unrecoverable without re-ingest with a better extractor.

### Cost model comparison

```
                     engram/vstash                neo4j-labs/neo4j
write cost           O(N²) cosine on consolidate  O(N) NER + relation per event
read cost            O(log N) hybrid + interleave  O(log N) graph + vector hybrid
failure mode         retrieval miss (recoverable)  extraction miss (structural)
query shape          text similarity               structured relational
extractor fragility  none                           high (NER/LLM quality)
content agnosticism  high (any text survives)       low (needs recognizable entities)
```

The asymmetry that matters most: **engram never loses
information by pipeline failure**. It may lose *access* to
information (a bad query, a wrong threshold), but the text is
always in episodic. neo4j-labs can lose information
structurally if the extractor misses an entity on ingest.

The counter-case: if the extractor IS good for your domain
(a stable schema, typed business entities, academic papers),
the structured queries you get from the graph are worth the
extraction fragility. Domains with stable entity types
probably prefer the graph approach.

### engram already has a seed of graph structure

Every semantic `Fact` in engram has `derived_from: list[str]`
— paths to the episodic events that produced it. That's a
bipartite graph:

```
episodic events ──derived_from──▶ semantic facts
```

Not traversable yet (no "give me all events that contributed
to fact X" query exists), but the structure is in the data.
A future `engram_edges` table would generalize this implicit
graph, not invent one from scratch.

### Why we're NOT building a KG layer now

Three concrete reasons, not just "later":

1. **Extraction source is the unsolved problem.** LLM
   extraction at `consolidate()` time inherits neo4j-labs'
   fragility. Rule-based extraction (spaCy/GLiNER) adds a
   pipeline dependency engram has explicitly avoided. User-
   supplied tags are safe but add nothing `vstash` tags don't
   already provide. None of the three options is clearly
   better than no-extraction for v1.

2. **Query shape changes.** Once you have edges, users expect
   graph queries ("all facts about entity X", "path from A
   to B"). That requires either a query language, or wrapping
   a graph engine, or reinventing traversal. The "lightweight
   KG" scope-creeps into an engineering project.

3. **CONSTITUTION §6 gates this.** The constitution already
   anticipates the KG question and defers it: "optional,
   gated on a benchmark." No `loop_quality/` scenario
   currently exposes a gap that relational queries would
   solve. The scenario is the gating requirement.

### What we learned from reading neo4j-labs

1. **engram and neo4j-labs are parallel, not competing.**
   Comparing them on LongMemEval (dense retrieval) would be
   like comparing a car and a boat on a road — the graph
   system isn't built for that benchmark and engram isn't
   built for entity-centric queries.

2. **The right framing for engram in the landscape:**
   engram is the **policy-loop layer for dense-retrieval
   substrates**. neo4j-labs is the **policy-loop layer for
   graph substrates**. Both are loops on substrates. The
   substrate choice is upstream of both.

3. **If engram ever adds a graph view,** the honest path is
   caller-supplied edges via tags (option 3 above), not
   extraction in the hot path. The graph view would be a
   secondary index on the primary dense-retrieval model, not
   a replacement.

### How this maps to engram decisions

| neo4j-labs finding | engram decision |
|---|---|
| Entity extraction at write-time is expensive and fragile | Keep consolidation write-time cost cheap (cosine, no NER/LLM) |
| Extraction miss = structural information loss | Episodic layer is write-once, never extracted from — information survives even when recall fails |
| POLE+O data model gives structured queries | `derived_from` tags are the seed of a graph view; generalize only if a scenario demands it |
| 3 memory tiers (short/long/reasoning) | 2 layers (episodic/semantic) + audit + tombstones. Procedural is the third layer from CONSTITUTION §5, gated and not implemented. |
| Dense-retrieval vs symbolic-extraction are two memory theories | engram explicitly bets on the dense side. The bet is: embeddings are sufficient and the loop only needs to decide what to keep/promote/forget. Reviewable if a scenario shows otherwise. |

---

*This file is working notes. Move stable conclusions into
`CONSTITUTION.md`. Move broken claims into a strikethrough section so
the mistake stays visible.*
