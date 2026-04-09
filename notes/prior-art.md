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

*This file is working notes. Move stable conclusions into
`CONSTITUTION.md`. Move broken claims into a strikethrough section so
the mistake stays visible.*
