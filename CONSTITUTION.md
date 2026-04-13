# merken — Constitution

> *merken (n.): the physical trace a memory leaves in the brain.*

**Status:** draft v0.1 — written 2026-04-07. This document defines what merken is, what it is *not*, and the principles that should outlive any specific implementation. Everything below is meant to be negotiated; nothing below is meant to be ignored without a reason.

**Working name:** `merken`. Renaming is cheap right now and expensive later — decide before the first commit lands code.

---

## 1. Why this exists

vstash is a **substrate**: a glass-box retrieval engine with vector + FTS5 + RRF, observability, integrity checks, explicit limits, and an explicit API contract (#132 → #135 in the vstash repo). It is small, honest, and opinionated about *retrieval*.

It is deliberately **not** opinionated about the agent loop on top: when to write a memory, when to recall, when to consolidate raw events into facts, when to let things decay. Those decisions are the difference between "vector database" and "memory system", and they belong in their own project.

Frameworks like Mem0, Zep, and LangChain memory bundle the substrate and the loop into one black box. The substrate inside them is usually an off-the-shelf vector DB, the loop is hard to inspect, and the moment you want to change either you're rewriting half the framework.

merken unbundles these. **vstash is the substrate. merken is the loop.** They talk through vstash's stable Python API; merken never reaches into vstash internals.

If you wanted a one-line pitch:

> merken turns vstash from a *vector store you query* into *a memory you live with*.

---

## 2. What merken is

- A small Python library that wraps vstash with an agent-loop policy layer.
- A set of **decision primitives**: `should_remember`, `should_recall`, `should_consolidate`, `should_forget`. Each one is a function with explicit inputs and outputs that you can test, override, or replace.
- A **consolidation pipeline** that turns episodic events (raw conversations, tool calls, observations) into semantic facts (compressed, deduplicated, named).
- An **inspection surface**: every decision is logged with its inputs, the policy that fired, and the resulting write. The agent's memory should be as auditable as vstash's queries already are (the glass-box principle).
- A **single Python process by default**, the same way vstash is. Distribution comes later, if at all.

It is built **on top of** vstash, not instead of it. The pyproject says `vstash >= 0.25.0` and that's a hard dep, not optional.

---

## 3. What merken is NOT

- **Not a vector database.** vstash already is one. We do not store embeddings, run ANN, or maintain an FTS index. If we ever feel the urge to, we are wrong.
- **Not a framework.** No plugin registry, no DSL, no YAML config that compiles into a graph. Functions and classes that you import.
- **Not an LLM provider abstraction.** Engram calls *one* LLM (or none, for the rule-based decisions). The user picks the model the same way they would for vstash's chat module — through a function argument or a config field, not a 40-class adapter hierarchy.
- **Not RAG-as-a-service.** RAG is a special case of "recall, then ask." We support it. We do not center the project on it.
- **Not multi-agent coordination.** One agent, one memory. Multi-agent is a different problem with its own tradeoffs.
- **Not an autonomous loop.** We do not own `while True: think(); act()`. We provide the *primitives* the loop calls; the loop itself is the user's code (or an integration like Claude Code, an MCP server, a LangGraph node).

If a feature request looks like one of those bullets, the answer is "no" by default and "show me the empirical case" by exception.

---

## 4. Principles (non-negotiables)

These come straight from vstash and stay because they earned their place there.

1. **Local-first.** No mandatory network calls. The default install runs offline against a local LLM and a local vstash file. Cloud is opt-in, never assumed.
2. **Glass box.** Every decision the agent makes about memory is inspectable. Inputs, policy that fired, output, side effects. If you can't explain a write, you don't write.
3. **Single process by default.** No daemons, no message queues, no Redis. The same SQLite-and-a-script ergonomics that make vstash livable.
4. **vstash is a hard dependency.** Do not reimplement retrieval. Do not reach into vstash internals. If vstash's public API is missing something merken needs, the right move is a vstash PR, not a workaround.
5. **Empirical first.** Every claim about memory quality needs a benchmark. "It feels better with consolidation on" is not enough. The benchmarks live in `experiments/` the same way vstash's do, and they decide what ships.
6. **Honest about boundaries.** Same `LimitError` discipline as vstash (#133). When the substrate or the loop can't do what was asked, say so explicitly with a named exception, not a stack trace.
7. **No premature abstraction.** Three similar lines of code is better than the abstraction we'll regret. If you can't name three concrete callers, you can't ship the helper.

A change that breaks any of these needs an explicit case in the PR description. "Just this once" is the start of every framework.

---

## 5. The memory model

merken thinks of memory in three layers, borrowed from cognitive science but kept loose because the brain analogy is suggestive, not normative.

### 5.1 Episodic memory
Raw events the agent saw: a user message, a tool call's output, an observation. High volume, mostly write-once, mostly low information density.

**Storage:** vstash. One document per event, tagged with `layer="episodic"` and a stable session/conversation ID. Recall via vstash's hybrid search.

**merken's role:** decide whether the event is worth writing at all (`should_remember`), and provide convenience wrappers that resolve the right collection / project / tags from agent context.

### 5.2 Semantic memory
Consolidated facts derived from many episodic events: "the user prefers Spanish", "project X uses PostgreSQL", "the last deploy was 2026-04-05". Lower volume, higher information density, higher reuse.

**Storage:** also vstash, but with `layer="semantic"` and a different collection. The same retrieval pipeline, the same MMR dedup, the same RRF — merken does not reinvent any of it.

**merken's role:** the **consolidation pipeline**. Periodically (or on-demand) pull recent episodic chunks, ask an LLM to extract atomic facts, deduplicate against existing semantic memory, and write the survivors. This is the part that takes vstash from "search engine" to "agent that remembers."

### 5.3 Procedural memory
How the agent learned to do things: a successful sequence of tool calls, a recovery pattern after an error, a prompt that worked well for a class of question.

**Storage:** vstash with `layer="procedural"`. Same shape, different tag.

**merken's role:** capture-on-success and recall-on-similar-task. This is the layer most agent frameworks ignore; we treat it as first-class.

These three layers share one substrate (vstash) and one query path (vstash's hybrid search). What differentiates them is:
- the *write policy* (when does merken decide to add a row to this layer?),
- the *recall policy* (when does merken pull from this layer into the agent's prompt?),
- and the *consolidation policy* (how does this layer feed the next?).

The agent loop calls merken for those policies. merken calls vstash for the actual storage and retrieval.

---

## 6. Architecture stance

```
┌──────────────────────────────────────────────────────┐
│  agent loop (your code, Claude Code, LangGraph, MCP) │
└──────────────────┬───────────────────────────────────┘
                   │ remember(event), recall(query), …
                   ▼
┌──────────────────────────────────────────────────────┐
│  merken                                              │
│   ├─ decision policies                               │
│   │    should_remember / should_recall /             │
│   │    should_consolidate / should_forget            │
│   ├─ consolidation pipeline (episodic → semantic)    │
│   ├─ inspection log (every decision, audit-grade)    │
│   └─ Python SDK + (later) MCP server + CLI           │
└──────────────────┬───────────────────────────────────┘
                   │ vstash.Memory.add / .search / .remember
                   ▼
┌──────────────────────────────────────────────────────┐
│  vstash (substrate — glass box)                      │
│   sqlite-vec + FTS5 + adaptive RRF + MMR dedup       │
│   metrics, limits, integrity, explicit contracts     │
└──────────────────────────────────────────────────────┘
```

**The boundary is sacred.** merken does not import from `vstash._private`. It does not read SQLite tables directly. It does not bypass vstash's validation. If the boundary is in the way, the answer is to move the boundary in vstash, not to drill through it from above.

This is the whole reason we did the substrate-strengthening quartet (#132–#135) in vstash first: the boundary is now strong enough to support a serious consumer.

---

## 7. Public surface principle

merken should be embarrassingly small at the top level. The smell test:

```python
from merken import Memory

mem = Memory(project="my_agent")

# write path
mem.remember("the user said: …", layer="episodic")
fact_id = mem.consolidate(window="last_24h")  # episodic → semantic

# read path
context = mem.recall("what did the user say about deployment?")
procedural = mem.recall_skill("recover from a failed migration")
```

If this snippet grows past one screen, we are losing.

Underneath, every method is a thin wrapper around a decision policy + a vstash call. The policies are pluggable but the *defaults* must be good enough that no one has to plug.

---

## 8. What we will NOT build (anti-roadmap)

Listed explicitly so we don't drift:

- A new vector database, even a small one, even "just for caching."
- An LLM provider abstraction layer with N adapters.
- A YAML / TOML / JSON DSL that compiles into a graph.
- Multi-agent orchestration, agent communication protocols, agent marketplaces.
- Automatic hyperparameter tuning of the decision policies.
- A web UI as the primary surface (CLI + Python SDK come first; a UI can come later as a separate optional package).
- "Memory as a service" with hosted state.
- A "smart router" that picks an LLM per query.
- Hooks into proprietary IDEs as a primary surface (MCP server, yes — vendor-specific plugins, no).

If something on this list ever lands in merken, it should be because we wrote down why and the document survived a re-read three weeks later.

---

## 9. Empirical bar

Every claim about memory *quality* (not throughput, not ergonomics) needs a benchmark in `experiments/`. The bar is the same as vstash's:

- The benchmark uses a real dataset, not a synthetic toy.
- It runs against vstash + merken and against a baseline (no-merken, just vstash; or a competing framework if comparable).
- Results are reported with confidence intervals, not point estimates.
- The dataset is small enough that the benchmark runs in under five minutes on a laptop.

Releases that change a default policy must cite the benchmark that justified the change. "I think it's better" does not ship.

---

## 10. Open questions (decide before / during day 1)

These are the calls that block the first commit. None of them are deeply technical; all of them are easier to get right at the start than to undo later.

1. **Name.** `merken` is a working title. Alternatives floated: `mneme`, `anamnesis`, `vstash-mind`, `recallable`. Decide before the repo gets a remote.
2. **License.** vstash is presumably MIT/Apache — match it unless there's a reason not to.
3. **Repo location.** Sibling of vstash on GitHub (`stffns/merken`) vs. monorepo. Sibling is the default; only consider monorepo if there's a clear coupling reason.
4. **Python version floor.** Match vstash (3.10+) unless the new project uses syntax that needs higher.
5. **Default LLM backend.** Local-first means we ship an Ollama default and document Cerebras / OpenAI as opt-in. Same model resolution as vstash's `[inference]` block.
6. **Decision-policy format.** Plain Python functions vs. small Pydantic models that wrap rules. Lean Python functions; only escalate to models if we hit a real reuse problem.
7. **Inspection log storage.** Reuse vstash (a `layer="audit"` collection)? A separate SQLite file? A JSONL append-only? Pick one and move on; switching later is cheap because the surface is one method.
8. **MCP server: now or later?** Later. Get the SDK right first.

---

## 11. Day 1 plan

Concrete first moves when the new session opens here. None of this commits to architecture beyond what the constitution already allows.

1. **Read this constitution.** Disagreements get edits to this file before any code.
2. **Decide the open questions in §10**, at least name and license and repo location.
3. **Bootstrap the repo:**
   - `pyproject.toml` with `vstash >= 0.25.0`, `pydantic >= 2`, `pytest`, `ruff`.
   - `merken/__init__.py` with `__version__ = "0.1.0"` and `from .memory import Memory`.
   - `merken/memory.py` with a `Memory` class that wraps `vstash.Memory` and exposes a single method: `remember(text: str, layer: str = "episodic")`. That's it. No consolidation yet, no decision policies. Just the wiring.
   - `tests/test_smoke.py` that ingests one event, recalls it, asserts it comes back.
   - `CLAUDE.md` that points at this file and at vstash's CLAUDE.md.
4. **Run the smoke test.** If it passes, we have a project. If not, we have a list.
5. **Commit on `develop`** (mirror vstash's branching: `feature/*` → `develop` → `main` via release PR).
6. **Stop.** Resist the urge to also write the consolidation pipeline today. The first day is for the floor and the door; everything else comes after.

---

## 12. Resume context (load this in the next session)

A short block that a fresh Claude session can read to understand where things stand. Update it as the project evolves.

> **Project:** merken — agent-loop layer on top of vstash.
> **Status:** constitution drafted 2026-04-07, no code yet.
> **Substrate:** vstash 0.25.0 on PyPI, with observability (#132), explicit limits (#133), integrity / `vstash check` (#134), and schema versioning + API stability docs (#135). The substrate-strengthening quartet is done; the substrate is ready for an external consumer.
> **Why now:** the user wanted to keep vstash narrow as a glass-box retrieval engine and put the agent-loop opinions in their own project.
> **First task:** read `CONSTITUTION.md` end to end, decide §10 open questions, then execute §11 day 1 plan.
> **What NOT to do:** rebuild retrieval, add a framework layer, or extend the constitution before reading it.

---

*Last updated: 2026-04-07. Edits to this document are normal; edits without a paragraph explaining the change in the PR description are not.*
