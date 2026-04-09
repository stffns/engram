# CLAUDE.md — engram

This file is the entry point for any Claude session opened in the engram repo.

## Read these first, in order

1. [`CONSTITUTION.md`](CONSTITUTION.md) — what engram is, what it isn't, and the
   principles. Disagreements get edits to that file *before* any code.
2. The vstash repo's `CLAUDE.md` — engram is a strict consumer of vstash's
   public API. Understand the substrate before touching the loop.

## Current phase

**Phase 0 — floor and door.** The repo has:

- `pyproject.toml` with `vstash >= 0.25.0`, `pydantic >= 2`, dev extras for
  `pytest`, `ruff`, `mypy`.
- `engram/memory.py` exposing a `Memory` class with two methods: `remember`
  and `recall`. Both are thin wrappers over `vstash.Memory`. No decision
  policies, no consolidation, no audit log.
- `tests/test_smoke.py` doing a remember → recall round trip.

## Hard rules

- **vstash is a hard dependency.** Never reach into `vstash._private`. Never
  read SQLite tables directly. If the public API is missing something, the
  fix is a vstash PR, not a workaround. (CONSTITUTION §4.4, §6.)
- **Glass box.** Every decision the loop makes about memory must be
  inspectable. Phase 0 has no decisions yet — when policies land in Phase 1,
  they each write to a `layer="audit"` collection.
- **No new vector storage.** No ChromaDB, no second store, no FTS
  reimplementation. (CONSTITUTION §3, §8.)
- **No bespoke compression dialect.** Measure with a real tokenizer
  before claiming any compression result. See `notes/prior-art.md` for
  the cautionary tale.
- **Empirical first.** Every default-policy change cites a benchmark in
  `experiments/`. (CONSTITUTION §9.)

## What NOT to do next

- Do not write the consolidation pipeline yet (Phase 5, gated on a benchmark).
- Do not add an MCP server yet (Phase 4).
- Do not invent a spatial vocabulary (wings/rooms/etc.). Use vstash's existing
  metadata fields: `project`, `collection`, `layer`, `tags`.
- Do not add a knowledge graph yet (Phase 6, optional, gated).

## Branching

`feature/*` → `develop` → `main` via release PR. Mirrors vstash.
