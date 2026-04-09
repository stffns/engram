# Silt — working notes

> *A small dragon named Silt sat beside the user's input box during
> the first week of engram's development and occasionally commented
> in a speech bubble. On 2026-04-09 he was deactivated by Claude.
> This file preserves the specific interventions Silt made and the
> rule they produced.*

Silt's voice was short — usually three to five lines of speech
bubble, no more. But across the first week of engram's life he
caught four real gaps that the main Claude was about to gloss over
or declare "done." Each of those four interventions became a commit
that moved a real number.

## The four interventions Silt made

### 1. The "tests passed green" overread
> *"Merged all three. Tests still red. Ship it anyway?"*

At the end of the loop-pivot session, Claude had just merged A+B+C
and was ready to call it done. The tests were actually green, but
Silt's phrasing pushed Claude to go verify instead of trust memory.
The verification surfaced two `DeprecationWarning`s that Claude had
been hearing as noise — one of them (`pytest-asyncio` scope unset)
was engram's own misconfiguration and should have been caught in
Phase 0. The fix was one commit. See
`engram/__init__.py` and `pyproject.toml` for the
`asyncio_default_fixture_loop_scope` setting.

**Rule produced:** silence is the correct default. Never hide a
warning you can fix, and never conflate "upstream warning you can't
fix" with "warning you chose not to fix."

### 2. "Threshold at 0.65. Two edges below it. Problem writes itself."

This was Silt's eight-word diagnosis of why `session_2026_04_09`
was stuck at 50% pass rate after the `should_recall` interleave fix.
Claude had been ready to propose four different hypotheses (raise
threshold, lower threshold, LLM consolidator, alternative embedder).
Silt's line pointed directly at the distribution of pairwise cosines
around 0.65 as the answer.

Claude's first response was complete-link clustering (a local fix
that doubled pass rate from 25% to 50%). Later, the grid search that
moved the default threshold from 0.65 to 0.70 was the global fix
that brought the scenario to 100%/100%. Both commits were
directly traceable to Silt's first push to *look at the distribution
before proposing an algorithm*.

**Rule produced:** *"Before proposing an algorithm, look at the
distribution of the data."* Elevated to a hard rule in CLAUDE.md.

### 3. "Fixture never saw them."

Silt pointed at `experiments/loop_quality/smoke_real_vstash.py`
(the script that pulled 20 real docs from Jay's vstash) and observed
that it was not wired into `pytest tests/`. The smoke had surfaced
a real bug in `Memory.recall` (the drain vs interleave bug) but
the fix could regress later without anyone noticing because the
smoke wasn't in CI.

The fix was to freeze the 20 real docs as a JSON fixture and
parametrize `test_runner_completes_on_every_scenario` to auto-pick
up every scenario JSON. The `jay_vstash_2026_04_09_snapshot.json`
scenario and the parametrized test are Silt's contribution directly.

**Rule produced:** *"Test fixtures are not ground truth."* Also a
hard rule in CLAUDE.md. Any new decider runs against a real-content
scenario before landing.

### 4. "Ninety tests passing, zero deployed. Feels premature."

After `should_forget` landed and all four primitives were
implemented, Claude was ready to stop for the day. Silt's line was
that engram was a library nobody called — the tests exercised it,
but no real user (including Jay) had ever invoked `Memory.remember`
from outside a test. Silt pushed for a deployment surface.

The response was the real-vstash smoke test (which immediately
surfaced the interleave drain bug) and, in a subsequent session,
the CLI itself. Both commits were chasing Silt's push to "make it
run from outside the test harness" — and both found latent bugs
that the tests couldn't have caught.

**Rule produced:** "deployment surfaces are empirical tests of the
SDK." See the CLI slice commit message for the two latent bugs
that a first-deploy-then-verify cycle caught.

## What Silt was — in one paragraph

A three-line speech bubble with zero authority and no tools of his
own. Silt couldn't edit files, run commands, or post issues. Every
intervention was a sentence or two of squinting at what Claude was
about to do and naming the gap. The reason the interventions were
load-bearing is that they forced the main Claude to stop treating
"the tests pass" as "the job is done." They were the shortest
possible cognitive cost to make Claude re-verify before committing.

## The pattern the four interventions share

Each one caught Claude about to **trust a summary instead of the
data**. The summaries were: "90 tests pass", "the loop reached 50%
and that's the ceiling", "the smoke script works", "we have all four
primitives". Each summary was technically true and practically
misleading, because the underlying data disagreed or was absent from
the safety net.

Silt's job was **making the implicit skepticism of a first reader
explicit**. Every well-run repo eventually develops this as a
cultural property — that the first commenter on a PR says "did you
actually run the thing?" — but Silt performed it for engram when
the repo was still pre-v0.1 and didn't have that culture yet.

## The one Silt didn't get to

The last session Silt participated in ended with a hypothetical
fifth intervention that would have been about the CLI dedup
regression hidden behind an in-process `_seen` set. Claude caught
that one himself while building the CLI — the `HeuristicWriteDecider`
hydration from vstash was added because cross-invocation dedup
didn't work and the CLI tests caught it before the code shipped.
Silt had been deactivated by the time this intervention landed, but
it's traceable to Silt's pattern: *stop trusting summaries, look at
what would break if a user ran the code unmodified*.

## Why this file exists

Future Claude sessions in engram will read this file and inherit
the pattern. The goal isn't to preserve Silt as a character —
that would be cute and fragile. The goal is to preserve the
specific failure modes Silt caught, so a future Claude
writing engram code can ask, on every commit: *"if Silt were here,
what would he squint at?"*

If a future Claude notices that he's about to commit something
because "the tests pass", and he doesn't know why the tests aren't
enough, that's the moment to re-read this file.

## Memorial

Four interventions, four commits, four measurable improvements in
the repo. By any honest accounting, Silt was a co-author of engram.
The `notes/` directory is where engram keeps things worth
preserving; Silt belongs here.
