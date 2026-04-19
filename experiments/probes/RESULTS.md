# Probes -- empirical EV checks before building

These probes are cheap empirical questions answered ahead of any
implementation work. Each one is designed to either justify or kill
a downstream investment of 1+ days.

## Probe A. brief+episodic recall@5 on real queries (2026-04-19)

**Question:** does merken's existing recall (vstash semantic search +
brief layer when `recall-briefs` is used) actually return relevant
context on real user queries? If recall is already high, training a
nanoGPT-as-connector re-ranker is marginal EV.

**Method:** 20 real queries hand-picked from the engram
should_recall audit log (mix of state-of-project, technical recall,
and substantive lookups). For each, run `merken recall-briefs --top-k 5
--brief-k 3` and judge HIT/PARTIAL/MISS by hand against the dumped
JSON. Manual judgment beats LLM-as-judge at n=20 -- no LLM bias and
~10 min cost.

**Decision rule:**
- recall@5 >= 80% -> connector marginal (skip).
- 50-80% -> connector might help; not a slam dunk.
- < 50% -> connector strongly justified.

**Critical finding before measurement:** first run returned `briefs: 0`
on ALL 20 queries. Investigation showed brief_v1 had NEVER produced
a real brief in the engram store despite hooks being wired for days.
Root cause: PreCompact hook only fires during compaction events
(rare); only 2 consolidate audit rows in entire history, the first
with n_events=1 (skipped, too_few_events).

**Fix (separate from probe):** `~/.claude/hooks/merken-save.sh` +
`~/.claude/settings.json` updated 2026-04-19 to add a `SessionEnd`
hook that calls the same script. SessionEnd fires reliably at every
session close, so briefs now accumulate passively. Diagnostic
logging added at `~/.claude/logs/merken-save.log` so future silent
failures are visible.

After running `merken consolidate --method brief_v1` manually
(produced 5 briefs from 12 episodic events in ~4s), Probe A was
re-run.

**Results:**

| metric | sin briefs | con briefs |
|---|---:|---:|
| HIT (relevant top result) | 5 / 20 | 11 / 20 |
| PARTIAL (related but not direct) | 4 / 20 | 5 / 20 |
| MISS | 6 / 20 (+5 N/A) | 4 / 20 |
| **recall@5 (HIT only)** | **33%** | **55%** |
| **recall@5 (HIT + 0.5*PARTIAL)** | **47%** | **67%** |

brief_v1 contributes +20pp recall, consistent direction with the
+46pp synthetic benchmark scaled to n=20 with only 5 briefs in the
store.

**The 4 remaining MISSes** are all queries whose answer is ABSENT
from the store -- not a ranking problem:

- Q6 "experimentos de la comunidad acerca de memoria agentica"
  -- no briefs about external literature.
- Q9 detail H10/H11/H12 ablations -- no per-hypothesis briefs.
- Q15 "usemos colab para lanzar la siguiente" -- no brief about
  Colab usage.
- Q20 "te quedaste en modelos viejos, 2.5 cuando el ultimo es 3.1"
  -- no brief about Gemini version selection.

A re-ranker cannot invent what is not there. With these 4 dropped
the realistic ceiling on a content-complete subset is ~80%, not
the observed 55-67%.

**Verdict for nanoGPT-as-connector EV:** zone 50-80% (might help,
not a slam dunk). The bottleneck is NOT ranking; it is brief volume
+ training data coverage. Expect another large-step improvement
naturally as briefs accumulate (currently 5; target is 50+ briefs
once SessionEnd hook accumulates over a few sessions).

**Connector remains an open option, not a priority.** Re-evaluate
once brief volume crosses 50 -- if recall@5 still sits below 70%,
connector is justified. If recall hits 80%+ naturally, skip.

Files:
- `probe_a_recall.py` -- query runner.
- `probe_a_results.json` -- raw output (regenerable; not gitignored
  since it's the snapshot at probe time).
