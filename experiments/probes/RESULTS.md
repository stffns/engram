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

## Probe B. multi-hop synthesis from briefs+episodic (2026-04-19)

**Question:** is a Knowledge Graph layer (CONSTITUTION §6, gated)
justified? If the LLM cannot synthesize multi-hop answers from the
existing briefs+episodic context, KG might pay off. If it can, KG
is overengineering.

**Method:** 10 hand-crafted multi-hop queries about engram project
history. Each requires chaining >=2 facts from different briefs /
events. For each: retrieve top brief_k=3 briefs + top_k=5 episodic
via `merken recall-briefs`, format as context, send to Gemini 2.0
Flash. Dump (query, context, answer) for manual judgment.

**Decision rule:**
- correct >= 70% -> KG overengineering, skip.
- < 50% -> KG justified.
- 50-70% -> KG might help; second-order question.

**Results (manual judgment at n=10):**

| # | query topic | verdict |
|---|-------------|---------|
| Q1 | H2 + hook bug connection | **WRONG** -- "H2 not discarded" (brief was written before today's H2 rejection) |
| Q2 | v6 -> v7 markdown FPR fix | HONEST MISS (info not in briefs) |
| Q3 | v4 -> v7 training data evolution | HONEST MISS |
| Q4 | H8 rejection + H2 common conclusion | **WRONG** -- both rejected today; brief is stale |
| Q5 | training-signal vs capacity | PARTIAL -- correct framing, numbers stale |
| Q6 | hook bug + Probe A | PARTIAL/MISS -- explains a DIFFERENT past bug |
| Q7 | LoCoMo H16/H17 + retrieval filter | **HIT** -- clean multi-hop synthesis |
| Q8 | brief_v1 breakthrough | MISS -- ironic; the briefs themselves are evidence |
| Q9 | H2/H2b/H2c relationships | HONEST MISS (all today) |
| Q10 | "wait for 300+ DEC labels" justification | HONEST MISS (today) |

**Aggregate:** HIT 1, PARTIAL 1.5, HONEST_MISS 4, WRONG 3, MISS 0.5
-> ~15% correct, 30% wrong (hallucinated).

**Confounder:** 5 of 10 queries (Q1, Q4, Q6, Q9, Q10) ask about
events from TODAY's session (H2/H2b/H2c/H8 rejections, hook bug
fix, Probe A finding) which are NOT YET in briefs. Briefs reflect
state up to yesterday. The test is partly unfair on those.

**Filtered to queries where info SHOULD be in the store (Q2, Q3,
Q5, Q7, Q8):** 1 HIT (Q7), 2 PARTIAL/MISS (Q5, Q8), 2 HONEST MISS
(Q2, Q3 -- info exists in project history but briefs don't yet
cover it).

### The bigger finding -- briefs go stale fast

The most important finding of Probe B is NOT about multi-hop
synthesis. It is about **brief freshness**:

- Q1 / Q4 / Q5 reference yesterday's briefs whose contents have
  been SUPERSEDED by today's experimental results, but the briefs
  do not know that.
- Gemini answered Q1 with "H2 contrastive loss is the active
  training path; next session will use it" -- which was true 24h
  ago and is now wrong.
- The LLM has no temporal awareness; it trusts the brief as
  current state.

**Implication:** brief generation must be MORE FREQUENT, not just
more comprehensive. SessionEnd hook helps with this (1 brief
generation pass per session vs PreCompact's "rarely"). It also
implies briefs should carry temporal markers ("as of 2026-04-18;
verify against most recent episodic events") and stale briefs
should be tombstoned by a future consolidation cycle when their
claims are contradicted.

This is a separate concern from connector / KG architecture.
Document for future work.

### Verdict for Knowledge Graph EV

**NOT JUSTIFIED YET.** Same structural answer as Probe A: the
bottleneck is brief volume + freshness, not synthesis architecture.

**The one positive evidence point (Q7)** shows the LLM CAN do
multi-hop synthesis cleanly when given correct + complete context.
This argues against KG: the synthesis capacity is already there
for free via Gemini.

**KG would only become justified IF:** with 50+ briefs covering
broad project history AND fresh briefs from SessionEnd-driven
accumulation, we still see Q1/Q4-style failures despite the
relevant facts being in retrieved context. We are nowhere near
that test condition.

**Re-evaluate KG when:** brief count >= 50 AND a re-run of Probe B
on FRESH briefs still scores below 50% correct. Until then,
investing 1-2 weeks in a KG layer is overengineering.

### Concerning side-finding: hallucination rate

Gemini gave WRONG answers (not "I don't know") on 3 of 10 queries.
That's a 30% hallucination rate when context is stale or
incomplete. The HONEST_MISS rate (4 of 10) is encouraging -- it
sometimes refuses -- but the WRONG rate is high enough to be a
production concern.

If brief-based synthesis ever lands in user-facing code, the
prompt needs an explicit "if any fact contradicts another, prefer
the most recent timestamp; if you cannot verify with the context,
say 'I cannot determine'" guardrail. Stricter than the current
"do not speculate beyond context" instruction, which clearly
isn't strong enough.

Files:
- `probe_b_multihop.py` -- query runner.
- `probe_b_results.json` -- raw output with full Gemini answers
  and the retrieved context per query.
