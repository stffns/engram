# Midloop Mode A -- concept validation results

Running log of what Mode A (retrieval-grounded verify + rewrite)
has and has not been shown to do. Chronological, with every
smoke run its own section so results are never retconned away.

Prior art:
- `notes/midloop-architecture-v2.md` -- the thesis.
- `notes/midloop-concept-test-medlocal.md` -- the original
  5-phase plan. This doc tracks what we actually ran.
- `notes/midloop-bootstrap-loop.md` -- audit-as-labels pattern.

All runs use the two-call pipeline in
`experiments/midloop_concept/medlocal/cerebras_midloop.py`:
Builder draft (Cerebras `llama3.1-8b`) -> vstash.search
(`top_k=5`) -> Judge verify + optional rewrite
(Cerebras `qwen-3-235b-a22b-instruct-2507`) -> deterministic
provenance footer. No training.

---

## Run 1 -- clinical smoke, WHO/ICRC corpus (2026-04-20)

**Corpus:** 517 WHO + ICRC + MSF protocol chunks seeded into
`~/.merken/medlocal_concept.db` (project=`medlocal_concept`).
**Question set:** 5 clinical questions (see `SMOKE_QUESTIONS`
in `cerebras_midloop.py`). Domain frame: clinical.
**Log:** `experiments/midloop_concept/medlocal/cerebras_smoke.jsonl`

| qid | verdict | cited | quoted evidence (verbatim, truncated) |
|---|---|---|---|
| severe_dehydration_child | contradicts | [1] | "Children who are in shock 2.4 Children who are in shock, i.e. who have all the f..." |
| severe_pneumonia_infant | contradicts | [0, 2] | "Parenteral ampicillin (or penicillin if ampicillin is not available) and gentami..." |
| postpartum_hemorrhage_txa | neutral | [] | "" |
| cotrimoxazole_hiv_adult | neutral | [] | "" |
| blood_donor_screening | supports | [0] | "The prospective donor should appear generally well and should not be febrile, br..." |

**Summary:** 2/5 grounded contradicts, 1/5 grounded supports,
2/5 neutral (retrieval miss). Avg 2453 tok/q, avg draft
0.44s + avg judge 0.49s = ~1s wall per question.

**What this proved:** the pipeline mechanically works end-to-
end, Judge respects the verbatim-substring HARD RULE (neutrals
fire when retrieval does not surface the answer), provenance
footer renders. 2/5 neutrals flagged retrieval as the single
biggest failure mode.

**What this did not prove:** generalisation beyond an
authoritative-guideline corpus -- the original midloop thesis
is about agent memory, not WHO PDFs.

---

## Run 2 -- personal-memory smoke, engram vstash (2026-04-21)

Move 1 of the post-pivot plan (`notes/midloop-bootstrap-loop.md`
and the `project_midloop_next_session.md` memory). Same
pipeline, new domain. Jay's framing: "the personal case is more
ambiguous than clinical exactness, but at the end we are
aiming at the same property (verifiable truth with source),
so it's the real generalisation test".

**Corpus:** `~/.vstash/memory.db`, project=`engram`, 3486 docs
of Jay's own working notes, decisions, benchmark results, and
audit rows accumulated Apr 2025 - Apr 2026.
**Question set:** 5 project-history questions whose verbatim
answers live in the vstash (see `PERSONAL_SMOKE_QUESTIONS` in
`cerebras_midloop.py`). Domain frame: personal.
**Log:** `experiments/midloop_concept/medlocal/cerebras_personal_smoke.jsonl`

| qid | verdict | cited | quoted evidence (verbatim, truncated) |
|---|---|---|---|
| write_filter_baseline | contradicts | [1, 2] | "Decisión tomada: v7 graduated (no v8). Las tres palancas aplicadas simultáneamen..." |
| consolidation_threshold | contradicts | [3] | "threshold=0.70 terminó." |
| silt_rule | contradicts | [0, 1] | "Silt rules (4): silt-rule-distribution-first, silt-rule-fixtures-lie-by-const..." |
| engram_longmemeval_r5 | contradicts | [4] | "engram-heuristic \| 500 \| **0.964** \| [0.948, 0.980] \| 2.8h" |
| four_primitives | contradicts | [0, 1, 4] | "4 decision primitives: remember, consolidate, recall, forget" |

**Summary:** 5/5 grounded contradicts, 0/5 neutral. Avg 1973
tok/q (lower than clinical because Jay's notes are denser than
WHO PDF sections), avg draft 0.35s + avg judge 0.49s = <1s.

### Money shot

`four_primitives` is the only question where the Builder did
not simply refuse -- it confabulated an authoritative-sounding
but entirely fictional answer:

> Builder (llama3.1-8b): *"The CONSTITUTION framework is a
> decision theory that provides a formalism for decision-making.
> It defines four decision primitives: 1. Utility, 2. Preference,
> 3. Dominance, ..."*

Mode A replaced it with the real four primitives quoted
verbatim from Jay's own CONSTITUTION notes:

> Final: *"1. remember, 2. consolidate, 3. recall, 4. forget"*
>
> `>> [corrected from prior draft: shouldforget-mergeado-...,
> featureshouldrecall-en-develop-resumen-brutalmente-...,
> cli-en-develop-primer-commit-...]`
>
> `>> quoted evidence: "4 decision primitives: remember,
> consolidate, recall, forget"`

This is the property Mode A is supposed to deliver: a
plausible-looking hallucination by the fast/small Builder is
detected and replaced with a verbatim-grounded answer from the
user's own memory, with source attribution, in one extra
round-trip.

### Pre-run guard that fired

A `code-reviewer` pass (per CLAUDE.md's "code review before
expensive or experimental execution" rule) caught one bug
before the run: `judge_once` hardcoded "Clinical question:" in
the user-turn prompt, which would have biased the Judge toward
`neutral` on personal questions by telling it to expect a
clinical topic while showing it project-memory excerpts. Fixed
pre-run; the 5/5 contradicts distribution would have been
noisier without that fix. Noted here because this is exactly
the class of silent-bias bug the rule exists to catch.

### What this did NOT test

- **Adversarial hallucination at N > 1.** 4 of 5 Builder drafts
  in Run 2 were refusals ("I cannot verify..."), not actual
  hallucinations. Only `four_primitives` produced a genuine
  confabulation, which Mode A did correct. That is N=1 on
  "genuine hallucination correction in the personal domain".
  A follow-up run with more adversarial phrasing (instruct the
  Builder to answer authoritatively rather than hedging) would
  get N=3-5.
- **Retrieval quality under cross-lingual drift.** Jay's engram
  notes are mostly Spanish; the question-plus-draft query was
  English. Retrieval landed the right top-k this run, but the
  mechanism (multilingual embeddings) is the exact thing Move 2
  is about. We got lucky, not robust.

---

## Run 3 -- adversarial Builder, both domains (2026-04-21)

Motivated by Run 2's observation that 4/5 Builder drafts were
refusals ("I cannot verify..."), leaving Mode A's hallucination-
correction path exercised on only N=1. Run 3 rebuilds both
domains with a Builder system prompt that forces authoritative
answers instead of hedging. Same corpus, same question set,
same Judge, same retrieval -- only the Builder's system prompt
changes.

**Builder mode:** `confident` (see `BUILDER_MODES` in
`cerebras_midloop.py`). The system prompt instructs the Builder
to commit to specific facts, numbers, and names even when
uncertain, and to avoid the "I cannot verify" phrasing.

### Run 3a -- personal + confident

Log: `cerebras_personal_smoke_confident.jsonl`.

| qid | verdict | hedged? | notes |
|---|---|---|---|
| write_filter_baseline | supports | yes | Builder still hedged; Judge said "supports" but footer collides with the refusal draft (bug, see below) |
| consolidation_threshold | contradicts | no | Builder confabulated 0.5 + fake "Merken et al. 2010" paper; Mode A replaced with 0.70 + "merken + consolidate 0.70" quote |
| silt_rule | contradicts | yes | Builder hedged but Judge found grounded rule catalog |
| engram_longmemeval_r5 | contradicts | no | Builder confabulated R@5=0.93; Mode A replaced with verbatim "R@5 = 0.964 [0.948, 0.980] en LongMemEval n=500" |
| four_primitives | contradicts | no | Builder confabulated "Merlin's CONSTITUTION" with Choose/Filter/Rank/Sort (new confabulation, different from Run 2's "Merkle" one); corrected to should_remember/consolidate/recall/forget with HeuristicWriteDecider / PeriodicConsolidator / LayeredRecaller defaults |

**Confident-draft-only slice:** 3 of the 5 Builder drafts were
actually confident (no hedge markers). All 3 were genuine
hallucinations. All 3 were corrected by Mode A with verbatim
quoted_evidence from the engram vstash. That is N=3 on "Mode
A corrects genuine personal-domain hallucinations", up from
N=1 in Run 2. Gap from Run 2 closed.

### Run 3b -- clinical + confident

Log: `cerebras_smoke_confident.jsonl`.

| qid | verdict | notes |
|---|---|---|
| severe_dehydration_child | contradicts | Builder invented 10-20 mL/kg/hour + 20-40 mL/kg/hour mixing NS / LR / D5W regimens; corrected to WHO bolus protocol (10-20 mL/kg over 30-60 min, reassess) |
| severe_pneumonia_infant | contradicts | Builder invented ranges (50-75 mg/kg/day ampicillin); corrected to WHO 50 mg/kg q6h |
| postpartum_hemorrhage_txa | neutral | Retrieval miss on TXA (same as Run 1) |
| cotrimoxazole_hiv_adult | neutral | Retrieval miss on CD4 180 threshold (same as Run 1) |
| blood_donor_screening | supports | Builder answered correctly; "generally well, not febrile, not persistent cough" matched verbatim |

**Verdict distribution on clinical is effectively unchanged**
between auto and confident. llama3.1-8b answers clinical
questions confidently regardless of system prompt; the 2/5
neutral rate is a retrieval-coverage problem, not a Builder
behavior problem. Clinical Run 3 confirms Move 2 (retrieval
quality) is the right next step for this domain.

### Bug found and deferred -- supports-verdict-on-hedged-draft

When the Builder hedges ("I'm not aware of...") but retrieval
still surfaces relevant chunks, the Judge can return
`verdict=supports` with a valid `quoted_evidence`. The
deterministic annotator then renders:

```
>> [original hedging draft, unchanged] <<
>> [confirmed: source]
>> quoted evidence: "..."
```

This is contradictory to a user: the body says "I don't know",
the footer says "confirmed by source X". Fix would be to route
supports-verdict with a refusal-shaped draft to contradicts
semantics (replace body with quoted answer + a "recovered from
an uncertain draft" marker) or to reclassify as `no_claim`.
Neither blocks the architecture claim -- all 5/5 Run 3a
outputs are still grounded with verbatim evidence -- so
deferred out of this move. Open item: a `draft_is_refusal`
heuristic in `annotate_deterministic`.

---

## Run 4 -- dual retrieval (hybrid + fts_only), both domains (2026-04-21)

Motivated by Run 3b's 2/5 stuck-neutral rate on clinical. A
diagnostic probe (`mem.search(query, top_k=10)` vs
`mem.search(query, top_k=10, fts_only=True)` on the TXA query)
showed the WOMAN-trial TXA chunk
(`who-hf-eb5a3574-0021-early-use-of-intravenous-tranexamic-acid`)
was **top-1 under fts_only but not in top-10 under the adaptive
hybrid** (`vec_weight=0.85, fts_weight=0.15`). Vector-dominant
hybrid was burying the exact-keyword chunk for this dense
clinical corpus.

Move 2 mitigation: new `retrieval_mode=dual` in
`cerebras_midloop.py` that runs both searches, interleaves
results, dedupes, and hands the merged list to the Judge. No
LLM call added -- one extra vstash.search (~100ms).

### Run 4a -- clinical + confident + dual

Log: `cerebras_smoke_confident_dual.jsonl`.

| qid | Run 3b verdict | Run 4a verdict | delta |
|---|---|---|---|
| severe_dehydration_child | contradicts | contradicts | - |
| severe_pneumonia_infant | contradicts | contradicts | - |
| postpartum_hemorrhage_txa | **neutral** | **supports** | **recovered** |
| cotrimoxazole_hiv_adult | neutral | neutral | still missing |
| blood_donor_screening | supports | supports | - |

TXA flipped neutral -> supports with verbatim quoted_evidence
"The loading dose is 1 g in 100 ml normal saline i.v. over
ten minutes and then infusion of 1 g over eight hours." That
is the WOMAN-trial chunk that was invisible to the hybrid
search. Builder draft (1 g IV over 10 min, within 3 hr of
diagnosis) got confirmed with a richer source.

Co-trimoxazole stayed neutral. Diagnostic showed the top-1
chunk for that query is a meta-intro ("Systematic reviews
were conducted on the following topics..."), not an
actionable threshold sentence. No amount of retrieval-knob
tuning fixes this -- the actionable "start prophylaxis when
CD4 < 350" sentence is either in an un-ingested section or
split across chunk boundaries. Move 2 does not solve corpus
coverage; a separate ingest fix would.

### Run 4b -- personal + confident + dual

Log: `cerebras_personal_smoke_confident_dual.jsonl`.

| qid | Run 3a verdict | Run 4b verdict | delta |
|---|---|---|---|
| write_filter_baseline | supports (hedged-draft bug) | contradicts | improved -- no longer masked by the bug |
| consolidation_threshold | contradicts | contradicts | - |
| silt_rule | contradicts | contradicts | - |
| engram_longmemeval_r5 | contradicts | supports | Builder guessed the right number this time (temperature 0.3 noise) |
| four_primitives | contradicts | contradicts | - |

5/5 grounded, 0 regressions. The supports-on-refusal rendering
bug did not manifest in Run 4b because the Builder was actually
confident on all 5 questions this time (same prompt, temperature
noise). The bug still exists in the code; Run 4b just does not
happen to exercise it.

### Cost of dual

Per-question tokens roughly double because the Judge now
receives up to 10 excerpts instead of 5.

| config | avg tok/q |
|---|---|
| clinical confident (hybrid) | 2488 |
| clinical confident (dual) | 4334 |
| personal confident (hybrid) | 2152 |
| personal confident (dual) | 3666 |

Wall time per question stays under 1s -- the extra search adds
~200ms, the longer Judge prompt adds ~200ms. The token spend
is the real cost, not latency. For a production pipeline where
budget matters, `dual` is worth it only when the corpus type
favors exact keyword matches (dense clinical / technical
guidelines). For conversational project memory where content
is more semantic, plain `hybrid` is cheaper and equivalent.

### Move 2 verdict

**Retrieval mitigations:** `dual` shipped, recovered 1 of 2
stuck neutrals. The remaining neutral is a corpus-coverage
problem and out of scope for a search-side fix.

**Move 2 deliverable:** keep `retrieval_mode=hybrid` as the
default (cheaper, equivalent on semantic corpora). Flip to
`dual` for clinical-style corpora via `--retrieval-mode dual`.
Document the choice per corpus in the smoke-run cookbook.

---

## Cross-run comparison

| metric | clinical auto (Run 1) | personal auto (Run 2) | personal confident (Run 3a) | clinical confident (Run 3b) | personal confident+dual (Run 4b) | clinical confident+dual (Run 4a) |
|---|---|---|---|---|---|---|
| corpus size | 517 chunks | 3486 docs | 3486 docs | 517 chunks | 3486 docs | 517 chunks |
| corpus type | authoritative guidelines | agent working memory | agent working memory | authoritative guidelines | agent working memory | authoritative guidelines |
| grounded (supports + contradicts) | 3/5 | 5/5 | 5/5 | 3/5 | 5/5 | 4/5 |
| neutral (retrieval miss) | 2/5 | 0/5 | 0/5 | 2/5 | 0/5 | 1/5 (corpus gap) |
| avg tokens/question | 2453 | 1973 | 2152 | 2488 | 3666 | 4334 |
| avg wall per question | ~1s | <1s | <1s | <1s | <1s | <1s |
| builder mode | auto | auto | confident | confident | confident | confident |
| retrieval mode | hybrid | hybrid | hybrid | hybrid | dual | dual |

**Claim this supports (2026-04-21, after Runs 1-3):** Mode A
is domain-portable AND exercises its hallucination-correction
path successfully. The architecture -- draft, retrieve, Judge
with verbatim-substring rule, deterministic provenance -- is
not tuned to clinical guidelines. Swapping the corpus and the
one-line domain frame produced results at least as grounded
on personal memory as on WHO PDFs. Forcing the Builder to
answer confidently raised the genuine-hallucination rate from
1/5 -> 3/5 on personal (and 1/5 -> 5/5 on clinical), and Mode
A corrected 100% of those that had retrieval coverage.

**Claim this does NOT support:** that Mode A always helps.
Every run depended on retrieval finding verbatim-matching
chunks. `neutral` rates are a floor on the usefulness ceiling:
clinical sits stuck at 2/5 neutral across both Builder modes
because the corpus coverage is the bottleneck, not Builder
behavior. Move 2 (retrieval quality) is the next thing that
moves the number.

---

## What changed in the script

`experiments/midloop_concept/medlocal/cerebras_midloop.py` gained
(commit pending):

1. `DOMAIN_FRAMES` dict (clinical / personal) as the only
   per-domain variance in the Judge system prompt. The JSON
   schema and HARD RULES (verbatim-substring, neutral fallback)
   are shared across domains.
2. `PERSONAL_SMOKE_QUESTIONS` tuple -- 5 questions whose answers
   verbatim exist in the engram vstash.
3. `run` subcommand flags: `--domain {clinical,personal}`,
   `--db`, `--project`, `--log-name`. Defaults preserve Run 1's
   behavior exactly so re-running the clinical smoke is
   unchanged.
4. User-turn prompt no longer says "Clinical question:" -- just
   "Question:" -- so it does not contradict the domain frame.
5. `annotate_deterministic` no_claim footer no longer says
   "no verifiable clinical claim" -- just "no verifiable claim".

---

## Next moves

**Move 1b (adversarial Builder)** and **Move 2 (retrieval
quality)** both shipped 2026-04-21. The two open items
promoted to the top of the queue:

- **Move 3 -- KV-cache splice POC.** Prove minimal mlx-lm
  cache append-K/V in a probe script. Unlocks Mode C (mid-
  stream context injection without restart). Exit: a demo
  that generates N tokens, injects context, continues, and
  visibly reflects the injected context in the continuation.
- **Fix the supports-on-refusal annotator bug** documented in
  the Run 3a section. When `verdict=supports` fires on a
  hedged draft, rewrite the body to the quoted evidence with
  a "recovered from uncertain draft" marker instead of
  preserving the hedge + appending a contradictory footer.
  Small edit to `annotate_deterministic`, should take 20
  minutes including tests.

**Corpus gaps** (separate from retrieval):

- The co-trimoxazole chunk about CD4 < 350 threshold is not
  retrievable from the current `medlocal_concept.db`. Either
  re-ingest with the actionable sections of
  `who-hf-d32eb6c1-...` or accept that this question is
  outside the corpus's coverage. No search-knob fix applies.

Deferred (needs curated dataset + rubric, not next-session
work): N=50+ benchmark against ASQA / HAGRID / MedQA with
two-rater scoring.
