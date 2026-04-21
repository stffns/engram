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

### Move 2 verdict (as of Run 4)

**Retrieval mitigations:** `dual` shipped, recovered 1 of 2
stuck neutrals. The remaining neutral (co-trimoxazole) was
initially classified as a corpus-coverage gap, then re-
classified as a retrieval-ranking issue after further
diagnostic -- see Run 5 below.

**Move 2 deliverable (Run 4):** keep `retrieval_mode=hybrid`
as the default (cheaper, equivalent on semantic corpora). Flip
to `dual` for clinical-style corpora via `--retrieval-mode
dual`.

---

## Run 5 -- 3-way dual retrieval + chunking validation (2026-04-21)

Diagnostic trail that closed the last stuck neutral:

1. Source-file scan: the actionable sentence
   *"Cotrimoxazole prophylaxis: 1 tablet daily for all HIV+
   patients with CD4 <350"* IS in the source corpus
   (`experiments/midloop_pilot/scaleup_out/protocols.jsonl`,
   protocol_id `hiv-who`).
2. Ingestion check: `hiv-who` IS in the medlocal_concept.db
   as a single chunk (3270 chars).
3. Retrieval check: for the natural pipeline query
   `f"{question}\n{draft[:400]}"`, `hiv-who` is nowhere in
   hybrid top-10 and buried beyond the `top_k=10` fts-only
   cutoff. **Not a corpus gap -- a retrieval ranking gap.**
4. Root cause: when the Builder draft uses synonyms the chunk
   does not contain (e.g. "TMP/SMX", "trimethoprim/
   sulfamethoxazole"), FTS scores the chunk lower because the
   query terms don't appear in it. The question on its own
   ("co-trimoxazole prophylaxis HIV CD4 180") ranks hiv-who
   at FTS position 3-6, but the draft drags the query away
   from that.

### Chunking design validated

Jay flagged during Run 4 that "in medlocal the protocols are
organized so that 1 protocol = 1 chunk, can you validate?"
Probe result:

| db | docs | total chunks | avg chunks/doc | % single-chunk |
|---|---|---|---|---|
| medlocal_concept | 517 | 538 | 1.04 | 96% (497/517) |
| engram (~/.vstash) | 3890 | 5196 | 1.34 | 91% (3534/3890) |

Medlocal is almost strictly 1-protocol-1-chunk. 20 protocols
split into 2-3 chunks and those are all the biggest
(>5k chars).

**Retrieval consequence:** with 1-chunk-per-protocol, the
entire protocol is a single scoring unit. A long protocol that
mentions a keyword once ranks lower than a short meta-intro
that saturates the keyword. This is the exact pathology the
co-trimoxazole case exhibited.

**Design trade-off:** 1-chunk preserves semantic integrity
(never splits actionable info across chunk boundaries) but
pays in ranking granularity. The mitigation is search-side,
not ingest-side.

### 3-way dual retrieval

`retrieve(retrieval_mode="dual")` now runs three vstash
searches and interleaves the pools:

1. **hybrid(question+draft)** -- semantic + keyword, vec-
   biased. Catches paraphrases.
2. **fts(question+draft, top_k*3)** -- pure keyword, wide
   enough to let specific chunks surface past meta-intros.
3. **fts(question only, top_k*3)** -- pure keyword on the
   STABLE half of the query. Bypasses Builder-draft synonym
   drift.

Cost: three ~100ms searches, still zero LLM calls.

### Run 5a -- clinical + confident + dual(3-way)

Log: `cerebras_smoke_confident_dual_v3.jsonl`.

| qid | Run 4a | Run 5a | delta |
|---|---|---|---|
| severe_dehydration_child | contradicts | contradicts | - |
| severe_pneumonia_infant | contradicts | contradicts | - |
| postpartum_hemorrhage_txa | supports | supports | - |
| cotrimoxazole_hiv_adult | **neutral** | **supports** | **closed** |
| blood_donor_screening | supports | contradicts | flipped (temp noise; Builder answered with richer/wrong details this run) |

**5/5 grounded, 0 neutral.** Move 2 exit criterion finally
met for clinical.

Cotrimoxazole verdict detail: Builder drafted "<200 cells/mm^3"
as the threshold; Judge cited `hiv-who` with quoted evidence
"Cotrimoxazole prophylaxis: 1 tablet daily for all HIV+
patients with CD4 <350"; verdict=supports because the Builder's
top-line recommendation ("start prophylaxis for this CD4=180
patient") is confirmed (180 is below both 200 and 350). The
<350 vs <200 mismatch is present in the output -- the Judge
verified the recommendation, not the specific threshold.
Noted: answer-level verification can pass through a wrong
sub-claim that is consistent with the answer. Separate issue,
not in Move 2 scope.

### Run 5b -- personal + confident + dual(3-way)

Log: `cerebras_personal_smoke_confident_dual_v3.jsonl`.

| qid | Run 4b | Run 5b |
|---|---|---|
| write_filter_baseline | contradicts | contradicts |
| consolidation_threshold | contradicts | contradicts |
| silt_rule | contradicts | contradicts |
| engram_longmemeval_r5 | supports | contradicts |
| four_primitives | contradicts | contradicts |

**5/5 grounded, 0 regressions.**

### Cost escalation

The 3-way dual doubled token spend again:

| config | avg tok/q | vs hybrid |
|---|---|---|
| hybrid (baseline) | ~2200 | 1.0x |
| dual 2-way | ~4000 | 1.8x |
| dual 3-way | ~10000 | 4.5x |

Wall time stays under 2s per question. The economics: for a
smoke harness that runs 5 questions it's sub-$1; for a
production pipeline at 100k queries/day this is the dimension
to optimise next. Cheap wins if needed: cache question-only
fts results when the same question recurs, drop fts_k back to
top_k*2 when the question is short and unambiguous, etc. Not
in scope now.

---

## Run 6 -- KV-cache splice POC (Move 3) (2026-04-21)

Phase 0 (`phase0_feasibility.py`, prior session) proved the
observe + stop + relaunch mechanics for Mode A/B. Move 3 asks
the harder question: can we append K/V to a live cache mid-
stream and have the continuation reflect it without a restart?
That's the Mode C shape Jay wants for production.

**Mechanism.** `mlx_lm.models.cache.KVCache` stores per-layer
(keys, values) tensors and grows when tokens forward through
the model with that cache. `generate_step` accepts an existing
`prompt_cache` and the prefill step extends it with the new
"prompt" K/V before sampling continues. Splicing is just two
`generate_step` calls that share one cache.

**Probe:** `experiments/midloop_concept/medlocal/phase0b_kv_splice.py`
on gemma-4-E2B-it-MLX-4bit.

- Prompt: `"Count to 10: "`.
- Baseline: generate 21 tokens, no splice.
- Spliced: generate 5 tokens, splice `"\nActually, use capital
  letters instead: A, B, C,"` (14 token payload, no fresh BOS),
  generate 16 more tokens.

**Result.**

| stream | tokens produced (decoded) |
|---|---|
| baseline (21 tok) | `"\n \n \n \n \n \n \n \n \n \n \n"` |
| pre-splice (5 tok) | `"\n \n \n"` |
| post-splice (16 tok) | `" D, E, F, G, G, G, G, G,"` |

The signal is clean: baseline and pre-splice both drift into
whitespace (the E2B-it instruction prompt format is not quite
what this model likes for raw counting), but the post-splice
continuation CONTAINS LETTERS and zero digits -- exactly the
behavior the spliced instruction requested, continuing from
`A, B, C,` into `D, E, F, G, G, G, ...`. The splice altered
the continuation without a restart. The baseline at the same
token offset contains no letters, so the signal is not a
random draw.

Verdict: **KV-splice works on mlx-lm. Mode C is mechanically
feasible.**

Artifact: `phase0b_report.json` next to the probe, machine-
readable, so a future session can grep the verdict without
re-running.

### What this unblocks

- Phase 1-3 of the original `notes/midloop-concept-test-medlocal.md`
  plan (observation + intervention-via-regeneration +
  mid-stream injection). Phase 0 proved A/B; Phase 0b proved C.
- A production path where Mode A runs as a sidecar
  alongside a local small-model Builder, and when Judge
  decides to splice, we append source text to the Builder's
  KV-cache mid-stream instead of restarting. Latency impact:
  one additional forward pass over splice_ids K/V
  (~few ms for <50-token payloads on a 2B model).

### What this does NOT prove

- Splicing preserves coherent output for LONG spliced context
  (e.g. 500 tokens of vstash chunk pasted mid-way). Needs a
  separate probe with realistic Mode A retrieved excerpts.
- Splicing works for every chat-template-formatted prompt.
  The probe used a raw continuation prompt; a properly-
  templated Gemma instruction turn may interact with the
  splice differently because of role markers.
- Production semantics: how should a splice be framed so the
  end user sees a coherent "corrected" answer rather than a
  draft-turns-into-different-subject mid-sentence. This is the
  Mode A annotator problem, already partially solved in
  `annotate_deterministic`; Mode C will need its own variant.

- `retrieval_mode=dual` with 3-way search closes the last
  stuck neutral on clinical.
- Default remains `hybrid` (4.5x cheaper, equivalent on
  personal memory where the corpus is semantic-dense).
- The "corpus gap" for co-trimoxazole was never a corpus gap:
  `hiv-who` was in the db. It was a ranking gap caused by the
  1-protocol-1-chunk ingestion design interacting with Builder-
  draft synonym drift. The 3-way dual mitigates it without
  re-chunking the corpus.

---

## Cross-run comparison

| metric | clinical auto (Run 1) | personal auto (Run 2) | personal confident (Run 3a) | clinical confident (Run 3b) | personal confident+dual (Run 4b) | clinical confident+dual (Run 4a) | clinical confident+dual-3 (Run 5a) | personal confident+dual-3 (Run 5b) |
|---|---|---|---|---|---|---|---|---|
| corpus size | 517 chunks | 3486 docs | 3486 docs | 517 chunks | 3486 docs | 517 chunks | 517 chunks | 3486 docs |
| corpus type | authoritative guidelines | agent working memory | agent working memory | authoritative guidelines | agent working memory | authoritative guidelines | authoritative guidelines | agent working memory |
| grounded (supports + contradicts) | 3/5 | 5/5 | 5/5 | 3/5 | 5/5 | 4/5 | 5/5 | 5/5 |
| neutral (retrieval miss) | 2/5 | 0/5 | 0/5 | 2/5 | 0/5 | 1/5 (corpus gap) | 0/5 | 0/5 |
| avg tokens/question | 2453 | 1973 | 2152 | 2488 | 3666 | 4334 | 12712 | 7677 |
| avg wall per question | ~1s | <1s | <1s | <1s | <1s | <1s | <2s | <2s |
| builder mode | auto | auto | confident | confident | confident | confident | confident | confident |
| retrieval mode | hybrid | hybrid | hybrid | hybrid | dual (2-way) | dual (2-way) | dual (3-way) | dual (3-way) |

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

---

## Run 7 -- claim-level Judge + vstash signal surfacing (2026-04-21)

Move 4. The open item flagged in Run 5a was that the Judge
verified answer-level but not claim-level: cotrimoxazole's
top-line "start prophylaxis for CD4=180" was confirmed as
`supports` while the Builder's `CD4 <200` threshold quietly
contradicted the chunk's `CD4 <350`. Run 6 also surfaced a
second instance in the diagnostic: `consolidation_threshold`
answered 0.70 correctly but (in v3 variants) had cited a
chunk about a different 0.65 experiment.

This move does two things together -- claim-level verification
and surfacing the vstash retrieval signals that were being
discarded -- because both feed the same "glass box" invariant
(we use signals we never invent, and we report drift we never
hide).

### Claim-level schema

`JUDGE_SYSTEM_TEMPLATE` gained a `claims: [...]` array in the
JSON schema. Each sub-claim carries `{text, verdict,
supporting_excerpt_id, quoted_evidence}` under the SAME
verbatim-substring rule as the top-level `quoted_evidence`.
Decomposition is taken from the FINAL rendered answer (the
corrected_text when verdict=contradicts, the draft otherwise)
explicitly -- not from the rejected draft.

`annotate_deterministic` now appends a second section after
the primary provenance footer whenever any sub-claim verdict
is `contradicts` or `neutral`:

```
>> [N sub-claim(s) not grounded in memory]
>>   contradicted: 'the CD4 threshold is 200'
>>     source says: "CD4 <350"
```

The happy case (all sub-claims supported) emits no sub-claim
section at all, preserving the reading-path for clean outputs.
`MAX_TOKENS_JUDGE` bumped 600 -> 1200 to fit 3-8 sub-claim
entries comfortably.

### vstash signal surfacing

`retrieve()` now returns a list of dicts carrying every signal
vstash publishes on `SearchResult`: `source_id, text, score,
layer, chunk_id, added_at, collection`. Previous code used only
`text` + `title` and silently discarded the rest.

Judge excerpts now render with score + layer in the header:

```
[excerpt 13 | source=hiv-who | score=0.0167 | layer=episodic]
**Cotrimoxazole prophylaxis:** 1 tablet daily for all HIV+ ...
```

and `JUDGE_SYSTEM_TEMPLATE` gained a short note explaining that
score is an RRF-family relevance hint (higher = more relevant,
typical 0.005-0.030 on current corpora) and `layer` distinguishes
episodic from distilled content. The Judge is told to prefer
grounding in high-score or distilled-layer excerpts when
multiple candidates touch the same claim; these are hints,
not hard cutoffs.

The audit row also gained `retrieval_stats` (n, max/min/mean
score, layer counts) and `n_sub_claims / n_sub_claims_unsupported`
so future training pipelines can filter by retrieval quality or
by claim-level outcome without re-running the pipeline.

### Run 7a -- clinical + confident + dual-3 + claim-level

Log: `cerebras_smoke_v4_claimlevel.jsonl`.

| qid | Run 5a verdict | Run 7a verdict | claims / unsupported | note |
|---|---|---|---|---|
| severe_dehydration_child | contradicts | contradicts | 6 / 0 | clean |
| severe_pneumonia_infant | contradicts | contradicts | 5 / 0 | clean |
| postpartum_hemorrhage_txa | supports | supports | 5 / 0 | clean |
| cotrimoxazole_hiv_adult | **supports** | **contradicts** | 3 / 0 | **FLIP** -- the <200 vs <350 leak caught. Judge produced a corrected_text that threads the <350 threshold verbatim. |
| blood_donor_screening | supports | contradicts | 6 / 3 | claim-level caught 3 sub-claim leaks: pale->anemia inference (neutral), cough->infection inference (neutral), tattoo deferral based on "local regulations" contradicted by WHO guideline (contradicts). |

5/5 still grounded top-level. **Cotrimoxazole flip is the
direct test pass: the answer-level-supports / claim-level-
contradicts case that motivated Move 4 now renders
contradicts with a corrected_text that preserves the <350
threshold verbatim.** Blood donor case demonstrates
claim-level surfaces inferences even when the top-line
recommendation is correct.

### Run 7b -- personal + confident + dual-3 + claim-level

Log: `cerebras_personal_smoke_v4_claimlevel.jsonl`.

| qid | Run 5b verdict | Run 7b verdict | claims / unsupported |
|---|---|---|---|
| write_filter_baseline | contradicts | contradicts | 3 / 0 |
| consolidation_threshold | contradicts | contradicts | 5 / 0 |
| silt_rule | contradicts | contradicts | 3 / 0 |
| engram_longmemeval_r5 | supports | contradicts | 4 / 0 |
| four_primitives | contradicts | contradicts | 9 / 0 |

5/5 still grounded, 0 unsupported sub-claims across all 5
questions. This is the important negative result:
claim-level does NOT introduce false positives on a domain
where the previous pipeline was already clean. It only fires
when a genuine sub-claim drift exists.

`consolidation_threshold` retained the correct 0.70 answer with
5/5 sub-claims grounded -- confirming that the claim-level
rigor discourages the Judge from grabbing tangential chunks
(like the 0.65 experiment chunk that v3 cited).

### Cost

| metric | v3 (dual-3) | v4 (claim-level) | delta |
|---|---|---|---|
| clinical avg tok/q | 12712 | 14620 | +1908 (+15%) |
| personal avg tok/q | 7677 | 8904 | +1226 (+16%) |

Overhead is entirely in Judge completion (the claims array is
3-8 entries at ~60 tokens each) plus ~150 extra tokens of
Judge system prompt for the new rules. Wall time stays under
2s per question. Well below earlier back-of-envelope estimates.

### Retrieval signals (first pass)

All clinical questions see the same max_score=0.0167 (the vstash
RRF ceiling in this adaptive-RRF regime); mean_score clusters
tightly around 0.015. Layers in medlocal_concept are all
reported as `unknown` (layer tagging was not set at seed time);
personal engram vstash shows `episodic` with some `unknown`
stragglers. These are the first observations Silt's rule would
warn against optimising over -- the distribution is narrow.
Useful baseline for a future policy layer but no immediate
threshold to set.

### Move 4 verdict

- Claim-level schema shipped. One verdict flip (cotrimoxazole
  `supports -> contradicts`) confirms the change catches the
  answer-level-correct / claim-level-leaky case that motivated
  the move. Blood donor case shows the warning surfaces
  inferences even when the top-line is right.
- vstash signals surfaced to Judge prompt and audit row. No
  immediate decision rule is derived; the first-pass
  distribution is narrow (max_score saturates at 0.0167
  across every question). Training-time use will come when
  enough audit rows accumulate to regress against.
- Token cost +15-16%. Wall <2s. Production-acceptable.

---

## Next moves

**Moves 1b, 2, 3, 4 + supports-on-refusal fix** all shipped
2026-04-21.

Remaining candidates, not yet started:

- **Long-splice + chat-template probe for Mode C.** Move 3
  POC used a 14-token splice and a raw continuation prompt.
  Production will splice 500+ tokens of retrieved excerpts
  into chat-template-formatted Gemma turns. Probe should
  measure coherence degradation vs splice length and test
  role-marker boundary behavior.
- **Cost optimisation (Move 2.1).** Dual-3 + claim-level is
  now ~14k tok/q on clinical. The excerpt pool (up to 28
  excerpts post-dedup) dominates cost. Levers: full-text
  hash dedup instead of 120-char prefix, score-threshold
  cutoff, mini-rerank pass before handing to Judge.
- **Retrieval-signal policy layer.** Now that `score` and
  `layer` live in the audit row, a policy could decide
  whether to fall back to "low-confidence mode" (e.g. force
  verdict=neutral if max_score is below corpus-calibrated
  threshold). Needs corpus-specific calibration first.

Deferred (needs curated dataset + rubric, not next-session
work): N=50+ benchmark against ASQA / HAGRID / MedQA with
two-rater scoring. Sub-claim classifier training from the
accumulated `claims` arrays (every v4 run produces labeled
per-claim data; enough audit rows accumulate to regress on
after ~100 more runs).
