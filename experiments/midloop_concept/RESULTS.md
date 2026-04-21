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

## Cross-run comparison

| metric | clinical (Run 1) | personal (Run 2) |
|---|---|---|
| corpus size | 517 chunks | 3486 docs |
| corpus type | authoritative guidelines | agent working memory |
| grounded (supports + contradicts) | 3/5 | 5/5 |
| neutral (retrieval miss) | 2/5 | 0/5 |
| avg tokens/question | 2453 | 1973 |
| avg wall per question | ~1s | <1s |
| judge model | qwen-3-235b | qwen-3-235b |
| builder model | llama3.1-8b | llama3.1-8b |

**Claim this supports (2026-04-21):** Mode A is domain-portable.
The architecture -- draft, retrieve, Judge with verbatim-
substring rule, deterministic provenance -- is not tuned to
clinical guidelines. Swapping the corpus and the one-line
domain frame in the Judge system prompt produced results at
least as grounded on personal memory as on WHO PDFs.

**Claim this does NOT support:** that Mode A always helps.
Both runs depended on retrieval finding verbatim-matching
chunks. `neutral` rates are a floor on the usefulness ceiling.
Move 2 (retrieval quality) is the next thing that moves the
number.

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

## Next moves (not yet run)

From `project_midloop_next_session.md`, in signal-per-effort
order:

- **Move 2 -- retrieval quality.** Run the same 5-question
  personal smoke with `top_k=8`, Judge-derived query
  extraction, and MMR reranking (lambda 0.5-0.7). Pick the
  variant that keeps the 5/5 grounded rate AND improves on
  Run 1's 2/5 neutral rate. Bake into the default. Exit:
  >= 4/5 grounded on clinical, 5/5 on personal.
- **Move 3 -- KV-cache splice POC.** Prove minimal mlx-lm
  cache append-K/V in a probe script. Unlocks Mode C (mid-
  stream context injection without restart). Exit: a demo
  that generates N tokens, injects context, continues, and
  visibly reflects the injected context in the continuation.

Deferred (needs curated dataset + rubric, not next-session
work): N=50+ benchmark against ASQA / HAGRID / MedQA with
two-rater scoring.
