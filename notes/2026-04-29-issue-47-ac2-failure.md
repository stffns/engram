# Issue #47 -- AC#2 production-shape extension test FAIL

Date: 2026-04-29
Branch: `feature/role-classifier-2class-obs-pref` (off develop)
Run dir: `experiments/role_markers/runs/extension_20260429_083338/`
Bundle: `merken/data/role_prototypes_obs_pref.json` (5 obs + 5 pref, V3-prod)

## TL;DR

AC#2 hard-FAILS for the obs_pref taxonomy. The 3-seed STRONG verdict
on V3-prod 2-class data (macro-F1 1.000 +- 0.000) does **not**
generalize to real benchmark chat content at production shape. On
N=200 oracle-labeled samples (100 LoCoMo per-session + 100 LME
per-turn, gpt-oss-120b oracle), pooled in-taxonomy macro-F1 = 0.471
(gate: >= 0.85). Same structural lesson as #46 (V2 SCR 4-class,
0.929 sentence -> 0.794 production -> closed without merge).

AC#1 PASSED cleanly: 3-seed mean macro-F1 1.000 +- 0.000 on V3-prod
2-class subset, CI lower 1.000 (gates: 0.95 / 0.85). The bundled
artifact reproduces the validation result. The failure is between
"V3-prod stylized prototype shape" and "real chat shape," not in the
machinery.

## What was implemented (and what landed)

A full taxonomy gate was prototyped on
`feature/role-classifier-2class-obs-pref`:

- `merken/role_classifier.py` -- taxonomy registry (`_TAXONOMY_REGISTRY`),
  `RoleClassifier.default(taxonomy=...)`, `taxonomy` and
  `is_production_grade` instance properties. Default kept as
  `v2_4class` for back-compat with PR #43.
- `merken/memory.py` -- taxonomy gate at `Memory.remember`: only
  classifiers with `is_production_grade=True` would produce role
  tags; research-only taxonomies emit a one-time warning and no tag.
  This was a cherry-pick of PR #46 + the 2-class gate on top.
- `tests/test_role_classifier.py` + `tests/test_memory_role_classifier_integration.py`
  -- 35/35 role-classifier tests passed; full suite 417/417.
- `experiments/role_markers/extension_test_obs_pref.py` -- AC#2
  extension test runner. FAIL recorded.

After AC#2 failed, the branch was hard-reset to develop and the
machinery was discarded. None of the `merken/` or `tests/` changes
land on develop. What lands on develop instead is research-only:

- `experiments/role_markers/bundles_research_only/role_prototypes_obs_pref.json`
  -- the 2-role bundle copied from V3-prod (5 observation + 5
  preference, production shape). Lives under
  `bundles_research_only/`, not `merken/data/`, because merken does
  not ship a production-grade role classifier.
- `experiments/role_markers/extension_test_obs_pref.py` -- the
  AC#2 reproducer, rewired to the public `RoleClassifier`
  constructor (no taxonomy registry dependency).
- The prior-session probe scripts and writeups
  (`notes/2026-04-28-*`) and this NO_GO writeup.

## AC#2 numbers (full N=200 run)

Sample seed = 42 (deterministic). Oracle = `gpt-oss-120b` on Cerebras,
temperature 0.0, max_tokens 4096 (reasoning-model headroom). Pre-flight
+ `OracleHealthGuard` PASS with zero oracle errors over 200 calls.

| Pool | n in-taxonomy | n neither | n bad oracle | macro-F1 in-taxonomy | rec(obs) | rec(pref) | high-conf neither |
|---|---|---|---|---|---|---|---|
| LoCoMo per-session | 99  | 1  | 0 | **0.290** | 0.389 | 0.250 | 0/1   = 0% |
| LME per-turn       | 75  | 25 | 0 | **0.444** | 0.094 | 0.953 | 0/25  = 0% |
| Pooled             | 174 | 26 | 0 | **0.471** | 0.315 | 0.894 | 0/26  = 0% |

Gate breakdown:

- macro-F1 0.471 >= 0.85 -> **FAIL** (-0.379 absolute).
- "neither" high-confidence (margin > 0.5) fraction 0.000 <= 0.10 ->
  PASS (trivial: confidence margins on real chat are universally
  small, all observed margins < 0.10 in dry-run).
- Overall: **FAIL**.

## Why it failed

The V3-prod prototypes are stylized at the *shape* axis (paragraphs
700-1200 chars) but not at the *content* axis. They were authored
from the V2 style guide:

- observation prototypes -- log/metric/dashboard reports ("Logs show
  ...", "metrics indicate ...", "the data correlates with ...").
- preference prototypes  -- first-person ops opinions ("I lean
  toward ...", "I would rather ...", "I prefer ...").

Real chat content (LoCoMo per-session, LME per-turn) is overwhelmingly
first-person, evaluative, and verbose, but factually about life events
(LoCoMo: catching up on activities; LME: assistant turns giving travel
advice, etc.). BGE-small reads such text as closer to the preference
prototype shape than to the observation prototype shape, so the
classifier biases hard toward "preference" -- preference recall 0.89
pooled, observation recall 0.31. Across LME assistant turns where the
oracle correctly labels long advice as "preference," the classifier
matches at 95%; across LoCoMo factual life-event recaps where the
oracle labels them "observation," the classifier predicts "preference"
2-3x as often as the correct "observation."

This is exactly the failure mode predicted in the
`feedback_no_corpus_manipulation_for_benchmark` and the
`project_role_program_audit_closed_2026_04_28` memory entries:
**matched-shape eval inflates gates**. V3-prod was matched on length
but not on tone or topic distribution.

## Why this kills the production-grade claim

Issue #47 specifies that all 5 acceptance criteria must pass for the
feature to ship. AC#2 was the load-bearing one because it tests
generalization to the actual benchmark units that merken ingests.
Without AC#2, the only defensible claim is "BGE-small + V3-prod
2-class prototypes is STRONG on V3-prod 2-class eval data," which is
a tautology (matched shape, matched topic, matched tone). That is
not a feature; it is a probe outcome.

Confidence margins on this run never exceeded 0.10. So even applying
the AC#3 recommended threshold of 0.10 (uncertain-below) would
suppress every prediction -- the safe production behavior would be
"never tag." That confirms the classifier has no operational utility
on this content, not just bad calibration.

## Recommendation

Close #47 as NO_GO with the same lesson as #46. Add to the role-
program lessons learned (issue #44 monoculture meta) that 2-class
also fails, narrowing the 4-class lesson rather than salvaging it.
Specifically:

1. **PR not opened.** The branch is local-only on
   `feature/role-classifier-2class-obs-pref`. Code passes all 417
   tests but the AC#2 gate kills the production-grade contract.
2. **Close issue #47** with this writeup linked. Tag
   "shape-bias / NO_GO_PRODUCTION (2-class as well)".
3. **Update issue #44** with the lesson: "matched-shape eval
   inflates gates" generalizes from 4-class to 2-class. The right
   eval is oracle-labeled real ingestable text, not stylized
   prototypes that happen to match the shape axis.
4. **Branch fate:** delete or keep as a research artifact. The
   taxonomy gate logic in `merken/memory.py` is reusable if a
   future classifier passes AC#2; the bundle itself is dead weight.
5. **Cost ledger:** ~200 Cerebras gpt-oss-120b calls, ~600K total
   tokens. Single-digit USD. AC#1 + AC#2 are reproducible from the
   committed scripts.

## Files generated by this session

Landing on develop (research-only):

- `experiments/role_markers/bundles_research_only/role_prototypes_obs_pref.json`
  -- the 2-role bundle that was tested.
- `experiments/role_markers/extension_test_obs_pref.py` -- AC#2
  reproducer (FAIL recorded at
  `experiments/role_markers/runs/extension_20260429_083338/extension_test_results.json`,
  gitignored).
- This file.

Plus the prior-session research artifacts that were already on
disk but not yet committed: `experiments/role_geometry/production_shape_validation.py`,
`experiments/role_markers/{eval_v3_production,prototypes_v3_production}.json`,
and the `notes/2026-04-28-*` files. AC#1 PASS is reproduced by
`notes/2026-04-28-role-classifier-subset-probe.py` (no separate
runner needed).

Reverted (NOT landing on develop):

- All `merken/role_classifier.py` and `merken/memory.py` changes.
- The cherry-picked PR #46 integration commit.
- The new tests (the taxonomy gate they tested no longer exists).

## Pre-run code review

`extension_test_obs_pref.py` was reviewed by the code-reviewer
subagent (per CLAUDE.md global "code review before expensive
experiments"). Three issues addressed before the run:

1. LME tiktoken special-token strip mirrored from canonical
   `longmemeval.runner._format_turn` (LoCoMo strip skipped --
   canonical phase2 explicitly notes LoCoMo is plain text).
2. `OracleHealthGuard` integration: pre-flight + per-call recording.
   Run was clean (zero oracle errors).
3. Composite-turn disambiguation added to the oracle prompt.

A 5-per-pool dry-run caught a separate bug: gpt-oss-120b is a
reasoning model and the canonical 200-token oracle budget left it
truncated mid-reasoning (8/10 calls returned no JSON). Bumped to
4096; 200/200 production calls produced parseable JSON.
