## Motivation

Pre-trained embedders like BGE-small encode rich semantic information from large training corpora.  The hypothesis is that this includes enough signal to distinguish broad semantic roles (`DECISION`, `INVESTIGATION`, `OBSERVATION`, `PREFERENCE`) without training a dedicated classifier.

If the hypothesis holds, the embedder itself becomes the role classifier via few-shot prototype matching, and we can produce role markers as metadata at write time without external models, without fine-tuning, and without human-labeled training data.

This is *complementary* to the directional residual geometry approach in #38: #38 detects role-contamination at the cluster level without naming the role; markers name the role explicitly per event.  See "Integration with sibling issue" below for the combined-gate hypothesis if both succeed.

## Role taxonomy

The four-role taxonomy (`DECISION`, `INVESTIGATION`, `OBSERVATION`, `PREFERENCE`) is taken from merken paper section X.Y *(fill in exact section before publishing)*.  Pre-committing to this taxonomy means the experiment evaluates the embedder's ability to recover *this specific* axis of meaning, not arbitrary role decomposition.  If the four roles turn out to overlap heavily in BGE-small's embedding space, that itself is a finding -- it tells us the taxonomy is at the wrong granularity for embedding-only classification.

## Proposed experiment

Two phases.

### Phase 1: prototype-based role classification

1. Define `K` prototype examples per role.  Primary K = 7; report accuracy as a function of K ∈ {1, 3, 5, 7, 10} to characterize the saturation curve.  Total prototype count at primary: 4 × 7 = 28.
2. **Prototype provenance (anti-leak).** Prototypes are written from a style guide *before* looking at the evaluation events.  Source pool: events from a scenario *disjoint from* the `knowledge_update` evaluation scenarios -- ideally a held-out subset reserved for prototype authoring.  Prototypes are committed to `prototypes.json` with the source scenario id; reviewers can verify disjointness from the eval set.
3. Embed each prototype with the active model (BGE-small-en-v1.5).
4. For each event in the `knowledge_update` scenarios with role ground truth, embed it and compute cosine similarity against each role's prototypes.  Assign role = argmax of mean similarity per role.
5. **Confidence reporting (mandatory).** Alongside the argmax label, record:
   - top-1 similarity
   - margin = top-1 - top-2 similarity (per role mean)
   - margin distribution (median, IQR) per ground-truth class
   - Optional reject-option threshold τ; report accuracy on confident-only (margin > τ) vs all events
6. **Metrics (mandatory).** Report all of:
   - Macro-F1
   - Per-class accuracy (must clear minimum threshold, see Success Criteria)
   - Confusion matrix (4×4)
   - Class balance in the eval set

### Phase 2: marker-conditioned embedding (only if Phase 1 succeeds, see gating below)

1. For events with classified role, prepend a textual prefix `[role: decision]` (lowercase, exact format) to the event text before embedding.
2. **Critical baseline: random-prefix control.** A separate cohort of events receives a random prefix `[role: <RANDOM_TAG>]` drawn from a held-out vocabulary.  This isolates whether the effect is from semantic role signal or from prefix-induced clustering.  If random-prefix contamination drops by a comparable amount to role-prefix, the effect is artifact, not signal.
3. Re-run the consolidation evaluation pipeline.
4. Operationalization of "matched recall": hold global recall@K constant for K ∈ {5, 10}; measure contamination rate at that K for vanilla, role-prefix, and random-prefix arms.

## Success criteria (decided before looking at results)

### Phase 1 gate

| Macro-F1 + per-class min accuracy | Verdict |
|-----------------------------------|---------|
| Macro-F1 ≥ 0.85 AND per-class accuracy ≥ 0.75 (all 4 classes) | strong signal, proceed to Phase 2 with role-prefix as primary |
| Macro-F1 0.70 - 0.85 OR one class < 0.75 with others ≥ 0.85 | partial signal, proceed to Phase 2 reporting "marker-assisted" not "marker-only" |
| Macro-F1 < 0.70 | no go on Phase 2; embedder lacks the functional role signal we hoped for; markers must come from upstream metadata (commit type, ticket field, audit trail type) instead |

Note on the plain-accuracy threshold from the original draft (0.85 over the 4-class problem): we replaced it with macro-F1 because the eval set is unlikely to be class-balanced and macro-F1 cannot be gamed by neglecting a minority class.

### Phase 2 gate (pre-registered direction)

Predicted directions (recorded **before** running the experiment):

| Cohort                         | Expected contamination@matched-recall |
|--------------------------------|---------------------------------------|
| vanilla (no prefix)            | baseline (current merken numbers)     |
| role-prefix (this experiment)  | reduced by ≥ 30% relative vs vanilla |
| random-prefix (control)        | reduced by < 10% relative vs vanilla |

Phase 2 succeeds if **role-prefix contamination drops ≥ 30% relative** to vanilla, AND **the role-prefix - random-prefix gap is at least 20 percentage points**.  The two-condition gate prevents declaring victory on prefix-induced clustering artifacts.

Bootstrap 95% CI required on each contamination measurement (1000 resamples).

## Why few-shot, not training

Three reasons:

1. **Zero training cost**; experiment runs in hours, not days.
2. **Generalizes across domains** by changing prototypes only; no per-domain fine-tune required.
3. **Auditable**; the prototype set is a small, inspectable JSON artifact that any reviewer can read.

If Phase 1 succeeds, comparing few-shot accuracy against a trained classifier is a follow-up issue; this issue tests whether the simpler approach is sufficient.

## Out of scope

- Training a dedicated role classifier.
- Implementing this in the production `consolidate()` path.
- Comparing against zero-shot LLM classification (e.g., asking GPT to label events as decision/investigation).
- Expanding the role taxonomy beyond the four roles named above.

## Integration with sibling issue (combined-gate hypothesis)

The directional residual geometry issue and this one test orthogonal approaches to the same problem:

- **Sibling issue** (residual geometry): detects role-contamination at the cluster level *without naming the role*.
- **This issue** (markers): names the role explicitly per event *without detecting cluster-level mismatch*.

If both issues independently produce partial-or-better signal (AUC ≥ 0.65 for the sibling, macro-F1 ≥ 0.70 for this one), there is a third deliverable not in scope of either issue but worth a follow-up:

> **Combined gate**: classify each event with markers, then check whether the marker labels within a cluster agree.  Disagreement is a stronger contamination signal than either geometry alone or markers alone, because it is a *labelled* disagreement (we know which roles are present, not just that the cluster is heterogeneous).

The expected AUC of the combined gate is bounded below by `max(sibling_AUC, this_issue_AUC)` if the errors of the two approaches are not perfectly correlated, and exceeds it if they are partially independent.  Whether the errors are independent is itself an empirical question and the third follow-up issue.

## Deliverable

- `experiments/role_markers/few_shot_classifier.py` -- Phase 1 implementation.
- `experiments/role_markers/marker_conditioned_eval.py` -- Phase 2 implementation (only run if Phase 1 gate passes).
- `experiments/role_markers/prototypes.json` -- canonical examples per role with source scenario id, K=7 minimum, with a `version` field for future updates.
- `experiments/role_markers/prototype_style_guide.md` -- the style guide used to author prototypes blind from the eval set.
- Markdown writeup under `notes/`:
  - Macro-F1, per-class accuracy, confusion matrix, margin distribution, K-saturation curve.
  - Phase 2 (if executed): contamination by cohort with bootstrap CIs, role-prefix vs random-prefix delta.
  - Go/no-go recommendation against the success criteria above.

Output files:

- `phase1_classification.csv` (one row per event, ground-truth label, predicted label, top-K similarities, margin).
- `phase1_metrics.json` (macro-F1, per-class accuracy, confusion matrix, K-saturation table).
- `phase2_contamination.csv` (one row per event in each cohort, cluster id, contamination contribution).
- `phase2_metrics.json` (per-cohort contamination at matched recall, bootstrap CIs, gate decision).

## Related issues

- #38 (sibling): directional residual geometry as a role-contamination signal.  Tests an orthogonal approach to the same underlying problem.
- #40 (depends on this and #38): combined entropy + marker gate for consolidation decisions.  Opens only if both #38 and #39 pass their individual gates.

## References

- merken paper section 7.6 (role-aware classification as future work).
- merken paper section X.Y *(fill in)* (origin of the four-role taxonomy).
- **Snell, Swersky, Zemel 2017** -- "Prototypical Networks for Few-shot Learning."  Academic foundation of the prototype-distance approach.
- **Schick, Schütze 2020** -- "Pattern-Exploiting Training" / "Exploiting Cloze Questions for Few-Shot Text Classification."  Evidence that prefix-conditioning shifts pretrained representations toward task-relevant subspaces.
- **SetFit** (Tunstall et al. 2022, sentence-transformers' few-shot library) -- implementation reference.
- vstash adaptive IDF as a parallel example of "use information the system already has, do not train new components."
