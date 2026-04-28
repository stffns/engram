> **Status: Blocked.** This issue opens *only after* both #38 (directional residual geometry) and #39 (semantic markers via few-shot prototypes) pass their individual success criteria.  If either upstream issue fails, this issue is closed without execution and the conclusion is drafted into the merken paper as a negative finding.

## Motivation

Issues #38 and #39 test two orthogonal approaches to the same underlying problem -- detecting role contamination in consolidation clusters:

- **#38 (residual geometry)** detects role-contamination at the cluster level *without naming the role*.
- **#39 (semantic markers)** names the role explicitly per event *without detecting cluster-level mismatch*.

This issue tests the integration: a combined decision rule that uses both signals as inputs to merken's consolidation gate.  The combined rule, not either component in isolation, is the production deliverable.  The two upstream issues are *validation studies* of the rule's components.

## Combined decision rule (the deliverable)

```python
ROLE_CONFIDENCE_THRESHOLD = ...  # tuned on held-out tuning set
ENTROPY_THRESHOLD = ...          # tuned on held-out tuning set

class ConsolidationDecision(Enum):
    MATERIALIZE = "materialize"  # consolidate as a retrieval fact
    BRIEF       = "brief"        # produce a brief instead, no consolidation
    ABSTAIN     = "abstain"      # defer to next pass with more events

def should_consolidate_cluster(cluster) -> ConsolidationDecision:
    role        = classify_role_via_prototypes(cluster.events)   # from #39
    entropy_pr  = participation_ratio(cluster.embeddings)        # from #38

    high_conf = role.confidence >= ROLE_CONFIDENCE_THRESHOLD
    low_pr    = entropy_pr      <= ENTROPY_THRESHOLD

    # 2x2 decision table; see "Decision-table rationale" below.
    if  high_conf and  low_pr:  decision = ConsolidationDecision.MATERIALIZE
    if  high_conf and not low_pr: decision = ConsolidationDecision.BRIEF
    if not high_conf and  low_pr: decision = ConsolidationDecision.ABSTAIN  # operator choice
    if not high_conf and not low_pr: decision = ConsolidationDecision.BRIEF

    audit_log.emit({
        "cluster_id":         cluster.id,
        "role_label":         role.label,
        "role_confidence":    role.confidence,
        "participation_ratio": entropy_pr,
        "decision":           decision,
        "thresholds":         {"role": ROLE_CONFIDENCE_THRESHOLD, "pr": ENTROPY_THRESHOLD},
    })
    return decision
```

The audit log makes every consolidation decision observable, debuggable, and revertible if thresholds turn out to need adjustment.

## Decision-table rationale

The original AND-rule (`high_conf AND low_pr -> MATERIALIZE; else BRIEF`) collapses two distinct signals:

- **Low role confidence** = "the few-shot classifier is unsure what role this cluster is" -- a *classifier uncertainty* signal.
- **High participation ratio** = "the residuals span multiple directions" -- a *cluster-content* signal.

These are not the same.  A cluster of one consistent but unknown-to-the-classifier role (low conf, low PR) is a different case from a cluster of mixed roles (low conf, high PR or high conf, high PR).  The 2x2 table makes the distinction explicit:

| | Low PR (geometry: clean) | High PR (geometry: mixed) |
|---|---|---|
| **High role conf**  | MATERIALIZE                                    | BRIEF (genuine role mixing)         |
| **Low role conf**   | ABSTAIN (defer) or MATERIALIZE (conservative)  | BRIEF (worst case, both signals bad)|

The `Low conf + Low PR` cell is the operator-chosen branch; both options are defensible:

- **ABSTAIN** = wait for more events, hope the classifier gets confident later.  Safer; reduces materialization rate.
- **MATERIALIZE** = trust the geometry; ship it.  More aggressive; higher recall but risk of mis-labeled materialization.

Pre-register the choice before running the experiment.  The merken paper draft assumes ABSTAIN for the headline numbers and reports MATERIALIZE as a sensitivity analysis.

## AND vs OR is a product decision, not a technical one

The AND in the pseudo-code (`high_conf AND low_pr -> MATERIALIZE`) is conservative -- more BRIEFs, fewer materializations, higher precision per materialization.  An OR-formulation (`high_conf OR low_pr -> MATERIALIZE`) would be liberal -- more materializations, higher recall.

The choice depends on merken's operational cost asymmetry:

- **Cost of brief-when-should-have-materialized**: a fact that *should* be retrievable as a consolidated entity is instead exposed as a brief, which costs LLM calls + lower retrieval quality on that topic.
- **Cost of materialize-when-should-have-briefed**: a contaminated cluster is consolidated and pollutes the retrieval index with a confused fact.

If the second cost dominates (the "polluted index" failure is more expensive than the "brief cost"), keep AND.  If the first dominates (brief generation is the bottleneck), consider OR.  This is a product decision; the issue must surface it explicitly.

**Pre-registered choice:** AND, on the assumption that index pollution is the dominant operational cost in current merken deployments.  An OR variant is reported as sensitivity analysis if AND fails the success criteria below.

## Validation experiment

Once issues #38 and #39 pass individually, this experiment measures whether the combined rule outperforms either signal alone on the same known-ground-truth scenarios.

### Baselines

1. **Consolidate everything.** Materialize every cluster regardless of signal.  Upper bound on recall, naive precision.
2. **Geometry-only.** Materialize iff `participation_ratio <= ENTROPY_THRESHOLD`.  Tests #38 in isolation as a gate.
3. **Markers-only.** Materialize iff `role.confidence >= ROLE_CONFIDENCE_THRESHOLD`.  Tests #39 in isolation as a gate.
4. **Proposed: combined rule.** The 2x2 decision table above.

### Metrics

For each baseline + the proposed rule:

- **Precision** = `clean clusters materialized / total clusters materialized`.  How often a materialized cluster is actually clean.
- **Recall** = `clean clusters materialized / total clean clusters`.  How often a clean cluster is correctly materialized.
- **F1** at chosen operating point.
- **Brief rate** = `clusters routed to BRIEF / total clusters`.  Operational cost driver.
- **Abstain rate** = `clusters routed to ABSTAIN / total clusters`.  Only meaningful for the combined rule.

### Threshold tuning protocol (pre-registered, anti-overfit)

Split the known-ground-truth scenarios into three disjoint subsets, stratified by ground-truth class:

1. **Tuning set** (~50% of data): used only for sweeping ROLE_CONFIDENCE_THRESHOLD ∈ {0.40, 0.45, ..., 0.80} × ENTROPY_THRESHOLD ∈ {0.10, 0.20, ..., 0.60}.  Pick the (T1, T2) pair that maximizes F1 of the combined rule on tuning data.
2. **Sensitivity set** (~25% of data): heatmap of precision/recall over the full (T1, T2) grid; identifies whether the gate is robust (wide region of acceptable thresholds) or fragile (single sharp peak).  Reported alongside the headline numbers.
3. **Test set** (~25% of data): sealed.  Used **once** to report final precision / recall / F1 of all four baselines + the combined rule.  No re-tuning permitted.  This is the headline number.

### Pre-registered success criteria for the combined rule

The combined rule is declared successful if it satisfies *either*:

- **Precision-priority gate**: combined precision ≥ max(geometry-only precision, markers-only precision) + 5pp **at matched recall** to the better single-component baseline, on the test set.

  *or*

- **Pareto-domination gate**: at every recall level achieved by either single-component baseline, the combined rule achieves higher precision; equivalently, the combined ROC curve dominates both component curves.

If neither holds, the combined rule is declared a negative result.  See "Pre-committed negative result framing" below.

Bootstrap 95% CI on precision and recall (1000 resamples) required.

### Signal-correlation diagnostic (mandatory side measurement)

Report the empirical correlation between the two signals across all clusters in the test set:

- Frequency of "geometry says clean / markers say mixed" (and vice versa).
- If discrepancy frequency > 5%, the two signals are measurably independent and the combined rule is genuinely more informative than the better single component.
- If discrepancy frequency < 1%, the two signals are nearly redundant and the combined rule cannot improve substantially over the better single component.  This is itself a finding; report it.

## Pre-committed negative result framing

If the combined rule fails to satisfy either success gate, the merken paper is updated with the following framing rather than silently dropping the result:

> "Combining the entropy and marker signals does not improve the consolidation gate beyond the better single component (precision X, recall Y).  The signal-correlation analysis (Z% discrepancy frequency) suggests the two approaches are measuring overlapping rather than complementary aspects of role contamination."

This finding tightens the merken paper rather than weakening it: it tells the reader that role contamination has a single underlying axis observable from two angles, not two independent axes.

## Out of scope

- Production rollout (A/B testing, telemetry-driven threshold revision, gradual deployment).  This is follow-up issue #P.
- Replacing either component with a different implementation (e.g., a trained classifier instead of few-shot markers).  Issues #38 and #39 validate the chosen component implementations; this issue assumes them.
- Multi-cluster decision rules (e.g., "BRIEF cluster A but only because cluster B contains the same role").  Single-cluster gate only.
- Threshold adaptation over time (drift detection, online recalibration).

## Deliverable

- `merken/consolidation/combined_gate.py` -- production implementation of `should_consolidate_cluster` and the audit-log emitter.
- `experiments/combined_gate/validation.py` -- the four-baseline experiment with the pre-registered protocol.
- `experiments/combined_gate/threshold_sensitivity.py` -- heatmap script.
- `notes/2026-XX-XX-combined-gate-validation.md` -- writeup with:
  - Tuning-set selected thresholds.
  - Test-set precision / recall / F1 / brief-rate for all four baselines + combined.
  - Threshold sensitivity heatmap.
  - Signal-correlation diagnostic.
  - Bootstrap 95% CIs on every reported metric.
  - Go/no-go recommendation against success criteria.
- Decision rule wired into the merken consolidation path *behind a feature flag*.  Feature flag stays off by default until production rollout (issue #P).

## References

- Issue #38 (directional residual geometry).
- Issue #39 (semantic markers via few-shot prototypes).
- merken paper section 7.6 (role-aware consolidation as future work).
- vstash v2 paper section 5.6 / 6 (gated domain-adaptation loop as a parallel example of "validate components, then ship the integrated decision rule under a feature flag").
