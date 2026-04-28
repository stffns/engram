# Prototype style guide -- 4-role taxonomy for #39

This guide is the *single source of truth* for what counts as each
of the four roles in the #39 semantic-markers experiment. Authored
2026-04-28 *before* writing prototypes.json or any eval-set vocabulary.
Pre-registered: any future addition to either pool must conform to
the definitions below; if a candidate event does not clearly fit one
role, it does not enter the pool.

## The four roles

### DECISION

An event where someone (a person, a team, an organization) actively
selected one option over alternatives and committed to it. The
defining property is *commitment to a choice*. Past or perfect tense
is typical. The text typically names the chosen option.

Linguistic markers (not exhaustive): "decided to", "chose", "selected",
"approved", "switched to", "settled on", "went with", "adopted",
"replaced X with Y", "committed to".

Borderline cases:

- A past investigation that *resulted* in a decision: classify by
  the act of committing -- "investigated A vs B and chose A" is a
  decision. "Investigated A vs B; results pending" is investigation.
- A team's standing position absent recent commitment: that is
  PREFERENCE, not DECISION.

### INVESTIGATION

An event where someone is *actively diagnosing or exploring* an open
question. The defining property is *inquiry without conclusion*.
Present or progressive tense is typical. The text typically names
the question being explored.

Linguistic markers: "investigating", "looking into", "trying to
understand", "diagnosing", "exploring", "examining", "checking
whether", "running tests on", "root-causing", "evaluating".

Borderline cases:

- "Investigated A and chose B" -> DECISION (resolved).
- "Investigating B; A is currently in production" -> INVESTIGATION
  (open).
- "Investigation team is exploring options" -> still INVESTIGATION
  unless an actual choice is made.

### OBSERVATION

An event reporting *factual state or what happened*, without
selecting between options or actively inquiring. The defining
property is *passive recording of fact*. Past or present descriptive
tense is typical. The text typically describes what was seen, not
what was chosen or what is being explored.

Linguistic markers: "noticed", "saw", "observed", "found", "data
shows", "metrics indicate", "logs show", "report shows", "X happened",
"X occurred", "X measured at".

Borderline cases:

- "Noticed slow queries; investigating" -> if the *primary content*
  is the observation (slow queries), classify as OBSERVATION; if
  the primary content is the diagnosis intent, INVESTIGATION.
  Default: OBSERVATION wins when the diagnosis intent is incidental.
- "Data shows X; we should switch to Y" -> the recommendation is
  PREFERENCE; if the text is dominated by the data report, OBSERVATION.

### PREFERENCE

An event expressing a *subjective stance, recommendation, or
positional preference* without representing a committed decision or
an active investigation. The defining property is *stated leaning*
without commitment. Present tense and modal verbs are typical.

Linguistic markers: "prefer", "lean toward", "would rather",
"team's preference", "recommend", "advocate for", "in favor of",
"my take is", "personally I'd", "the right move would be".

Borderline cases:

- "Prefer X; switching now" -> DECISION (the switch is committed).
- "Prefer X over Y" without action -> PREFERENCE.
- "Recommended switching to X; team is reviewing" -> PREFERENCE
  (recommendation, not yet committed).

## Disjointness for prototype vs eval pools

The prototype pool (`prototypes.json`) and the eval pool (the
generated scenario from `vocab_eval_*.json`) MUST share zero events
and SHOULD share minimal lexical overlap on topic-specific words.

Concretely:

- Prototypes draw from a small set of "prototype topics"
  (~5-10 topics) that DO NOT appear in the eval pool's topic set.
- Eval pool topics come from a separate vocabulary (~20-30 topics)
  authored AFTER the prototype pool was frozen.
- Both pools follow this style guide, so the *role definitions* are
  shared; only the topics and specific phrasing differ.

The disjointness is enforced by construction (different topic strings
in the two JSONs), not by automated filtering. Reviewers can verify
by inspecting the topic sets in each file.

## Anti-leak commitments

1. The style guide above is frozen on 2026-04-28 EOD. Edits after
   that date must be marked with a separate revision history.

2. Prototypes are written first, committed, then eval-pool events
   are written without re-reading prototypes. The "blind" property
   is informal but the commit timestamps establish ordering.

3. The eval pool's vocab JSONs and the prototype JSON share zero
   exact-string events. This is checked at run time by the
   classifier (it errors out if any prototype text matches any
   eval text exactly).

4. K-saturation curves at K in {1, 3, 5, 7, 10} are computed by
   subsetting the prototype pool deterministically by seed; the
   pool itself is not re-authored.

## V2 redefinition (added 2026-04-28 EOD after #39 Phase 1+P2 NO_GO)

#39 v1 found that BGE-small cannot reliably distinguish DECISION from
PREFERENCE because both express comparative judgment with a named
choice ("Approved X over Y" vs "Prefer X over Y"). The pre-committed
follow-up #39b replaces the DECISION role with **STATE_CHANGE_REPORT**,
designed to share zero lexical structure with PREFERENCE.

### STATE_CHANGE_REPORT (V2 only)

An event that documents a *confirmed change to system state*, framed
as a factual report of what is now in production rather than as the
act of choosing. The defining property is *post-hoc state reporting*.
Past tense and date/version markers are typical. The text typically
names the changed entity and the change itself, NOT the alternatives
considered.

Linguistic markers: "now runs", "is now configured", "has been
running on X since DATE", "deployed to production on DATE",
"migrated to X on DATE", "shipped X", "launched X", "removed X",
"configured X for Y", "decommissioned", "rolled out", "retired",
"version X went live".

What MUST be avoided (these are the patterns that collide with
PREFERENCE):

- "X over Y" comparative structure
- "Approved A; chose A; selected A; settled on A" (focuses on the
  act of choosing rather than the resulting state)
- Modal verbs about future action ("will deploy", "going to switch")
- Recommendation framing ("decided to", "agreed to")

Borderline cases:

- "Migrated to Aurora; old Postgres decommissioned" -> STATE_CHANGE_REPORT
  (state is reported, no alternatives)
- "Switched from Postgres to Aurora" -> NOT STATE_CHANGE_REPORT (uses
  "switched X to Y" comparative; would be DECISION in v1, but in v2
  this category does not exist; such events are EXCLUDED from v2 eval)
- "Production now serves search through Vespa as of 2026-04-15" ->
  STATE_CHANGE_REPORT (clean state report, no comparative).

### Other roles in V2

INVESTIGATION, OBSERVATION, PREFERENCE keep their V1 definitions
unchanged. The disjointness commitments (prototype topics vs eval
topics) and the anti-leak protocol apply identically.

### V1 vs V2 file naming

| File | V1 role set | V2 role set |
|------|-------------|-------------|
| `prototypes.json` (V1) | decision, investigation, observation, preference | -- |
| `prototypes_v2.json` (V2) | -- | state_change_report, investigation, observation, preference |
| `eval.json` (V1) | as above | -- |
| `eval_v2.json` (V2) | -- | as V2 prototypes |
| `few_shot_classifier.py` (V1) | hardcoded ROLES tuple | -- |
| `few_shot_classifier_v2.py` (V2) | -- | hardcoded ROLES tuple |

V2 is a separate experiment artifact; V1 is preserved for #39's
record.

## What this is NOT

- A complete typology of all event roles. Many real events fall
  outside these four (e.g., "scheduled", "deferred", "blocked").
  Out-of-taxonomy events should be EXCLUDED from the eval set,
  not forced into a role.

- A judgment about role *importance*. All four roles are tested
  symmetrically; a finding of "OBSERVATION is hard to detect"
  does not mean OBSERVATION is unimportant.

- A canonical merken role taxonomy. The merken paper section X.Y
  *(reference pending)* may name the same four roles, but this
  experiment commits to these definitions independently. If the
  paper version drifts later, this experiment's prototypes and
  eval pool are versioned (`version` field in JSON) so the
  divergence is auditable.
