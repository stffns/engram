"""Type-A (derived) vs Type-B (authoritative) memory tagging.

Two ontologies coexist in merken's storage:

- **Type A (derived):** episodics, briefs, traces, outcomes. Mutable,
  decays, consolidatable, tombstoneable. No stable ground truth.
- **Type B (authoritative):** protocols, fixed corpora, laws,
  technical docs, specs. Immutable, no decay, no consolidation,
  no arbitrary tombstone -- only versioned replacement.

The distinction is expressed as a **tag convention** on the event:
``source:<value>``. Mutative operations (`consolidate`, `forget`,
future `decay`) check this tag and skip Type B events.

**Fail-closed semantics:** the operative predicate
``is_safely_mutable`` returns True ONLY when we are confident the
event is Type A. If the source tag is present but the value is
unrecognized (typo, format change, deprecated category), the event
is treated as IF it were Type B and the mutation is skipped. The
asymmetric cost:

- False negative (unknown-tag Type A skipped): one extra event
  in the store; can be recovered via re-tagging + re-running the
  mutation. Recoverable.
- False positive (mistakenly-not-tagged Type B mutated): irreversible
  loss of authoritative content (e.g. dropping a clinical protocol
  from a clean store). Catastrophic in the MedLocal use case.

Legacy (untagged) events stay Type A by default so existing flows
continue to work. Only events that EXPLICITLY tag a source value get
the safety check.
"""

from __future__ import annotations

# Source values that mark an event as derived/Type-A and therefore
# safe to mutate (consolidate, forget, decay). Add new derived
# sources HERE -- if you forget, the event will be treated as Type B
# (fail-closed).
KNOWN_DERIVED: frozenset[str] = frozenset({
    "derived",        # default for explicit derived marking
    "session",        # in-session UserPromptSubmit / SessionEnd ingest
    "transcript",     # transcript replay (PreCompact, backfill)
    "precompact",     # legacy hook tag
    "backfill",       # legacy backfill tag
    "auto",           # any auto-derived
    "agent",          # agent-derived events
    "user",           # user-typed remember
})

# Source values that mark an event as authoritative/Type-B. Listed
# explicitly for documentation only -- the predicate uses the
# negation of KNOWN_DERIVED, so adding to this set is unnecessary
# for safety. Adding here just helps readability.
KNOWN_AUTHORITATIVE: frozenset[str] = frozenset({
    "authoritative",
    "protocol",
    "protocol_who",
    "protocol_msf",
    "spec",
    "law",
    "literature",
    "reference",
})


def _iter_source_values(tags: str | None) -> list[str]:
    """Pull every `source:<value>` off a tags string."""
    if not tags:
        return []
    out = []
    for tag in tags.split(","):
        tag = tag.strip()
        if tag.startswith("source:"):
            out.append(tag[len("source:"):].strip().lower())
    return out


def is_safely_mutable(tags: str | None) -> bool:
    """True iff this doc can be safely mutated by consolidate/forget/decay.

    Returns:
      - True when ``tags`` is None / empty (legacy default = Type A).
      - True when every ``source:<value>`` tag is in KNOWN_DERIVED.
      - False when any ``source:<value>`` is in KNOWN_AUTHORITATIVE,
        unknown, or malformed.
    """
    sources = _iter_source_values(tags)
    if not sources:
        return True  # legacy / untagged
    return all(value in KNOWN_DERIVED for value in sources)


def is_authoritative(tags: str | None) -> bool:
    """True iff any ``source:<value>`` tag is in KNOWN_AUTHORITATIVE.

    Convenience for explicit authoritative checks (e.g. citation
    preference in retrieval). Distinct from ``is_safely_mutable``:
    an event with an unknown source value is BOTH "not safely
    mutable" AND "not explicitly authoritative" -- the unknown
    case is conservatively treated as immutable for mutations,
    but should not get authoritative-citation preference.
    """
    return any(
        value in KNOWN_AUTHORITATIVE
        for value in _iter_source_values(tags)
    )
