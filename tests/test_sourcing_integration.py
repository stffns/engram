"""Integration tests for the Type-A/B fail-closed filter.

Verifies that `Memory.consolidate()` and `Memory.forget()` skip
events tagged Type B, including the fail-closed case where the
source value is unknown / malformed. These are the load-bearing
guarantees for the MedLocal use case (clinical protocols must
survive every consolidation cycle and every forget pass).
"""

from __future__ import annotations

from pathlib import Path

from merken import (
    AlwaysWrite,
    ConsolidationResult,
    Memory,
)
from merken.policies.should_forget import ForgetConsolidated


def _new_memory(tmp_path: Path) -> Memory:
    return Memory(
        project="test_sourcing_integration",
        db=str(tmp_path / "test.db"),
        write_decider=AlwaysWrite(),
    )


def test_consolidate_skips_authoritative_event(tmp_path: Path) -> None:
    """Type B (source:authoritative) does NOT enter consolidate's pool."""
    mem = _new_memory(tmp_path)
    # 11 derived events (above min_events=10) + 1 authoritative
    for i in range(11):
        mem.remember(
            f"derived event number {i} with enough text to count",
            tags="source:session",
        )
    mem.remember(
        "WHO protocol clause: amoxicillin 50 mg/kg/day for pneumonia",
        title="protocol_who_pneumonia_amoxicillin",
        tags="source:authoritative",
    )

    result: ConsolidationResult = mem.consolidate(method="jaccard_v1", force=True)
    # Only the 11 derived events should appear in the consolidation pool.
    # The protocol must NOT be examined.
    assert result.events_examined == 11, (
        f"expected 11 derived events examined, got {result.events_examined} "
        f"-- the authoritative event should be skipped"
    )


def test_consolidate_skips_unknown_source_fail_closed(tmp_path: Path) -> None:
    """Unknown source value must be treated as immutable (fail-closed)."""
    mem = _new_memory(tmp_path)
    for i in range(10):
        mem.remember(
            f"derived event number {i} with enough text to count",
            tags="source:session",
        )
    # A typo or unrecognized source value -- might be a real Type B
    # event tagged poorly. Fail-closed means we DO NOT mutate.
    mem.remember(
        "ambiguously sourced content that we are not sure about",
        title="ambiguous_doc",
        tags="source:autoritative",  # typo on purpose
    )

    result = mem.consolidate(method="jaccard_v1", force=True)
    assert result.events_examined == 10, (
        f"expected 10 derived events examined (typo'd source treated as "
        f"immutable), got {result.events_examined}"
    )


def test_forget_skips_authoritative_event(tmp_path: Path) -> None:
    """Type B events must not be tombstoned by forget()."""
    mem = _new_memory(tmp_path)
    mem.remember(
        "derived event A with enough text for vstash min-length",
        title="derived_a",
        tags="source:session",
    )
    mem.remember(
        "WHO protocol clause: amoxicillin 50 mg/kg/day",
        title="protocol_who_amoxicillin",
        tags="source:authoritative",
    )

    # Use ForgetConsolidated decider; without consolidation
    # nothing has derived_in_facts so nothing is tombstoned anyway.
    # But the asserted behavior is that the protocol is never even
    # examined as a candidate.
    forget_mem = Memory(
        project="test_sourcing_integration",
        db=str(tmp_path / "test.db"),
        write_decider=AlwaysWrite(),
        forget_decider=ForgetConsolidated(),
    )
    result = forget_mem.forget(force=True)
    # force=True means "tombstone every (mutable) episodic event."
    # The protocol must not be in the tombstoned list.
    assert "text://protocol_who_amoxicillin" not in result.tombstoned, (
        f"protocol was tombstoned: {result.tombstoned}"
    )
    # Sanity: derived_a SHOULD be tombstoned (force=True wipes mutables)
    assert any("derived_a" in p for p in result.tombstoned), (
        f"derived event was not tombstoned despite force=True: "
        f"{result.tombstoned}"
    )


def test_forget_skips_unknown_source_fail_closed(tmp_path: Path) -> None:
    """Unknown source on forget too -- fail-closed across the board."""
    mem = _new_memory(tmp_path)
    mem.remember(
        "derived event B with enough text for vstash min-length guard",
        title="derived_b",
        tags="source:session",
    )
    mem.remember(
        "ambiguously sourced content with enough length to be ingested by vstash",
        title="ambiguous_b",
        tags="source:made_up_unknown_value",
    )

    forget_mem = Memory(
        project="test_sourcing_integration",
        db=str(tmp_path / "test.db"),
        write_decider=AlwaysWrite(),
        forget_decider=ForgetConsolidated(),
    )
    result = forget_mem.forget(force=True)
    assert "text://ambiguous_b" not in result.tombstoned, (
        f"unknown-source event was tombstoned despite fail-closed: "
        f"{result.tombstoned}"
    )
