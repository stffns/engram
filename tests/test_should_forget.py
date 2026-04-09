"""Tests for ``should_forget`` — decision primitive #4 (the last).

Split between pure unit tests (no vstash) and integration tests
that exercise the full Memory.forget() path end-to-end.
"""

from __future__ import annotations

from pathlib import Path

from engram import (
    ForgetConsolidated,
    ForgetContext,
    ForgetDecision,
    ForgetResult,
    Memory,
    NeverForget,
    PeriodicConsolidator,
)


def _ctx(*facts: str) -> ForgetContext:
    return ForgetContext(project="unit", derived_in_facts=list(facts))


# ------------------------------------------------------------- pure policy


def test_never_forget_always_says_no() -> None:
    d = NeverForget().decide("text://event_1", "some text", _ctx("fact_a", "fact_b"))
    assert not d.tombstone
    assert d.reason == "never_auto"
    assert isinstance(d, ForgetDecision)


def test_forget_consolidated_below_threshold() -> None:
    d = ForgetConsolidated(min_facts=2).decide(
        "text://event_1", "text", _ctx("fact_a")
    )
    assert not d.tombstone
    assert d.reason.startswith("not_consolidated")


def test_forget_consolidated_at_threshold() -> None:
    d = ForgetConsolidated(min_facts=2).decide(
        "text://event_1", "text", _ctx("fact_a", "fact_b")
    )
    assert d.tombstone
    assert d.reason == "consolidated_in_2_facts"


def test_forget_consolidated_default_min_facts_is_one() -> None:
    d = ForgetConsolidated().decide("text://e", "t", _ctx("fact_a"))
    assert d.tombstone
    assert d.reason == "consolidated_in_1_facts"


def test_forget_consolidated_no_facts_means_keep() -> None:
    d = ForgetConsolidated().decide("text://e", "t", _ctx())
    assert not d.tombstone
    assert "0<1" in d.reason


# ------------------------------------------------------- integration w/ Memory


def test_memory_forget_default_is_never_forget(tmp_path: Path) -> None:
    """Without configuring a forget decider, Memory.forget() is a no-op
    on every event (NeverForget policy). Safe by default."""
    with Memory(
        project="forget_default",
        db=tmp_path / "e.db",
        consolidate_decider=PeriodicConsolidator(min_events=2),
    ) as mem:
        mem.remember("The user chose postgres for the analytics project.")
        mem.remember("User picked postgres for the analytics project this sprint.")
        mem.consolidate()

        result = mem.forget()

    assert isinstance(result, ForgetResult)
    assert result.tombstoned == []
    assert result.events_examined == 2
    assert result.decider == "NeverForget"
    # All events should have been examined and skipped with
    # "never_auto" as the reason
    assert all(reason == "never_auto" for _, reason in result.skipped)


def test_memory_forget_consolidated_tombstones_covered_events(
    tmp_path: Path,
) -> None:
    """With ForgetConsolidated, events whose paths appear in a fact's
    derived_from get tombstoned; unconsolidated events do not."""
    with Memory(
        project="forget_consolidated",
        db=tmp_path / "e.db",
        consolidate_decider=PeriodicConsolidator(min_events=2),
        forget_decider=ForgetConsolidated(),
    ) as mem:
        # Two events that should cluster into one fact
        mem.remember("The user chose postgres for the analytics project.")
        mem.remember(
            "User picked postgres for the analytics project this sprint."
        )
        # One event about a totally different topic — won't cluster
        mem.remember(
            "Marketing approved new brand colors for summer launch campaign."
        )

        cons = mem.consolidate()
        assert cons.facts_written >= 1

        result = mem.forget()

    # The two consolidated events should be tombstoned; the marketing
    # singleton should be skipped.
    assert len(result.tombstoned) == 2
    assert len(result.skipped) == 1
    assert result.skipped[0][1].startswith("not_consolidated")


def test_memory_forget_force_tombstones_everything(tmp_path: Path) -> None:
    """force=True bypasses the decider. Every episodic event gets
    tombstoned regardless of consolidation state."""
    with Memory(
        project="forget_force",
        db=tmp_path / "e.db",
        forget_decider=NeverForget(),
    ) as mem:
        mem.remember("A singleton event about wombats.")
        mem.remember("Another singleton event about kangaroos.")

        result = mem.forget(force=True)

    assert len(result.tombstoned) == 2
    assert result.skipped == []


def test_memory_forget_tombstone_preserves_full_text(tmp_path: Path) -> None:
    """After forgetting, the tombstone collection must contain the
    full text of the event so it can be recovered by unforgetting."""
    with Memory(
        project="forget_tombstone",
        db=tmp_path / "e.db",
    ) as mem:
        original_text = (
            "The canary sentence for the tombstone test — it must appear "
            "verbatim in the tombstone collection after forget runs."
        )
        mem.remember(original_text)
        mem.forget(force=True)

        rows = mem.tombstones(query="canary tombstone", top_k=5)

    assert rows, "expected at least one tombstone row"
    found = any("canary sentence" in (r.text or "") for r in rows)
    assert found, (
        f"tombstone did not preserve full text. rows: "
        f"{[(r.title or '')[:50] for r in rows]}"
    )


def test_memory_forget_removes_from_recall(tmp_path: Path) -> None:
    """After forget, the event must no longer surface in recall
    (only the tombstone record remains, and tombstones are in a
    separate collection that recall doesn't touch)."""
    with Memory(
        project="forget_recall",
        db=tmp_path / "e.db",
    ) as mem:
        mem.remember(
            "A distinctive phrase about platypus biology for recall isolation."
        )

        # Before forget: recall finds it
        pre_hits = mem.recall("platypus biology", top_k=5)
        assert pre_hits, "pre-forget recall should find the event"

        mem.forget(force=True)

        # After forget: recall should no longer return it
        post_hits = mem.recall("platypus biology", top_k=5)

    assert not any(
        "platypus biology" in (h.text or "")
        for h in post_hits
    ), f"post-forget recall still returned the event: {[h.text[:60] for h in post_hits]}"


def test_memory_forget_audit_row_recorded(tmp_path: Path) -> None:
    """Every forget decision (tombstone or skip) writes a should_forget
    audit row."""
    with Memory(
        project="forget_audit",
        db=tmp_path / "e.db",
        forget_decider=NeverForget(),
    ) as mem:
        mem.remember("An event for the audit test.")
        mem.forget()

        rows = mem.audit(query="should_forget", top_k=10)

    assert rows
    bodies = " ".join((r.text or "") for r in rows)
    assert "should_forget" in bodies
    assert "never_auto" in bodies


def test_memory_forget_does_not_touch_semantic_facts(tmp_path: Path) -> None:
    """Forgetting episodic events must not delete the semantic facts
    derived from them. The fact's derived_from provenance stays
    pointing at the tombstoned path — that's the provenance record
    by design."""
    with Memory(
        project="forget_facts_survive",
        db=tmp_path / "e.db",
        consolidate_decider=PeriodicConsolidator(min_events=2),
        forget_decider=ForgetConsolidated(),
    ) as mem:
        mem.remember("User prefers postgres for the analytics project now.")
        mem.remember(
            "User picked postgres for the analytics project this quarter."
        )
        cons = mem.consolidate()
        fact_count_before = cons.facts_written

        mem.forget()

        # Semantic facts should still be there
        remaining_facts = mem._vstash.list(
            collection=mem.collection, layer="semantic"
        )

    assert len(remaining_facts) == fact_count_before
    assert fact_count_before >= 1
