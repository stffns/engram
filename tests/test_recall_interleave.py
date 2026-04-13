"""Regression test for Memory.recall round-robin layer interleave.

The bug: the v0 implementation drained layers sequentially, so a
decider plan like ``[semantic:5, episodic:3]`` would return up to
five semantic hits before ever looking at episodic. When the user's
``top_k`` was 3 and semantic had ≥ 3 hits, episodic was never
visited — even if the user's query was really about an event that
only lived in episodic.

The fix: Memory.recall now fetches every layer in the plan first,
then interleaves round-robin, dedupes by path, and truncates to the
caller's top_k. Each layer is guaranteed at least one slot until
the budget runs out.
"""

from __future__ import annotations

from pathlib import Path

from merken import Memory


def test_recall_interleaves_semantic_and_episodic(tmp_path: Path) -> None:
    """Given several semantic facts and one episodic singleton, a
    decider-routed recall with top_k=3 must include the episodic
    singleton in its results, even though semantic alone could fill
    the budget."""
    with Memory(project="interleave", db=tmp_path / "e.db") as mem:
        # Four semantic facts about the same topic
        for i in range(4):
            mem.remember(
                f"A semantic fact about the analytics warehouse, entry {i}.",
                title=f"fact_{i}",
                layer="semantic",
            )
        # One episodic singleton about a different topic
        mem.remember(
            "The Kafka merchant pipeline meeting decided to delay the migration.",
            title="kafka_meeting",
            layer="episodic",
        )

        hits = mem.recall("Kafka merchant pipeline meeting", top_k=3)

    # With the old sequential behavior, the 4 semantic hits would
    # fill top_k=3 and the Kafka episodic would never appear.
    # With round-robin interleave, semantic and episodic alternate,
    # so the Kafka event must be in the top-3.
    assert hits, "expected at least one hit"
    all_text = " ".join((h.text or "") for h in hits)
    assert "kafka" in all_text.lower(), (
        f"kafka episodic singleton was not surfaced; "
        f"recall returned: {[(h.title or '')[:40] for h in hits]}"
    )
