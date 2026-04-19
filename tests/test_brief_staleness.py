"""Tests for brief staleness fix shipped 2026-04-19.

Two changes covered:

1. ``generate_briefs`` interpolates ``today`` into the prompt and
   the resulting briefs MUST contain the ``**As of:** YYYY-MM-DD``
   line under each ``##`` header.

2. ``Memory.consolidate(method="brief_v1")`` tombstones any prior
   briefs whose events_fingerprint differs from the new one BEFORE
   writing the new brief set. Old briefs land in
   ``merken_tombstones``; ``recall-briefs`` only sees current.
"""

from __future__ import annotations

from pathlib import Path

from merken import AlwaysWrite, Memory
from merken.consolidation import generate_briefs


def _fake_synth(today_marker: str = "2026-04-19"):
    """Produce a synthesize_fn that emits one brief per call.

    The brief mirrors the schema in `_BRIEF_PROMPT_TEMPLATE` so the
    parser splits it correctly. Includes the **As of:** line that
    the prompt requires; tests verify the fix wires the interpolation
    properly.
    """
    def synth(prompts: list[str]) -> str:
        # Echo the today_marker so we can assert it propagates.
        return (
            f"## Topic alpha\n"
            f"**As of:** {today_marker}\n"
            f"- [v1]: alpha was decided\n"
            f"- **Current state:** alpha is in place\n"
            f"\n"
            f"## Topic beta\n"
            f"**As of:** {today_marker}\n"
            f"- **Identity:** beta entity\n"
        )
    return synth


def test_generate_briefs_injects_today_into_prompt() -> None:
    """The prompt sees `today` interpolated; the synth fn can read it."""
    captured = {}

    def synth_capture(prompts: list[str]) -> str:
        captured["prompt"] = prompts[0]
        return "## Empty brief\n**As of:** 2026-04-19\nplaceholder"

    events = [("path1", "event one text"), ("path2", "event two text")]
    generate_briefs(events, synth_capture, today="2026-04-19")
    assert "Today is 2026-04-19" in captured["prompt"], (
        "today must be injected into the prompt header"
    )
    assert "**As of:** 2026-04-19" in captured["prompt"], (
        "the as_of template line must reach the model"
    )


def test_generate_briefs_default_today_is_iso_today() -> None:
    """When today=None, defaults to today's UTC date in ISO."""
    from datetime import datetime, timezone
    captured = {}

    def synth_capture(prompts: list[str]) -> str:
        captured["prompt"] = prompts[0]
        return "## x\n**As of:** today\nplaceholder"

    generate_briefs([("p", "text")], synth_capture)
    expected = datetime.now(timezone.utc).date().isoformat()
    assert f"Today is {expected}" in captured["prompt"]


def test_generate_briefs_parses_multi_brief_output() -> None:
    """Two briefs separated by `\\n## ` are returned as a list of two."""
    out = generate_briefs(
        [("p1", "e1"), ("p2", "e2")],
        _fake_synth(),
        today="2026-04-19",
    )
    assert len(out) == 2
    assert out[0].startswith("## Topic alpha")
    assert out[1].startswith("## Topic beta")
    for brief in out:
        assert "**As of:** 2026-04-19" in brief


def test_consolidate_brief_v1_supersedes_stale_briefs(tmp_path: Path) -> None:
    """When events change (new fingerprint), old briefs get tombstoned."""
    db = str(tmp_path / "stale.db")
    mem = Memory(project="stale_test", db=db, write_decider=AlwaysWrite())

    # Round 1: 2 events -> consolidate -> 2 briefs in semantic.
    mem.remember(
        "first event with enough length to be ingested by vstash guardrail",
        title="e1",
    )
    mem.remember(
        "second event with enough length to be ingested by vstash guardrail",
        title="e2",
    )
    r1 = mem.consolidate(
        method="brief_v1",
        force=True,
        synthesize_fn=_fake_synth("2026-04-18"),
    )
    assert r1.facts_written == 2

    # Snapshot: there should be 2 briefs in semantic.
    sem = mem._vstash.list(collection=mem.collection, layer="semantic")
    semantic_briefs = [
        d for d in sem if "method:brief_v1" in (getattr(d, "tags", "") or "")
    ]
    assert len(semantic_briefs) == 2

    # Round 2: add a 3rd event, fingerprint changes, re-consolidate.
    mem.remember(
        "third event with enough length to be ingested by vstash guardrail",
        title="e3",
    )
    r2 = mem.consolidate(
        method="brief_v1",
        force=True,
        synthesize_fn=_fake_synth("2026-04-19"),  # newer "today"
    )
    assert r2.facts_written == 2  # the fake synth always emits 2

    # Old briefs (from r1, fingerprint A) should be gone from semantic.
    sem_after = mem._vstash.list(collection=mem.collection, layer="semantic")
    semantic_after = [
        d for d in sem_after if "method:brief_v1" in (getattr(d, "tags", "") or "")
    ]
    # Only the NEW briefs remain (2). The old 2 should have been tombstoned.
    assert len(semantic_after) == 2, (
        f"expected only 2 fresh briefs in semantic; got {len(semantic_after)}. "
        f"stale briefs were not superseded."
    )

    # The tombstones collection should have the 2 old briefs.
    tomb = mem._vstash.search(
        "brief", top_k=10,
        collection="merken_tombstones",
        layer="tombstone",
    )
    paths_in_tombs = " ".join(getattr(t, "text", "") for t in tomb)
    # At least one tombstone must reference brief_v1 supersession.
    assert "brief_v1_supersede" in paths_in_tombs or "brief" in paths_in_tombs.lower(), (
        f"expected tombstoned briefs; got: {paths_in_tombs[:300]}"
    )


def test_consolidate_brief_v1_skips_when_fingerprint_matches(
    tmp_path: Path,
) -> None:
    """If events haven't changed, second consolidate is a no-op (no tombstone)."""
    db = str(tmp_path / "noop.db")
    mem = Memory(project="noop_test", db=db, write_decider=AlwaysWrite())

    mem.remember(
        "first event with sufficient length for vstash to ingest",
        title="e1",
    )
    mem.remember(
        "second event with sufficient length for vstash to ingest",
        title="e2",
    )
    r1 = mem.consolidate(
        method="brief_v1", force=True, synthesize_fn=_fake_synth(),
    )
    assert r1.facts_written == 2

    # Re-run without changing events.
    r2 = mem.consolidate(
        method="brief_v1", force=True, synthesize_fn=_fake_synth(),
    )
    # Skipped because fingerprint matches -- no new briefs written,
    # no old briefs tombstoned.
    assert r2.skipped is True
    assert r2.facts_written == 0
