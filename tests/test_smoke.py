"""Phase 0 smoke test (CONSTITUTION §11).

Ingest one event, recall it, assert it comes back. If this passes, we have a
project. If not, we have a list.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from merken import Memory, __version__


def test_version_exposed() -> None:
    assert __version__ == "0.1.0"


def test_smoke_remember_then_recall(tmp_path: Path) -> None:
    db = tmp_path / "merken.db"

    with Memory(project="merken_smoke", db=db) as mem:
        result = mem.remember(
            "The smoke test ingested this sentence about purple elephants on 2026-04-08.",
            title="smoke-canary",
        )
        assert result is not None

        hits = mem.recall("purple elephants", top_k=3)

    assert hits, "expected at least one hit for the canary phrase"
    assert any("purple elephants" in (hit.text or "").lower() for hit in hits), (
        f"canary phrase not found in recall results: {[h.text for h in hits]}"
    )


def test_layer_tag_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "merken.db"

    with Memory(project="merken_smoke_layer", db=db) as mem:
        mem.remember("episodic event about a teal giraffe", layer="episodic")
        mem.remember("semantic fact: the user prefers teal", layer="semantic")

        episodic_hits = mem.recall("teal giraffe", top_k=5, layer="episodic")
        semantic_hits = mem.recall("teal preference", top_k=5, layer="semantic")

    assert episodic_hits, "episodic layer recall returned nothing"
    assert semantic_hits, "semantic layer recall returned nothing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
