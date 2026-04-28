"""Tests for merken's embed-model resolution.

The bug (2026-04-09): merken hardcoded ``BAAI/bge-small-en-v1.5`` as
the consolidation embedder even though vstash might be using a
different model to ingest. That created a silent vector-space
mismatch between "how merken clusters events" and "how vstash
retrieves them." The fix is to read the model from the vstash
store_meta at runtime, falling back to vstash.config only if the
store is fresh.

These tests exercise the resolver in isolation (no clustering,
no consolidation, no LLM) and ensure the fallback chain is
correct.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from merken import Memory
from merken.memory import _resolve_vstash_embed_model


def test_resolver_returns_value_written_by_vstash_on_fresh_store(
    tmp_path: Path,
) -> None:
    """As of vstash 0.35.0, a fresh store has ``store_meta.embedding_model``
    populated automatically (previously vstash 0.27.0 did not write it).
    The resolver must return that authoritative value so merken's
    clustering and vstash's retrieval stay in the same vector space.

    Earlier behavior (vstash 0.27.0): no row -> fall back to config.
    Current behavior (vstash 0.35.0+): row present -> use it.

    Regression guard: if vstash ever stops writing this row again the
    precondition assertion below points to fixing the upstream
    observability rather than papering over it in merken.
    """
    with Memory(project="fresh", db=tmp_path / "e.db") as mem:
        con = sqlite3.connect(str(mem._vstash._store.db_path))
        row = con.execute(
            "SELECT value FROM store_meta WHERE key = ?",
            ("embedding_model",),
        ).fetchone()
        con.close()

        assert row is not None, (
            "vstash did not write store_meta.embedding_model on store "
            "creation. If vstash regressed on this, the resolver fallback "
            "path is now load-bearing again -- update this test to assert "
            "the fallback rather than papering over a vstash regression."
        )
        stored_model = row[0]

        resolved = _resolve_vstash_embed_model(mem._vstash)

    assert resolved == stored_model


def test_resolver_falls_back_when_store_meta_row_missing(
    tmp_path: Path,
) -> None:
    """Legacy stores (created by older vstash versions) and corrupted
    stores may lack the ``embedding_model`` row. The resolver must
    still resolve cleanly by falling back to the vstash config default
    rather than returning ``None`` or crashing.

    The "missing row" state is not currently reachable through the
    vstash public API on a new store (vstash 0.35.0 always writes it),
    so we simulate it by deleting the row after store init.
    """
    from vstash.config import EmbeddingsConfig

    with Memory(project="legacy", db=tmp_path / "e.db") as mem:
        con = sqlite3.connect(str(mem._vstash._store.db_path))
        con.execute(
            "DELETE FROM store_meta WHERE key = ?",
            ("embedding_model",),
        )
        con.commit()
        con.close()

        resolved = _resolve_vstash_embed_model(mem._vstash)

    assert resolved == EmbeddingsConfig().model


def test_resolver_prefers_store_meta_over_config(tmp_path: Path) -> None:
    """If store_meta.embedding_model is set to a value different from the
    config default, the resolver must return the store_meta value."""
    with Memory(project="override", db=tmp_path / "e.db") as mem:
        mem.remember("A real event so store_meta exists.")

        fake_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        con = sqlite3.connect(str(mem._vstash._store.db_path))
        con.execute(
            "INSERT OR REPLACE INTO store_meta (key, value, updated_at) "
            "VALUES (?, ?, datetime('now'))",
            ("embedding_model", fake_model),
        )
        con.commit()
        con.close()

        resolved = _resolve_vstash_embed_model(mem._vstash)

    assert resolved == fake_model


def test_resolver_tolerates_missing_db_path() -> None:
    """The resolver must not crash if the vstash memory lacks a db_path
    attribute for any reason. Fallback to config default."""
    from vstash.config import EmbeddingsConfig

    class BareStub:
        pass

    resolved = _resolve_vstash_embed_model(BareStub())
    assert resolved == EmbeddingsConfig().model
