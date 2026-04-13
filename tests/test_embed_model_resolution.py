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


def test_resolver_falls_back_to_config_on_fresh_store(tmp_path: Path) -> None:
    """A fresh store has no store_meta row for embedding_model because
    no ingest has happened yet. The resolver must return the vstash
    config default in that case."""
    from vstash.config import EmbeddingsConfig

    with Memory(project="fresh", db=tmp_path / "e.db") as mem:
        resolved = _resolve_vstash_embed_model(mem._vstash)

    assert resolved == EmbeddingsConfig().model


def test_resolver_falls_back_when_store_meta_lacks_embedding_model(
    tmp_path: Path,
) -> None:
    """As of vstash 0.27.0, a fresh store does not auto-write
    ``store_meta.embedding_model`` even after the first ingest —
    only ``schema_version`` and ``vstash_version`` land there. The
    resolver must treat a missing row the same as a fresh store
    and fall back to the config default. (Jay's legacy store has
    the row because it was created by an older vstash that used
    to write it — worth tracking as a vstash upstream regression
    in observability, not something merken should paper over.)
    """
    from vstash.config import EmbeddingsConfig

    with Memory(project="ingested", db=tmp_path / "e.db") as mem:
        mem.remember("A real event so vstash populates store_meta.")

        con = sqlite3.connect(str(mem._vstash._store.db_path))
        row = con.execute(
            "SELECT value FROM store_meta WHERE key = ?",
            ("embedding_model",),
        ).fetchone()
        con.close()

        resolved = _resolve_vstash_embed_model(mem._vstash)

    # Confirm the precondition: vstash did not write embedding_model
    assert row is None, (
        "vstash unexpectedly wrote store_meta.embedding_model on ingest. "
        "If vstash regained this behavior upstream, update this test to "
        "assert the resolver reads the row instead of falling back."
    )
    # Resolver falls back cleanly to config default
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
