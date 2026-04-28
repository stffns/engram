"""Integration tests for ``Memory(role_classifier=...)``.

The unit tests in ``test_role_classifier.py`` exercise the classifier
itself with a fake embedder. These tests exercise the wiring point
between ``Memory.remember()`` and ``RoleClassifier``: that the role
tag is attached at write time when a classifier is configured, that
it is not attached when no classifier is configured (default), and
that it composes cleanly with caller-supplied tags.

Uses a real ``Memory`` (real vstash, real SQLite) but stubs the
classifier with a fake to keep the tests fast and deterministic. The
classifier API contract is what we care about here, not the embedder
math (already covered upstream).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from merken import Memory
from merken.role_classifier import RoleClassification, RoleClassifier


class _StubClassifier:
    """Fake classifier that returns a fixed role for any input.

    Quacks like ``RoleClassifier`` for the one method ``Memory`` calls.
    The real classifier is a heavy object (loads bundled prototypes,
    embeds via vstash); this stub is enough to verify the wiring.
    """

    def __init__(self, role: str = "state_change_report", confidence: float = 0.5) -> None:
        self._role = role
        self._confidence = confidence
        self.calls: list[str] = []

    def classify(self, text: str) -> RoleClassification:
        self.calls.append(text)
        return RoleClassification(
            role=self._role,
            confidence=self._confidence,
            role_similarities={self._role: self._confidence + 0.5},
        )


def _read_back_tags(mem: Memory, ingest_path: str) -> str | None:
    """Pull the tags string vstash recorded for the given path."""
    docs = mem._vstash.list(collection=mem.collection)
    for doc in docs:
        if doc.path == ingest_path:
            return getattr(doc, "tags", None)
    return None


def test_no_role_tag_when_classifier_not_configured(tmp_path: Path) -> None:
    """Default Memory (no classifier) must not attach a role tag."""
    with Memory(project="default_no_tag", db=tmp_path / "e.db") as mem:
        result = mem.remember(
            "Production now runs on Vespa as of 2026-04-15."
        )
        assert result.written
        assert result.ingest is not None

        tags = _read_back_tags(mem, result.ingest.source)
        # Either None or a string that does not contain a role tag.
        assert tags is None or "role:" not in tags


def test_role_tag_attached_when_classifier_configured(tmp_path: Path) -> None:
    """With a classifier, ``Memory.remember`` must tag the ingest with
    ``role:<predicted_role>``."""
    stub = _StubClassifier(role="state_change_report")
    with Memory(
        project="with_clf",
        db=tmp_path / "e.db",
        role_classifier=stub,
    ) as mem:
        result = mem.remember("Vespa replaced Elasticsearch in production.")
        assert result.written
        assert result.ingest is not None

        tags = _read_back_tags(mem, result.ingest.source)
        assert tags is not None
        assert "role:state_change_report" in tags
        assert stub.calls == ["Vespa replaced Elasticsearch in production."]


def test_role_tag_appended_to_caller_tags(tmp_path: Path) -> None:
    """When the caller passes tags, the role tag must be appended,
    not replace them."""
    stub = _StubClassifier(role="preference")
    with Memory(
        project="composed_tags",
        db=tmp_path / "e.db",
        role_classifier=stub,
    ) as mem:
        result = mem.remember(
            "We would prefer Bazel over Make for the polyglot stack.",
            tags="source:notes,domain:build_system",
        )
        assert result.written
        assert result.ingest is not None

        tags = _read_back_tags(mem, result.ingest.source)
        assert tags is not None
        # Caller tags survived
        assert "source:notes" in tags
        assert "domain:build_system" in tags
        # Role tag landed
        assert "role:preference" in tags


def test_role_tag_skipped_when_write_skipped(tmp_path: Path) -> None:
    """If the write_decider rejects the event, classification should
    not happen (we don't pay the cost on skipped writes)."""
    from merken.policies.types import Decision, Event, WriteContext, WriteDecider

    class AlwaysSkip:
        name = "always_skip"

        def decide(self, event: Event, ctx: WriteContext) -> Decision:
            return Decision(
                write=False,
                reason="test_skip",
                confidence=1.0,
                policy="always_skip",
            )

    stub = _StubClassifier()
    with Memory(
        project="skip_no_classify",
        db=tmp_path / "e.db",
        write_decider=AlwaysSkip(),
        role_classifier=stub,
    ) as mem:
        result = mem.remember("This will be skipped.")
        assert not result.written

    # Classifier was never asked.
    assert stub.calls == []


def test_classify_role_returns_classification(tmp_path: Path) -> None:
    """``Memory.classify_role`` should return the classifier's result
    without writing anything to vstash."""
    stub = _StubClassifier(role="investigation", confidence=0.42)
    with Memory(
        project="classify_only",
        db=tmp_path / "e.db",
        role_classifier=stub,
    ) as mem:
        result = mem.classify_role("Investigating the consumer lag.")
        assert isinstance(result, RoleClassification)
        assert result.role == "investigation"
        assert result.confidence == pytest.approx(0.42)

        # Verify nothing was written.
        docs = list(mem._vstash.list(collection=mem.collection))
        assert docs == []


def test_classify_role_raises_when_unconfigured(tmp_path: Path) -> None:
    """``Memory.classify_role`` without a classifier must fail loudly,
    not silently return a default."""
    with Memory(project="no_classifier", db=tmp_path / "e.db") as mem:
        with pytest.raises(RuntimeError, match="role_classifier"):
            mem.classify_role("any text")


def test_role_classifier_default_real_embedder_smoke(tmp_path: Path) -> None:
    """End-to-end smoke with the real ``RoleClassifier.default()``.

    Confirms the integration works against a real BGE-small embedder,
    not just stubs. Uses one canonical SCR example so the test is
    deterministic enough not to flake; classifier validation lives in
    PR #43 (3-seed canonical eval, macro-F1 0.929).
    """
    clf = RoleClassifier.default()
    with Memory(
        project="real_clf_smoke",
        db=tmp_path / "e.db",
        role_classifier=clf,
    ) as mem:
        result = mem.remember(
            "Production logging has been running on OTLP since 2026-03-12."
        )
        assert result.written
        assert result.ingest is not None

        tags = _read_back_tags(mem, result.ingest.source)
        assert tags is not None
        assert "role:state_change_report" in tags
