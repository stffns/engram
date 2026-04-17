"""Tests for ChainedWriteDecider -- the graduation-mode write path.

Graduation replaces shadow mode's ``Shadow(Heuristic|classifier)`` with
``Chain(Heuristic -> classifier)``. The chain short-circuits on
gate-skip so the classifier never has to learn dedup / length rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from merken import Memory
from merken.policies import (
    ChainedWriteDecider,
    Decision,
    Event,
    HeuristicWriteDecider,
    WriteContext,
)


class _FixedClassifier:
    """Minimal classifier double. Always returns the preset decision."""

    def __init__(self, *, write: bool, reason: str, confidence: float = 0.9) -> None:
        self.name = f"FixedClassifier({reason})"
        self._write = write
        self._reason = reason
        self._confidence = confidence

    def decide(self, event: Event, ctx: WriteContext) -> Decision:  # noqa: ARG002
        return Decision(self._write, self._reason, self._confidence, self.name)


@pytest.fixture
def ctx() -> WriteContext:
    return WriteContext(project="test")


def test_gate_skip_short_circuits(ctx: WriteContext) -> None:
    """When Heuristic skips (empty text), the classifier MUST NOT run."""
    calls: list[str] = []

    class _TracingClassifier:
        name = "tracing"

        def decide(self, event: Event, ctx: WriteContext) -> Decision:  # noqa: ARG002
            calls.append(event.text)
            return Decision(True, "clf_write", 1.0, "tracing")

    chain = ChainedWriteDecider(HeuristicWriteDecider(), _TracingClassifier())
    d = chain.decide(Event(text=""), ctx)

    assert d.write is False
    assert d.reason == "empty", "gate reason must be preserved verbatim"
    assert calls == [], "classifier must not run when the gate skips"


def test_gate_ok_delegates_write(ctx: WriteContext) -> None:
    chain = ChainedWriteDecider(
        HeuristicWriteDecider(),
        _FixedClassifier(write=True, reason="clf_novel", confidence=0.77),
    )
    d = chain.decide(
        Event(text="A reasonably long event that passes the gate."), ctx
    )
    assert d.write is True
    assert d.reason.startswith("gate_ok|")
    assert "clf_novel" in d.reason
    assert d.confidence == pytest.approx(0.77)


def test_gate_ok_honours_classifier_skip(ctx: WriteContext) -> None:
    """Classifier can still veto a write after the gate lets it through."""
    chain = ChainedWriteDecider(
        HeuristicWriteDecider(),
        _FixedClassifier(write=False, reason="clf_looks_like_noise"),
    )
    d = chain.decide(
        Event(text="A reasonably long event that passes the gate."), ctx
    )
    assert d.write is False
    assert "gate_ok|" in d.reason
    assert "clf_looks_like_noise" in d.reason


def test_hydration_forwards_to_gate(tmp_path: Path) -> None:
    """The gate's dedup state must still be populated cross-invocation."""
    gate = HeuristicWriteDecider()
    chain = ChainedWriteDecider(
        gate, _FixedClassifier(write=True, reason="clf_novel")
    )
    with Memory(
        project="chain", db=tmp_path / "c.db", write_decider=chain
    ) as mem:
        first = mem.remember(
            "An event that should be written once and then deduped.",
            title="chain1",
        )
        second = mem.remember(
            "An event that should be written once and then deduped.",
            title="chain1_dup",
        )

    assert first.written is True
    # Dedup should fire in the gate on the second call, short-circuiting
    # the classifier.
    assert second.written is False
    assert second.decision.reason == "dup_exact"


def test_env_primary_overrides_shadow(monkeypatch, tmp_path: Path) -> None:
    """If MERKEN_PRIMARY wins, no ShadowWriteDecider is built.

    We do not have a real nanoGPT ckpt in CI, so we point PRIMARY at a
    missing path and assert that Memory degrades to plain
    HeuristicWriteDecider (no chain, no shadow). The behaviour that
    matters: misconfig must never take writes offline.
    """
    monkeypatch.setenv("MERKEN_SHADOW", "nanogpt")  # would otherwise wire
    monkeypatch.setenv("MERKEN_PRIMARY", "nanogpt")
    # Omit ckpt paths -> _build_classifier raises -> memory degrades.
    monkeypatch.delenv("MERKEN_PRIMARY_NANOGPT_CKPT", raising=False)
    monkeypatch.delenv("MERKEN_PRIMARY_NANOGPT_META", raising=False)
    monkeypatch.delenv("MERKEN_SHADOW_NANOGPT_CKPT", raising=False)
    monkeypatch.delenv("MERKEN_SHADOW_NANOGPT_META", raising=False)

    import importlib

    import merken._shadow
    importlib.reload(merken._shadow)

    with Memory(project="t", db=tmp_path / "e.db") as mem:
        assert isinstance(mem._write_decider, HeuristicWriteDecider)
