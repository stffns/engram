"""ShadowWriteDecider unit tests.

Shadow mode is the bootstrap step for any not-yet-trusted classifier:
the primary decider is authoritative, the shadow only annotates the
audit trail with its prediction. Tests cover the four
agree/disagree combinations plus shadow-failure isolation.
"""

from __future__ import annotations

import pytest

from merken import Memory, ShadowWriteDecider
from merken.policies import AlwaysWrite, Decision, Event, WriteContext


class _StubDecider:
    """Minimal test double with the WriteDecider protocol."""

    def __init__(self, *, write: bool, reason: str = "stub", confidence: float = 0.9) -> None:
        self.name = f"Stub({reason})"
        self._write = write
        self._reason = reason
        self._confidence = confidence

    def decide(self, event: Event, ctx: WriteContext) -> Decision:
        return Decision(self._write, self._reason, self._confidence, self.name)


class _RaisingDecider:
    name = "Raising"

    def decide(self, event: Event, ctx: WriteContext) -> Decision:  # noqa: ARG002
        raise RuntimeError("boom")


@pytest.fixture
def ctx() -> WriteContext:
    return WriteContext(project="test")


@pytest.fixture
def event() -> Event:
    return Event(text="a reasonable event text for the deciders")


def test_primary_write_and_shadow_write_is_agree(event: Event, ctx: WriteContext) -> None:
    primary = _StubDecider(write=True, reason="novel", confidence=0.8)
    shadow = _StubDecider(write=True, reason="shadow-write", confidence=0.7)
    d = ShadowWriteDecider(primary, shadow).decide(event, ctx)

    assert d.write is True
    assert d.confidence == 0.8
    assert "novel" in d.reason
    assert "shadow_agree:" in d.reason
    assert "=write:0.700" in d.reason


def test_primary_write_and_shadow_skip_is_disagree(event: Event, ctx: WriteContext) -> None:
    primary = _StubDecider(write=True, reason="novel")
    shadow = _StubDecider(write=False, reason="shadow-skip", confidence=0.6)
    d = ShadowWriteDecider(primary, shadow).decide(event, ctx)

    assert d.write is True, "primary must win"
    assert "shadow_disagree:" in d.reason
    assert "=skip:0.600" in d.reason


def test_primary_skip_and_shadow_write_is_disagree(event: Event, ctx: WriteContext) -> None:
    primary = _StubDecider(write=False, reason="too_short:<8")
    shadow = _StubDecider(write=True, reason="shadow-keep", confidence=0.95)
    d = ShadowWriteDecider(primary, shadow).decide(event, ctx)

    assert d.write is False, "primary must win"
    assert "too_short" in d.reason
    assert "shadow_disagree:" in d.reason


def test_primary_skip_and_shadow_skip_is_agree(event: Event, ctx: WriteContext) -> None:
    primary = _StubDecider(write=False, reason="dup_exact")
    shadow = _StubDecider(write=False, reason="shadow-skip")
    d = ShadowWriteDecider(primary, shadow).decide(event, ctx)

    assert d.write is False
    assert "shadow_agree:" in d.reason


def test_shadow_exception_does_not_block_primary(event: Event, ctx: WriteContext) -> None:
    primary = _StubDecider(write=True, reason="novel", confidence=0.8)
    d = ShadowWriteDecider(primary, _RaisingDecider()).decide(event, ctx)

    assert d.write is True
    assert d.confidence == 0.8
    assert "novel" in d.reason
    assert "shadow_error:RuntimeError" in d.reason


def test_set_hydrate_fn_forwards_to_primary() -> None:
    calls: list[str] = []

    class _HydratingDecider:
        name = "Hydrating"

        def set_hydrate_fn(self, fn) -> None:
            calls.append("primary")

        def decide(self, event, ctx):  # noqa: ARG002
            return Decision(True, "ok", 1.0, self.name)

    decider = ShadowWriteDecider(_HydratingDecider(), _StubDecider(write=True))
    decider.set_hydrate_fn(lambda: [])
    assert calls == ["primary"], "shadow must not receive hydrate_fn"


def test_end_to_end_with_memory(tmp_path) -> None:
    """Shadow mode surfaces disagreements in the audit reason string."""
    primary = AlwaysWrite()
    shadow = _StubDecider(write=False, reason="noise-like", confidence=0.73)

    with Memory(
        project="shadow_e2e",
        db=tmp_path / "shadow.db",
        write_decider=ShadowWriteDecider(primary, shadow),
    ) as mem:
        result = mem.remember(
            "An event worth writing under AlwaysWrite but that the shadow would skip.",
            title="shadow-canary",
        )
        assert result.written is True
        assert "shadow_disagree" in result.decision.reason
