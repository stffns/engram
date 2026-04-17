"""Unit tests for the oracular labeling loop.

Uses a stub backend (no LLM) so the tests run offline in CI and
exercise the iteration logic, idempotence, and error handling paths.
The real GeminiLabelBackend is smoke-tested manually against an API
key outside of pytest.
"""

from __future__ import annotations

from pathlib import Path

from merken import AlwaysWrite, Memory, ShadowWriteDecider
from merken.labeling import (
    Label,
    LocalLLMLabelBackend,
    _parse_backend_response,
    label_disagreements,
)
from merken.policies import Decision, Event, WriteContext


class _StubShadow:
    """A ShadowWriteDecider-compatible secondary that always skips."""

    name = "stub-shadow"

    def decide(self, event: Event, ctx: WriteContext) -> Decision:  # noqa: ARG002
        return Decision(False, "stub_skip", 0.8, self.name)


class _StubBackend:
    """Idempotent deterministic backend that labels by the event text."""

    name = "stub-backend"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def label(self, text: str) -> Label:
        self.calls.append(text)
        decision = "NOISE" if "noise" in text.lower() else "DECISION"
        return Label(
            decision=decision,
            confidence=0.9,
            rationale=f"stub said {decision}",
            backend=self.name,
        )


class _RaisingBackend:
    name = "raises"

    def label(self, text: str) -> Label:  # noqa: ARG002
        raise RuntimeError("backend offline")


def _primed_memory(tmp_path: Path) -> Memory:
    """Build a Memory wired in shadow mode and write 3 events.

    Uses substantive, lexically distinct texts so vstash does not
    reject any of them as empty/trivial -- we want all three to
    produce ``shadow_disagree`` audit rows.
    """
    mem = Memory(
        project="labeling_test",
        db=tmp_path / "lbl.db",
        write_decider=ShadowWriteDecider(AlwaysWrite(), _StubShadow()),
    )
    mem.remember(
        "Decision: switched the inventory service from CockroachDB to Yugabyte "
        "after a three-week spike showed better multi-region write latency.",
        title="t_decision",
    )
    mem.remember(
        "Routine noise log: warehouse inventory service nightly job completed "
        "at 04:12 UTC with zero errors reported and full report attached.",
        title="t_noise",
    )
    mem.remember(
        "Ambient meeting reminder for the platform sync on Tuesday at 14:00; "
        "agenda covers onboarding, access control, and the next refactor wave.",
        title="t_meeting",
    )
    return mem


# ------------------------------------------------------------- parser


def test_parse_backend_response_full() -> None:
    out = _parse_backend_response(
        "LABEL: DECISION | CONFIDENCE: 0.87 | REASON: it documents a real choice",
        "test",
    )
    assert out.decision == "DECISION"
    assert out.confidence == 0.87
    assert out.rationale.startswith("it documents")
    assert out.backend == "test"


def test_parse_backend_response_rejects_unknown_label() -> None:
    out = _parse_backend_response(
        "LABEL: MAYBE | CONFIDENCE: 0.5 | REASON: dunno",
        "test",
    )
    assert out.decision == "UNCERTAIN"


def test_parse_backend_response_clamps_confidence() -> None:
    out = _parse_backend_response(
        "LABEL: NOISE | CONFIDENCE: 2.0 | REASON: out of range",
        "test",
    )
    assert 0.0 <= out.confidence <= 1.0


def test_parse_backend_response_handles_malformed() -> None:
    out = _parse_backend_response("garbage output", "test")
    assert out.decision == "UNCERTAIN"
    assert out.confidence == 0.0


# ------------------------------------------------------------- loop


def test_label_disagreements_writes_labels_and_skips_second_run(
    tmp_path: Path,
) -> None:
    with _primed_memory(tmp_path) as mem:
        backend = _StubBackend()

        results = list(label_disagreements(mem, backend))
        labeled = [r for r in results if r[0] == "labeled"]
        assert len(labeled) == 3, (
            "all 3 writes should have produced shadow_disagree rows"
        )

        # Second pass: every event is already labeled.
        results_2 = list(label_disagreements(mem, backend))
        labeled_2 = [r for r in results_2 if r[0] == "labeled"]
        assert labeled_2 == [], "re-running must be idempotent"
        assert len(backend.calls) == 3, (
            "backend must NOT be called again after all events are labeled"
        )


def test_label_disagreements_respects_limit(tmp_path: Path) -> None:
    with _primed_memory(tmp_path) as mem:
        backend = _StubBackend()
        results = list(label_disagreements(mem, backend, limit=2))
        labeled = [r for r in results if r[0] == "labeled"]
        assert len(labeled) == 2


def test_label_disagreements_error_is_caught(tmp_path: Path) -> None:
    with _primed_memory(tmp_path) as mem:
        results = list(label_disagreements(mem, _RaisingBackend()))
        errored = [r for r in results if r[0] == "error"]
        assert len(errored) >= 1
        assert all("RuntimeError" in r[3] for r in errored)


def test_local_label_backend_maps_decision(monkeypatch) -> None:
    """LocalLLMLabelBackend converts WriteDecider output into a Label.

    The backend wraps ``LLMWriteDecider`` and maps its write/skip call
    onto the DECISION/NOISE oracle schema. We stub the decider so
    tests stay offline.
    """
    from merken.policies import Decision

    calls: list[str] = []

    class _StubDeciderFactory:
        def __init__(self, *, write: bool, policy: str = "stub-llm") -> None:
            self._write = write
            self._policy = policy

        def decide(self, event: Event, ctx: WriteContext) -> Decision:  # noqa: ARG002
            calls.append(event.text)
            return Decision(
                write=self._write,
                reason="P(D)=0.900 P(N)=0.100 raw_mass=0.950",
                confidence=0.9,
                policy=self._policy,
            )

    # Avoid loading transformers + torch in CI.
    def _fake_init(self, *, model_name, device="cpu", confidence_threshold=0.5):
        self._decider = _StubDeciderFactory(write=True)
        self.name = f"local-llm:{model_name}"

    monkeypatch.setattr(
        LocalLLMLabelBackend, "__init__", _fake_init
    )

    backend = LocalLLMLabelBackend(model_name="stub-model")
    out = backend.label("some event text")

    assert out.decision == "DECISION"
    assert out.backend == "local-llm:stub-model"
    assert 0.0 <= out.confidence <= 1.0
    assert "P(D)=" in out.rationale
    assert calls == ["some event text"]


def test_memory_label_roundtrip(tmp_path: Path) -> None:
    """Memory.remember_label -> Memory.search_labels should find it."""
    with Memory(project="lbl", db=tmp_path / "r.db") as mem:
        mem.remember_label(event_title="demo_evt", body='{"decision": "DECISION"}')
        rows = mem.search_labels(top_k=10)
        titles = [(r.title or "") for r in rows]
        assert any(t == "label:demo_evt" for t in titles)
