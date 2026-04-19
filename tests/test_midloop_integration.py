"""Integration tests for Memory.observe_step and the audit-row format.

The unit tests for the deciders themselves live in test_midloop.py.
This file verifies the WIRING:
- Memory.observe_step calls the configured midloop_decider.
- Trajectory state lives across calls on the same instance.
- The is_persistable_step filter is applied (audit doesn't flood).
- format_midloop_audit_row in audit.py matches what midloop.py
  exports (Phase 2b moved the canonical home; midloop.py re-exports).
"""

from __future__ import annotations

import time
from pathlib import Path

from merken import Memory
from merken.audit import format_midloop_audit_row as audit_module_fn
from merken.policies.midloop import (
    HeuristicMidloopDecider,
    HeuristicThresholds,
    MidloopDecision,
    NoopMidloopDecider,
    ShadowMidloopDecider,
    StepObservation,
    TaskCategory,
)
from merken.policies.midloop import format_midloop_audit_row as midloop_module_fn


def _obs(step_index: int, output: str = "step output text", task_id: str = "t1"):
    return StepObservation(
        step_id=f"s{step_index}",
        task_id=task_id,
        task_category=TaskCategory.EXPLORATORY,
        timestamp=time.time(),
        output_text=output,
        step_index=step_index,
        latency_ms=100,
    )


def _new_memory(tmp_path: Path, midloop=None) -> Memory:
    return Memory(
        project="test_midloop_integration",
        db=str(tmp_path / "midloop.db"),
        midloop_decider=midloop,
    )


# ----------------------------------------------------------------- canonical export

def test_format_midloop_audit_row_is_same_function() -> None:
    """audit.format_midloop_audit_row IS what midloop.format_midloop_audit_row points at."""
    assert midloop_module_fn is audit_module_fn


# ----------------------------------------------------------------- observe_step

def test_observe_step_returns_decision(tmp_path: Path) -> None:
    mem = _new_memory(tmp_path)
    decision = mem.observe_step(_obs(0))
    assert isinstance(decision, MidloopDecision)
    assert decision.intervene is False  # default Noop


def test_observe_step_appends_to_trajectory(tmp_path: Path) -> None:
    mem = _new_memory(tmp_path)
    for i in range(5):
        mem.observe_step(_obs(i))
    assert len(mem._trajectory) == 5


def test_observe_step_passes_trajectory_to_decider(tmp_path: Path) -> None:
    """Decider receives prior observations; can compute trends."""
    captured = {"sizes": []}

    class CapturingDecider:
        name = "CapturingDecider"

        def decide(self, observation, ctx, trajectory):
            captured["sizes"].append(len(trajectory))
            from merken.policies.midloop import (
                CognitiveState, InterventionAction,
            )
            return MidloopDecision(
                intervene=False,
                action=InterventionAction.NONE,
                confidence=1.0,
                reason="captured",
                state=CognitiveState.ON_TRACK,
                policy=self.name,
            )

    mem = _new_memory(tmp_path, midloop=CapturingDecider())
    for i in range(5):
        mem.observe_step(_obs(i))
    # Trajectory grows monotonically: 0, 1, 2, 3, 4.
    assert captured["sizes"] == [0, 1, 2, 3, 4]


def test_reset_trajectory_clears_window(tmp_path: Path) -> None:
    mem = _new_memory(tmp_path)
    for i in range(3):
        mem.observe_step(_obs(i))
    assert len(mem._trajectory) == 3
    mem.reset_trajectory()
    assert len(mem._trajectory) == 0
    assert mem._prev_midloop_decision is None


def test_trajectory_window_caps_at_default(tmp_path: Path) -> None:
    """trajectory_window=20 by default; deque drops oldest beyond that."""
    mem = _new_memory(tmp_path)
    for i in range(25):
        mem.observe_step(_obs(i))
    assert len(mem._trajectory) == 20  # capped


def test_trajectory_window_override(tmp_path: Path) -> None:
    mem = Memory(
        project="t",
        db=str(tmp_path / "x.db"),
        trajectory_window=5,
    )
    for i in range(10):
        mem.observe_step(_obs(i))
    assert len(mem._trajectory) == 5


# ----------------------------------------------------------------- audit persistence

def test_on_track_steps_dont_persist_to_audit(tmp_path: Path) -> None:
    """The is_persistable_step filter drops on_track-with-no-change rows."""
    mem = _new_memory(tmp_path)  # default Noop -> always on_track
    for i in range(5):
        mem.observe_step(_obs(i))
    # Query audit collection for should_intervene rows -- expect 0
    # because all 5 steps were on_track with no state change.
    hits = mem.audit("should_intervene", top_k=20)
    assert len(hits) == 0


def test_intervention_step_persists_to_audit(tmp_path: Path) -> None:
    """When the heuristic fires (intervene=True), the row IS persisted."""
    mem = _new_memory(tmp_path, midloop=HeuristicMidloopDecider())
    # Build a looping trajectory so the heuristic fires.
    for i in range(4):
        mem.observe_step(_obs(i, output="repeat repeat repeat"))
    final = mem.observe_step(
        _obs(4, output="repeat repeat repeat once more"),
    )
    assert final.intervene is True
    hits = mem.audit("should_intervene", top_k=20)
    # At least the looping intervention persists.
    assert len(hits) >= 1


def test_last_step_always_persists(tmp_path: Path) -> None:
    """is_last_step=True forces persistence even on ON_TRACK."""
    mem = _new_memory(tmp_path)
    mem.observe_step(_obs(0), is_last_step=True)
    hits = mem.audit("should_intervene", top_k=10)
    assert len(hits) == 1


# ----------------------------------------------------------------- shadow integration

def test_shadow_midloop_decider_via_memory(tmp_path: Path) -> None:
    """ShadowMidloopDecider as Memory's midloop -- primary controls,
    shadow audit-tags. End-to-end through observe_step."""
    primary = NoopMidloopDecider()
    shadow = HeuristicMidloopDecider()
    mem = _new_memory(
        tmp_path, midloop=ShadowMidloopDecider(primary, shadow),
    )
    # Build looping trajectory: heuristic shadow would intervene.
    for i in range(4):
        mem.observe_step(_obs(i, output="same content same content"))
    final = mem.observe_step(
        _obs(4, output="same content same content again"),
        is_last_step=True,
    )
    # Primary (Noop) wins -> no intervention at runtime.
    assert final.intervene is False
    # Shadow (Heuristic) disagreed -> annotation in reason.
    assert "shadow_disagree" in final.reason
    # Audit persists because is_last_step=True.
    hits = mem.audit("shadow_disagree", top_k=10, fts_only=True)
    assert len(hits) >= 1
