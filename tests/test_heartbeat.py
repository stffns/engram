"""Tests for the Heartbeat daemon.

These test the Heartbeat in isolation through its public methods,
and with minimal tick cycles to verify consolidation/forget triggers.
Each test uses an isolated tmp_path SQLite.
"""

from __future__ import annotations

import time
from pathlib import Path

from merken import Heartbeat, Memory
from merken.heartbeat import HeartbeatReport


def test_heartbeat_attributes(tmp_path: Path) -> None:
    """Heartbeat instantiates with sensible defaults."""
    db = tmp_path / "merken.db"
    with Memory(project="test_hb", db=db) as mem:
        hb = Heartbeat(memory=mem)
        assert hb is not None
        assert hb._consolidate_interval == 300
        assert hb._forget_interval == 3600
        assert hb._min_events == 5
        assert not hb._shutdown


def test_heartbeat_custom_intervals(tmp_path: Path) -> None:
    """Heartbeat accepts custom intervals."""
    db = tmp_path / "merken.db"
    with Memory(project="test_hb2", db=db) as mem:
        hb = Heartbeat(
            memory=mem,
            consolidate_interval=0,
            forget_interval=0,
            report_interval=0,
            min_events=1,
        )
        assert hb._consolidate_interval == 0
        assert hb._forget_interval == 0
        assert hb._min_events == 1
        assert hb._report_interval == 0


def test_heartbeat_shutdown_flag(tmp_path: Path) -> None:
    """Setting shutdown stops the loop."""
    db = tmp_path / "merken.db"
    with Memory(project="test_hb3", db=db) as mem:
        hb = Heartbeat(memory=mem, consolidate_interval=0, forget_interval=0, report_interval=0)
        assert not hb._shutdown
        hb.shutdown()
        assert hb._shutdown


def test_heartbeat_writes_audit_row(tmp_path: Path) -> None:
    """A heartbeat tick writes an audit row."""
    db = tmp_path / "merken.db"
    with Memory(project="test_hb4", db=db) as mem:
        hb = Heartbeat(memory=mem, consolidate_interval=0, forget_interval=0, report_interval=0)
        hb._min_events = 100  # prevent auto-consolidate from triggering
        hb._tick()

        # With all intervals=0 and no consolidation/forget, no audit row
        # is written. Force a report to test the audit path.
        hb._last_report = 0  # reset so next tick generates a report
        # Set a short report interval so the next tick fires
        save_report = hb._report_interval
        hb._report_interval = 300  # will fire since last_report=0
        hb._tick()
        hb._report_interval = save_report

        # Verify audit row written
        audit = mem.audit("should_heartbeat", top_k=10)
        found = any(
            "should_heartbeat" in (h.text or "") for h in audit
        )
        assert found, "heartbeat audit row not found"


def test_heartbeat_counts_events(tmp_path: Path) -> None:
    """A heartbeat tick counts episodic and semantic events correctly."""
    db = tmp_path / "merken.db"
    with Memory(project="test_hb5", db=db) as mem:
        hb = Heartbeat(memory=mem, consolidate_interval=0, forget_interval=0, report_interval=0)

        mem.remember("Event one about pineapples.", layer="episodic")
        mem.remember("Event two about pineapples.", layer="episodic")
        mem.remember("Semantic fact: user likes pineapples.", layer="semantic")

        # Force report by resetting the timer
        hb._last_report = 0
        save = hb._report_interval
        hb._report_interval = 300
        hb._tick()
        hb._report_interval = save

        assert hb._last_report_data is not None
        assert hb._last_report_data.episodic_count == 2
        assert hb._last_report_data.semantic_count == 1


def test_heartbeat_consolidates_when_threshold_met(tmp_path: Path) -> None:
    """Heartbeat runs consolidation when enough events exist."""
    db = tmp_path / "merken.db"
    with Memory(project="test_hb6", db=db) as mem:
        hb = Heartbeat(
            memory=mem,
            consolidate_interval=1,  # 1s = fire every tick (after initial delay)
            forget_interval=1,
            report_interval=0,
            min_events=2,
        )

        # Add enough similar events for consolidation to cluster
        mem.remember("The user switched to Postgres on 2026-04-08 for the analytics pipeline.",
                     layer="episodic")
        mem.remember("The analytics warehouse now runs on Postgres 16 according to the infra team.",
                     layer="episodic")
        # This one is different — won't cluster with the Postgres ones
        mem.remember("The user prefers teal color schemes in the dashboard.",
                     layer="episodic")

        hb._last_consolidate = 0  # force fire
        hb._last_report = 0
        save = hb._report_interval
        hb._report_interval = 300
        hb._tick()
        hb._report_interval = save

        assert hb._last_report_data is not None
        assert hb._last_report_data.consolidated


def test_heartbeat_skips_consolidation_below_min_events(tmp_path: Path) -> None:
    """Heartbeat does not consolidate when events are below threshold."""
    db = tmp_path / "merken.db"
    with Memory(project="test_hb7", db=db) as mem:
        hb = Heartbeat(
            memory=mem,
            consolidate_interval=300,
            forget_interval=0,
            report_interval=0,
            min_events=10,
        )

        mem.remember("Only one event.", layer="episodic")

        hb._last_report = 0
        save = hb._report_interval
        hb._report_interval = 300
        hb._tick()
        hb._report_interval = save

        assert hb._last_report_data is not None
        assert not hb._last_report_data.consolidated


def test_heartbeat_noop_with_zero_intervals(tmp_path: Path) -> None:
    """With all intervals at 0, the heartbeat is a no-op but still safe."""
    db = tmp_path / "merken.db"
    with Memory(project="test_hb8", db=db) as mem:
        hb = Heartbeat(memory=mem, consolidate_interval=0, forget_interval=0, report_interval=0)
        hb._min_events = 100  # prevent auto-consolidate
        # Force a report to verify the report is clean
        hb._last_report = 0
        save = hb._report_interval
        hb._report_interval = 300
        hb._tick()
        hb._report_interval = save

        assert hb._last_report_data is not None
        assert not hb._last_report_data.consolidated
        assert not hb._last_report_data.forgot
        assert hb._last_report_data.errors == []


def test_heartbeat_tick_monotonic(tmp_path: Path) -> None:
    """Tick counter increments monotonically."""
    db = tmp_path / "merken.db"
    with Memory(project="test_hb9", db=db) as mem:
        hb = Heartbeat(memory=mem, consolidate_interval=0, forget_interval=0, report_interval=0)
        assert hb._tick_count == 0

        hb._tick()
        assert hb._tick_count == 1

        hb._tick()
        assert hb._tick_count == 2

        hb._tick()
        assert hb._tick_count == 3


def test_heartbeat_with_forget(tmp_path: Path) -> None:
    """Heartbeat runs forget after successful consolidation."""
    db = tmp_path / "merken.db"
    with Memory(
        project="test_hb10",
        db=db,
        forget_decider=None,
    ) as mem:
        hb = Heartbeat(
            memory=mem,
            consolidate_interval=1,
            forget_interval=1,
            report_interval=0,
            min_events=1,
            force=True,
        )

        mem.remember("Event about temporal mechanics.", layer="episodic")

        hb._last_consolidate = 0  # force fire
        hb._last_forget = 0  # force fire
        hb._last_report = 0
        save = hb._report_interval
        hb._report_interval = 300
        hb._tick()
        hb._report_interval = save

        assert hb._last_report_data is not None
        assert hb._last_report_data.forgot


def test_heartbeat_report_dataclass() -> None:
    """HeartbeatReport dataclass fields are correct."""
    report = HeartbeatReport(
        tick=1,
        timestamp="2026-05-26T12:00:00",
        episodic_count=10,
        semantic_count=2,
        consolidated=True,
        consolidation_result=None,
        forgot=False,
        forget_result=None,
        elapsed=0.5,
        errors=[],
    )
    assert report.tick == 1
    assert report.episodic_count == 10
    assert report.consolidated
    assert not report.forgot
    assert report.elapsed == 0.5


def test_heartbeat_tick_errors_recorded(tmp_path: Path) -> None:
    """Errors during a tick are captured in the report."""
    db = tmp_path / "merken.db"
    with Memory(project="test_hb11", db=db) as mem:
        hb = Heartbeat(memory=mem, consolidate_interval=0, forget_interval=0, report_interval=0)

        # Force a failure by closing the underlying store
        mem.close()
        hb._last_report = 0
        save = hb._report_interval
        hb._report_interval = 300
        hb._tick()
        hb._report_interval = save

        assert hb._last_report_data is not None
        assert len(hb._last_report_data.errors) > 0


def test_heartbeat_respects_interval_timing(tmp_path: Path) -> None:
    """Heartbeat does not consolidate or forget before its interval elapses."""
    db = tmp_path / "merken.db"
    with Memory(project="test_hb12", db=db) as mem:
        hb = Heartbeat(
            memory=mem,
            consolidate_interval=36000,  # very long interval
            forget_interval=36000,
            report_interval=0,
            min_events=1,
        )

        mem.remember("Test event for timing.", layer="episodic")

        # First tick — should NOT consolidate because _last_consolidate
        # was initialized to time.time() and interval is 10 hours
        hb._last_report = 0
        save = hb._report_interval
        hb._report_interval = 300
        hb._tick()
        hb._report_interval = save

        assert hb._last_report_data is not None
        assert not hb._last_report_data.consolidated, (
            "should not consolidate before interval elapses"
        )

        # Force the last consolidate time to be way in the past
        import time
        hb._last_consolidate = time.time() - 72000  # 20 hours ago
        hb._tick()

        # Now it should have consolidated
        assert hb._last_report_data.consolidated


def test_heartbeat_run_stops_on_shutdown(tmp_path: Path) -> None:
    """Heartbeat.run() exits quickly when shutdown is called."""
    import threading

    db = tmp_path / "merken.db"
    with Memory(project="test_hb13", db=db) as mem:
        hb = Heartbeat(
            memory=mem,
            consolidate_interval=0,
            forget_interval=0,
            report_interval=0,
        )

        def _stop() -> None:
            time.sleep(0.1)
            hb.shutdown()

        t = threading.Thread(target=_stop, daemon=True)
        t.start()

        start = time.time()
        hb.run(poll_interval=0.05)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"run() took too long to exit: {elapsed:.2f}s"