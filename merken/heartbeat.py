"""Heartbeat — periodic memory lifecycle daemon.

The heartbeat runs ``Memory.consolidate()`` and ``Memory.forget()`` on
a configurable schedule, writes heartbeat audit entries, and reports
health. It is the background-maintenance half of the memory loop.

Designed for:
- Long-running agents that accumulate events during a session
- MCP / gateway deployments where memory should self-maintain
- Deployments where manual ``mem.consolidate()`` / ``mem.forget()``
  calls are easy to forget

Usage::

    from merken import Heartbeat, Memory

    with Memory(project="my_agent") as mem:
        hb = Heartbeat(memory=mem)
        hb.run()  # blocks until SIGINT/SIGTERM

Or from the CLI::

    $ merken heartbeat --consolidate-interval 300

Design invariants
-----------------

- **Non-destructive by default.** The heartbeat runs the decision
  primitives through the same policies as manual calls. It never
  force-consolidates or force-forgets unless told to (``force=True``).
- **No new dependencies.** stdlib only — ``signal``, ``time``,
  ``threading``, ``logging``.
- **Audited.** Every heartbeat tick writes a ``should_heartbeat``
  audit row so the operator can reconstruct what happened and when.
- **No recursion.** Heartbeat audit rows go directly to vstash,
  bypassing the merken loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from merken.audit import AUDIT_COLLECTION, AUDIT_LAYER

if TYPE_CHECKING:
    from merken import ConsolidationResult, ForgetResult, Memory

logger = logging.getLogger(__name__)

#: Default intervals (seconds). Zero = skip that phase.
_DEFAULT_CONSOLIDATE_INTERVAL: int = 300  # 5 min
_DEFAULT_FORGET_INTERVAL: int = 3600  # 1 hr
_DEFAULT_REPORT_INTERVAL: int = 3600  # 1 hr
_DEFAULT_MIN_EVENTS: int = 5


@dataclass
class HeartbeatReport:
    """Summary of one heartbeat tick."""

    tick: int
    timestamp: str
    episodic_count: int
    semantic_count: int
    consolidated: bool
    consolidation_result: ConsolidationResult | None
    forgot: bool
    forget_result: ForgetResult | None
    elapsed: float
    errors: list[str] = field(default_factory=list)


class Heartbeat:
    """Periodic memory maintenance loop.

    Call ``run()`` to enter the heartbeat loop. The loop processes
    consolidation and forget phases on separate schedules, logs
    health reports, and handles graceful shutdown via SIGINT/SIGTERM.

    Parameters
    ----------
    memory:
        The merken ``Memory`` instance to maintain.
    consolidate_interval:
        Minimum seconds between consolidation attempts. 0 = never.
    forget_interval:
        Minimum seconds between forget attempts. 0 = never.
    report_interval:
        Minimum seconds between health-report ticks (always written
        to the audit log). 0 = never.
    min_events:
        Minimum number of episodic events before consolidation runs.
    force:
        When True, bypass the ``should_consolidate`` and
        ``should_forget`` deciders (equivalent to passing
        ``force=True`` to those methods).
    """

    def __init__(
        self,
        memory: Memory,
        *,
        consolidate_interval: int = _DEFAULT_CONSOLIDATE_INTERVAL,
        forget_interval: int = _DEFAULT_FORGET_INTERVAL,
        report_interval: int = _DEFAULT_REPORT_INTERVAL,
        min_events: int = _DEFAULT_MIN_EVENTS,
        force: bool = False,
    ) -> None:
        self._mem = memory
        self._consolidate_interval = consolidate_interval
        self._forget_interval = forget_interval
        self._report_interval = report_interval
        self._min_events = min_events
        self._force = force

        self._tick_count = 0
        now = time.time()
        self._last_consolidate: float = now
        self._last_forget: float = now
        self._last_report: float = now
        self._shutdown = False
        self._last_report_data: HeartbeatReport | None = None
        self._original_sigint: object = None
        self._original_sigterm: object = None

    # ------------------------------------------------------------------ public

    def run(self, poll_interval: float = 10.0) -> None:
        """Enter the heartbeat loop. Blocks until shutdown.

        Parameters
        ----------
        poll_interval:
            Seconds between each tick check. Smaller = more responsive
            shutdown, larger = fewer no-op ticks during idle periods.
            Note: the actual sleep is capped at 1.0s per iteration
            to remain responsive to shutdown signals. Values > 1.0
            are treated as 1.0 for sleep duration. Use chunked
            sleeping — the loop repeatedly sleeps min(1.0, remaining)
            until the interval elapses or shutdown is requested.
        """
        self._install_signal_handlers()
        logger.info(
            "Heartbeat started: consolidate=%ds forget=%ds report=%ds min_events=%d",
            self._consolidate_interval,
            self._forget_interval,
            self._report_interval,
            self._min_events,
        )

        try:
            while not self._shutdown:
                self._tick()
                # Chunked sleep: honor poll_interval but stay responsive
                chunk_end = time.time() + poll_interval
                while time.time() < chunk_end and not self._shutdown:
                    remaining = chunk_end - time.time()
                    time.sleep(min(1.0, remaining))
        except KeyboardInterrupt:
            logger.info("Heartbeat stopped (KeyboardInterrupt)")
        finally:
            self._restore_signal_handlers()

    def shutdown(self) -> None:
        """Request graceful shutdown. Signal-safe."""
        self._shutdown = True

    # ----------------------------------------------------------------- reports

    @property
    def report(self) -> HeartbeatReport | None:
        """The most recent tick report, or None if no tick has run."""
        return self._last_report_data

    # ----------------------------------------------------------------- private

    def _install_signal_handlers(self) -> None:
        """Install handlers that set the shutdown flag."""
        import signal as _signal

        try:
            self._original_sigint = _signal.getsignal(_signal.SIGINT)
            self._original_sigterm = _signal.getsignal(_signal.SIGTERM)

            def _handler(signum: object, frame: object) -> None:  # noqa: ARG001
                self.shutdown()

            _signal.signal(_signal.SIGINT, _handler)
            _signal.signal(_signal.SIGTERM, _handler)
        except (ValueError, RuntimeError):
            pass

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers (best-effort)."""
        import signal as _signal

        try:
            if self._original_sigint is not None:
                _signal.signal(_signal.SIGINT, self._original_sigint)  # type: ignore[arg-type]
            if self._original_sigterm is not None:
                _signal.signal(_signal.SIGTERM, self._original_sigterm)  # type: ignore[arg-type]
        except (ValueError, RuntimeError):
            pass

    def _tick(self) -> None:
        """Run one heartbeat tick."""
        self._tick_count += 1
        errors: list[str] = []
        now = time.time()

        # --- Phase: count -------------------------------------------------
        try:
            episodic = list(self._mem._vstash.list(  # noqa: SLF001
                collection=self._mem.collection,
                layer="episodic",
            ))
            semantic = list(self._mem._vstash.list(  # noqa: SLF001
                collection=self._mem.collection,
                layer="semantic",
            ))
        except Exception as e:
            logger.warning("Heartbeat: list failed: %s", e)
            errors.append(f"list:{e}")
            # Write partial audit and bail — listing is the entry gate.
            self._last_report_data = HeartbeatReport(
                tick=self._tick_count,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                episodic_count=0,
                semantic_count=0,
                consolidated=False,
                consolidation_result=None,
                forgot=False,
                forget_result=None,
                elapsed=0.0,
                errors=errors,
            )
            self._write_audit(
                errors=errors,
                elapsed=0.0,
                episodic_count=0,
                semantic_count=0,
            )
            return

        n_episodic = len(episodic)
        n_semantic = len(semantic)

        # --- Phase: consolidate -------------------------------------------
        consolidation_result: ConsolidationResult | None = None
        did_consolidate = False

        if (
            self._consolidate_interval > 0
            and n_episodic >= self._min_events
            and now - self._last_consolidate >= self._consolidate_interval
        ):
            try:
                consolidation_result = self._mem.consolidate(force=self._force)
                self._last_consolidate = now
                did_consolidate = True
                if consolidation_result.facts_written > 0:
                    logger.info(
                        "Heartbeat: consolidated %d events into %d facts",
                        consolidation_result.events_examined,
                        consolidation_result.facts_written,
                    )
            except Exception as e:
                logger.warning("Heartbeat: consolidate failed: %s", e)
                errors.append(f"consolidate:{e}")

        # --- Phase: forget ------------------------------------------------
        forget_result: ForgetResult | None = None
        did_forget = False

        if (
            self._forget_interval > 0
            and (self._force or n_semantic > 0)
            and now - self._last_forget >= self._forget_interval
        ):
            try:
                forget_result = self._mem.forget(force=self._force)
                self._last_forget = now
                did_forget = True
                if forget_result.tombstoned:
                    logger.info(
                        "Heartbeat: tombstoned %d events",
                        len(forget_result.tombstoned),
                    )
            except Exception as e:
                logger.warning("Heartbeat: forget failed: %s", e)
                errors.append(f"forget:{e}")

        # --- Phase: report ------------------------------------------------
        elapsed = time.time() - now
        should_report = (
            did_consolidate
            or did_forget
            or (
                self._report_interval > 0
                and now - self._last_report >= self._report_interval
            )
        )

        if should_report:
            self._last_report = now
            report = HeartbeatReport(
                tick=self._tick_count,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                episodic_count=n_episodic,
                semantic_count=n_semantic,
                consolidated=did_consolidate,
                consolidation_result=consolidation_result,
                forgot=did_forget,
                forget_result=forget_result,
                elapsed=elapsed,
                errors=errors,
            )
            self._last_report_data = report
            self._write_audit(
                errors=errors,
                elapsed=elapsed,
                episodic_count=n_episodic,
                semantic_count=n_semantic,
            )

    def _write_audit(
        self,
        *,
        errors: list[str],
        elapsed: float,
        episodic_count: int,
        semantic_count: int,
    ) -> None:
        """Write one heartbeat audit row, bypassing the merken loop."""
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        body = (
            f"timestamp: {ts}\n"
            f"decision: should_heartbeat\n"
            f"tick: {self._tick_count}\n"
            f"episodic_count: {episodic_count}\n"
            f"semantic_count: {semantic_count}\n"
            f"errors: {'; '.join(errors) if errors else '(none)'}\n"
            f"elapsed: {elapsed:.3f}s\n"
        )
        title = f"audit:should_heartbeat:tick{self._tick_count}:{ts}"
        try:
            self._mem._vstash.remember(  # noqa: SLF001
                body,
                title=title,
                collection=AUDIT_COLLECTION,
                layer=AUDIT_LAYER,
            )
        except Exception as e:
            # Audit failures must never break the loop, but should be visible.
            logger.warning("Heartbeat: audit write failed: %s", e)