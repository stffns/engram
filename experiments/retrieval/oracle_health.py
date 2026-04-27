"""Oracle health guard for benchmark runners.

Background: 2026-04-27 audit found that LLM oracles (Gemini 2.5 Flash
in particular) silently fail with 429/RESOURCE_EXHAUSTED when monthly
quotas are hit, and our runners stored those errors as
`verdict: "neutral"` with a rationale starting "oracle_error". Counting
verdicts alone treated those as real refusals, producing false-negative
findings (e.g. "hybrid weights destroys LME 0/30" -- after Cerebras
rejudge: 17/30 = 56.7%, identical to baseline).

This guard catches the failure mode at three points:

1.  Pre-flight ping: before the main loop, score one trivial
    (q, gt, candidate) triple. If the oracle returns an error, abort
    with a clear message instead of silently corrupting an N=200 run.

2.  First-window guard: count errors in the first ``first_window``
    questions; abort if > ``max_errors_in_first``. Catches mid-run
    quota exhaustion early so we don't burn an hour on a dead oracle.

3.  Summary refuse: at the end of the run, the ``summary_safe``
    property returns False if any oracle error has been recorded.
    Runners use this to suppress the headline ``correct%`` print and
    emit a warning, forcing the operator to rejudge before citing a
    number.

Runner usage::

    from experiments.retrieval.oracle_health import OracleHealthGuard

    oracle = _CerebrasOracle()  # or _oracle_client()
    health = OracleHealthGuard()
    health.preflight(score_fn=lambda q, gt, ans: oracle.score(q, gt, ans))

    for qa in qas:
        ...
        verdict = oracle.score(q, gt, ans)
        health.record(verdict)  # may raise after first-window
        ...

    if not health.summary_safe:
        print(health.warning())
    else:
        print(f"correct: {correct}/{total} = {correct/total*100:.1f}%")
"""
from __future__ import annotations

from typing import Any, Callable


# Substrings that mark a verdict as oracle-side failure rather than a
# real model refusal. Match against ``oracle.rationale``.
ORACLE_ERROR_MARKERS: tuple[str, ...] = (
    "oracle_error",
    "RESOURCE_EXHAUSTED",
    "429",
    "oracle_timeout",
    "rate_limit",
)
# Note: the spaced variant "rate limit" was deliberately omitted: it
# can plausibly appear in free-text Gemini rationales without indicating
# an outage. The other markers only appear in our own error-injection
# paths (oracle_score / _CerebrasOracle / _alarm), so they are safe
# substring matches.


def is_oracle_error(verdict: dict[str, Any] | None) -> bool:
    """Return True iff ``verdict.rationale`` matches a known oracle-error
    marker. Empty/missing rationale counts as not-an-error.
    """
    if not verdict:
        return False
    rationale = verdict.get("rationale") or ""
    return any(m in rationale for m in ORACLE_ERROR_MARKERS)


class OracleHealthError(RuntimeError):
    """Raised by ``OracleHealthGuard`` when the oracle is unhealthy."""


class OracleHealthGuard:
    """Tracks oracle-side errors during a benchmark run and aborts when
    the failure rate would silently corrupt the result.

    Parameters
    ----------
    first_window : int
        Number of questions to watch closely at the start of the run.
    max_errors_in_first : int
        Abort if oracle error count exceeds this within ``first_window``.
    max_total_pct : float
        Abort if cumulative error rate exceeds this fraction once
        ``min_total_for_pct_check`` rows have been seen.
    min_total_for_pct_check : int
        Floor below which the percent-rate check is skipped (avoids
        2/3 = 66% triggering on a tiny sample).
    """

    def __init__(
        self,
        first_window: int = 10,
        max_errors_in_first: int = 2,
        max_total_pct: float = 0.05,
        min_total_for_pct_check: int = 30,
    ) -> None:
        self.first_window = first_window
        self.max_errors_in_first = max_errors_in_first
        self.max_total_pct = max_total_pct
        self.min_total_for_pct_check = min_total_for_pct_check
        self.errors = 0
        self.total = 0

    def preflight(
        self,
        score_fn: Callable[[str, str, str], dict[str, Any]],
    ) -> None:
        """Issue one trivial oracle call to confirm the backend is alive.
        Aborts before the main loop runs if it isn't.
        """
        verdict = score_fn(
            "What color is the sky in this context?",
            "blue",
            "The sky is blue.",
        )
        if is_oracle_error(verdict):
            raise OracleHealthError(
                f"oracle pre-flight failed: {verdict.get('rationale', verdict)!r}. "
                f"Refusing to start the run -- check oracle quota / API key / "
                f"network."
            )

    def record(self, verdict: dict[str, Any] | None) -> None:
        """Record one oracle response. Raises ``OracleHealthError`` if
        the run has crossed an unhealthy threshold.
        """
        self.total += 1
        if is_oracle_error(verdict):
            self.errors += 1

        if (
            self.total <= self.first_window
            and self.errors > self.max_errors_in_first
        ):
            raise OracleHealthError(
                f"oracle health: {self.errors}/{self.total} errors in the "
                f"first {self.first_window} questions (threshold "
                f"{self.max_errors_in_first}). Aborting -- the run would "
                f"produce contaminated percentages. Check the oracle backend."
            )

        if (
            self.total >= self.min_total_for_pct_check
            and self.errors / self.total > self.max_total_pct
        ):
            raise OracleHealthError(
                f"oracle health: {self.errors}/{self.total} errors "
                f"({self.errors / self.total * 100:.1f}%) exceeds "
                f"{self.max_total_pct * 100:.0f}% threshold. Aborting."
            )

    @property
    def summary_safe(self) -> bool:
        """True iff zero oracle errors have been recorded."""
        return self.errors == 0

    def warning(self) -> str:
        """Multi-line warning message for contaminated runs. Empty
        string when the run is clean.
        """
        if self.summary_safe:
            return ""
        return (
            f"WARNING: {self.errors}/{self.total} oracle calls returned "
            f"errors (rationale matched {ORACLE_ERROR_MARKERS!r}). "
            f"The cited correct% / trust_score are CONTAMINATED -- the "
            f"failed verdicts were stored as 'neutral'. Rejudge with a "
            f"different oracle (see "
            f"experiments/retrieval/locomo/rejudge_with_cerebras.py) "
            f"before reporting any number from this artifact."
        )
