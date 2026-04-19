"""``should_intervene`` policies -- the fifth decision primitive.

The midloop sits between the Builder (inner loop) and the Judge
(outer loop). Builder generates output; Judge evaluates final
artifacts. Midloop watches the trajectory in flight and decides
whether to intervene mid-stream (whisper guidance, escalate,
rollback, or abort) before the Builder wastes more compute on a
trajectory that has gone bad.

Full design context: ``notes/midloop-spec.md``.

This module ships the **sketch** scaffolding -- Protocol +
dataclasses + three reference deciders. Production deployment
follows the same shadow-mode graduation path that
``NanoGPTWriteDecider`` did:

1. ``NoopMidloopDecider`` as primary (never intervenes), live in
   the audit log only. Acquires real (decision, task_outcome)
   pairs.
2. ``HeuristicMidloopDecider`` as shadow, observing every step.
   Disagreements with the noop primary surface as
   ``midloop_disagree`` audit tags for review.
3. After ~200 labeled disagreements, calibrate Heuristic
   thresholds against the labels. Only THEN promote Heuristic to
   primary, with action clamped to ``WHISPER`` (cheapest, most
   reversible).
4. Eventually a ``NanoGPTMidloopDecider`` trained on the labels
   replaces or augments the Heuristic.

The thresholds in ``HeuristicMidloopDecider`` are explicitly
PLACEHOLDERS until step 3 above produces real calibration data.
They are documented as such in the constructor and in the audit
log (the reason field includes the threshold values used) so a
reader can trace which numbers shipped at which point.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


# ---------------------------------------------------------------- enums

class TaskCategory(str, Enum):
    """How the task type affects whether the midloop should fire.

    Per the midloop spec sec "Politica de activacion":
    - STRUCTURED: deterministic stopping criterion (compile, test,
      DB query). Midloop monitoring on these tasks usually HURTS
      (-0.29 d effect size in the cited Cognitive Companion paper).
    - EXPLORATORY: open-ended (research, debugging without repro,
      synthesis). Midloop helps most here.
    - MIXED: only fire after step_index >= activation threshold.
    """
    STRUCTURED = "structured"
    EXPLORATORY = "exploratory"
    MIXED = "mixed"


class CognitiveState(str, Enum):
    """The labeled cognitive state the midloop assigned to this step."""
    ON_TRACK = "on_track"
    LOOPING = "looping"      # repetition, semantic or lexical
    DRIFTING = "drifting"    # output relevance to task drops
    STUCK = "stuck"          # progress reduces (latency up, output shrinks)


class InterventionAction(str, Enum):
    """The action the midloop recommends, ordered by cost."""
    NONE = "none"
    WHISPER = "whisper"       # silent guidance injection (cheapest)
    ESCALATE = "escalate"     # notify Judge for partial replanning
    ROLLBACK = "rollback"     # revert to last valid commit (Temporal)
    ABORT = "abort"           # terminate task, report failure


# ---------------------------------------------------------------- dataclasses

@dataclass
class ToolCall:
    """One tool invocation summary inside a step."""
    name: str
    args_summary: str = ""
    result_summary: str = ""


@dataclass
class StepObservation:
    """One step in the Builder's execution -- midloop input unit."""
    step_id: str
    task_id: str
    task_category: TaskCategory
    timestamp: float                          # unix seconds
    output_text: str                          # what the model emitted this step
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: int = 0
    tokens_used: int = 0
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MidloopContext:
    """Context for a midloop decision.

    ``trajectory_size`` is the number of prior observations the
    decider can see (NOT this one). ``task_description`` is the
    user's original ask, used by relevance signals when computing
    drift.
    """
    project: str
    task_id: str
    task_category: TaskCategory
    trajectory_size: int = 0
    task_description: str = ""


@dataclass(frozen=True)
class MidloopDecision:
    """The output of a midloop decider.

    ``intervene`` is the only field merken's downstream caller
    acts on. The rest are for the audit log -- a future reviewer
    needs ``signals`` to recompute thresholds, ``state`` to label
    the failure mode, and ``reason`` to read what happened
    without parsing signal numbers.
    """
    intervene: bool
    action: InterventionAction
    confidence: float
    reason: str
    state: CognitiveState
    policy: str
    signals: dict[str, float] = field(default_factory=dict)


class MidloopDecider(Protocol):
    """Protocol for ``should_intervene`` policies."""
    name: str

    def decide(
        self,
        observation: StepObservation,
        ctx: MidloopContext,
        trajectory: list[StepObservation],
    ) -> MidloopDecision: ...


# ---------------------------------------------------------------- noop

class NoopMidloopDecider:
    """Baseline: never intervenes.

    Use as PRIMARY during shadow-mode bootstrap. The primary
    decision controls runtime behavior; the heuristic shadow
    annotates the audit log with what IT would have decided.
    Disagreements between primary (Noop) and shadow (Heuristic)
    are the labeling pool: each one needs a human or oracle to
    judge "should the midloop have intervened here?"
    """

    name = "NoopMidloopDecider"

    def decide(
        self,
        observation: StepObservation,
        ctx: MidloopContext,
        trajectory: list[StepObservation],
    ) -> MidloopDecision:  # noqa: ARG002
        return MidloopDecision(
            intervene=False,
            action=InterventionAction.NONE,
            confidence=1.0,
            reason="noop",
            state=CognitiveState.ON_TRACK,
            policy=self.name,
        )


# ---------------------------------------------------------------- heuristic

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not (a or b):
        return 0.0
    return len(a & b) / len(a | b)


def _max_pairwise_jaccard(texts: Iterable[str]) -> float:
    token_sets = [_tokenize(t) for t in texts]
    n = len(token_sets)
    if n < 2:
        return 0.0
    best = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            score = _jaccard(token_sets[i], token_sets[j])
            if score > best:
                best = score
    return best


def _length_trend(observations: list[StepObservation]) -> float:
    """Negative slope = output shrinking (potential STUCK).

    Returns the linear slope of len(output_text) over step_index,
    normalized to [-1, 1] by dividing by the max length seen.
    """
    if len(observations) < 2:
        return 0.0
    xs = [float(o.step_index) for o in observations]
    ys = [float(len(o.output_text)) for o in observations]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = sum((x - mx) ** 2 for x in xs) or 1.0
    slope = num / den
    max_y = max(ys) or 1.0
    return max(-1.0, min(1.0, slope / max_y))


def _latency_trend(observations: list[StepObservation]) -> float:
    """Positive slope = latency growing (potential STUCK)."""
    if len(observations) < 2:
        return 0.0
    xs = [float(o.step_index) for o in observations]
    ys = [float(o.latency_ms) for o in observations]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = sum((x - mx) ** 2 for x in xs) or 1.0
    slope = num / den
    max_y = max(ys) or 1.0
    return max(-1.0, min(1.0, slope / max_y))


def _tool_diversity(observations: list[StepObservation]) -> float:
    """Lower = same tool repeatedly (potential STUCK).

    Returns the fraction of unique tool names over total tool
    calls in the window. 1.0 = every tool different. 0.0 = no
    tools called at all (treated as max diversity / on-track).
    """
    names = [tc.name for o in observations for tc in o.tool_calls]
    if not names:
        return 1.0
    return len(set(names)) / len(names)


def _task_relevance(
    observation: StepObservation, task_description: str,
) -> float:
    """Token-set Jaccard between this step's output and the task ask.

    Cheap proxy for "did the model drift away from the original
    task". A value > 0 means there is at least token overlap.
    Replace with embedding-cosine when a real embedder is wired
    here without circular imports.
    """
    if not task_description:
        return 1.0
    return _jaccard(_tokenize(observation.output_text), _tokenize(task_description))


@dataclass
class HeuristicThresholds:
    """All knobs for HeuristicMidloopDecider in one place.

    Defaults are PLACEHOLDERS (Jay 2026-04-19): they have not been
    calibrated against real shadow-mode labels yet. Adjust only
    after >= 200 (decision, task_outcome) pairs are in the audit
    log AND a held-out subset confirms the new values do not
    regress recall on real failures. Until then the policy ships
    in shadow mode, never as primary.
    """
    jaccard_looping: float = 0.5         # >= => LOOPING
    length_trend_stuck: float = -0.4     # <= => STUCK (output shrinking)
    latency_trend_stuck: float = 0.6     # >= => STUCK (latency growing)
    tool_diversity_stuck: float = 0.2    # <= => STUCK (same tool repeated)
    relevance_drifting: float = 0.05     # < => DRIFTING (drift from task)
    min_steps_to_decide: int = 3         # need at least N prior steps
    window_size: int = 5                 # last N observations considered


class HeuristicMidloopDecider:
    """Pure-Python signals decider. No LLM, no model load.

    Computes four signals over a sliding window:
      - jaccard_last_n: max pairwise lexical overlap (LOOPING)
      - length_trend: linear slope of output length (STUCK if shrinking)
      - latency_trend: linear slope of step latency (STUCK if growing)
      - tool_diversity: unique tools / total calls (STUCK if repeating)
      - task_relevance: jaccard with task description (DRIFTING if low)

    First state to fire wins; falls through to ON_TRACK if none do.
    All thresholds documented in HeuristicThresholds.

    State -> action mapping (cheapest action per state):
      LOOPING / DRIFTING / STUCK -> WHISPER
      (ESCALATE / ROLLBACK / ABORT are reserved for future
       deciders that have higher-confidence signals.)
    """

    name = "HeuristicMidloopDecider"

    def __init__(self, thresholds: HeuristicThresholds | None = None) -> None:
        self.thresholds = thresholds or HeuristicThresholds()

    def decide(
        self,
        observation: StepObservation,
        ctx: MidloopContext,
        trajectory: list[StepObservation],
    ) -> MidloopDecision:
        # STRUCTURED tasks: opt out by default. Empirical evidence
        # cited in the spec shows midloop monitoring HURTS structured
        # tasks (-0.29 d). Override by registering the task as
        # EXPLORATORY or MIXED.
        if ctx.task_category == TaskCategory.STRUCTURED:
            return MidloopDecision(
                intervene=False,
                action=InterventionAction.NONE,
                confidence=1.0,
                reason="structured_task_skip",
                state=CognitiveState.ON_TRACK,
                policy=self.name,
            )

        if len(trajectory) < self.thresholds.min_steps_to_decide:
            return MidloopDecision(
                intervene=False,
                action=InterventionAction.NONE,
                confidence=0.5,
                reason=(
                    f"insufficient_history:"
                    f"{len(trajectory)}<{self.thresholds.min_steps_to_decide}"
                ),
                state=CognitiveState.ON_TRACK,
                policy=self.name,
            )

        window = (trajectory + [observation])[-self.thresholds.window_size:]
        signals = {
            "jaccard_last_n": _max_pairwise_jaccard(o.output_text for o in window),
            "length_trend": _length_trend(window),
            "latency_trend": _latency_trend(window),
            "tool_diversity": _tool_diversity(window),
            "task_relevance": _task_relevance(observation, ctx.task_description),
        }

        # Decision tree -- first state to fire wins. Ordered by
        # actionability: looping is the easiest to detect and the
        # most likely to benefit from a whisper.
        if signals["jaccard_last_n"] >= self.thresholds.jaccard_looping:
            return self._intervene(
                state=CognitiveState.LOOPING,
                signals=signals,
                trigger="jaccard_last_n",
                threshold=self.thresholds.jaccard_looping,
            )
        if signals["task_relevance"] < self.thresholds.relevance_drifting:
            return self._intervene(
                state=CognitiveState.DRIFTING,
                signals=signals,
                trigger="task_relevance",
                threshold=self.thresholds.relevance_drifting,
            )
        stuck_triggers = []
        if signals["length_trend"] <= self.thresholds.length_trend_stuck:
            stuck_triggers.append("length_trend")
        if signals["latency_trend"] >= self.thresholds.latency_trend_stuck:
            stuck_triggers.append("latency_trend")
        if signals["tool_diversity"] <= self.thresholds.tool_diversity_stuck:
            stuck_triggers.append("tool_diversity")
        if stuck_triggers:
            return self._intervene(
                state=CognitiveState.STUCK,
                signals=signals,
                trigger=",".join(stuck_triggers),
                threshold=0.0,  # multi-trigger; reason carries detail
            )

        return MidloopDecision(
            intervene=False,
            action=InterventionAction.NONE,
            confidence=1.0 - max(
                signals["jaccard_last_n"],
                1.0 - signals["task_relevance"],
            ),
            reason="on_track",
            state=CognitiveState.ON_TRACK,
            policy=self.name,
            signals=signals,
        )

    def _intervene(
        self,
        state: CognitiveState,
        signals: dict[str, float],
        trigger: str,
        threshold: float,
    ) -> MidloopDecision:
        # Confidence = how far past the threshold we are. For LOOPING
        # and STUCK it is the relevant signal value; for DRIFTING it
        # is 1 - relevance (since LOW relevance => HIGH drift).
        if state == CognitiveState.DRIFTING:
            confidence = 1.0 - signals["task_relevance"]
        elif trigger in signals:
            confidence = signals[trigger]
        else:
            confidence = 0.5
        return MidloopDecision(
            intervene=True,
            action=InterventionAction.WHISPER,  # cheapest reversible
            confidence=max(0.0, min(1.0, abs(confidence))),
            reason=(
                f"{state.value}:{trigger}={signals.get(trigger, '?')}"
                f"{f' >= {threshold}' if threshold else ''}"
            ),
            state=state,
            policy=self.name,
            signals=signals,
        )


# ---------------------------------------------------------------- shadow

class ShadowMidloopDecider:
    """Run two midloop deciders side by side; primary is authoritative.

    Mirror of ``ShadowWriteDecider`` for the midloop primitive.
    The primary's ``intervene`` / ``action`` is what the runtime
    acts on; the shadow's verdict is appended to the reason string
    as ``shadow_agree`` / ``shadow_disagree`` so the audit log can
    be grepped for disagreements.

    Use this during graduation: NoopMidloopDecider as primary
    (never intervenes -- safe), HeuristicMidloopDecider as shadow
    (full opinion, no runtime effect). When ~200 disagreements
    accumulate AND a held-out subset shows the heuristic catches
    real failures without firing on healthy tasks, swap primary
    to the heuristic.
    """

    def __init__(
        self,
        primary: MidloopDecider,
        shadow: MidloopDecider,
    ) -> None:
        self._primary = primary
        self._shadow = shadow
        primary_name = getattr(primary, "name", type(primary).__name__)
        shadow_name = getattr(shadow, "name", type(shadow).__name__)
        self.name = f"Shadow({primary_name}|{shadow_name})"

    def decide(
        self,
        observation: StepObservation,
        ctx: MidloopContext,
        trajectory: list[StepObservation],
    ) -> MidloopDecision:
        primary = self._primary.decide(observation, ctx, trajectory)
        try:
            shadow = self._shadow.decide(observation, ctx, trajectory)
        except Exception as exc:
            return MidloopDecision(
                intervene=primary.intervene,
                action=primary.action,
                confidence=primary.confidence,
                reason=f"{primary.reason}|shadow_error:{exc.__class__.__name__}",
                state=primary.state,
                policy=self.name,
                signals=primary.signals,
            )

        agree = (
            shadow.intervene == primary.intervene
            and shadow.action == primary.action
        )
        tag = "shadow_agree" if agree else "shadow_disagree"
        shadow_policy = getattr(shadow, "policy", type(self._shadow).__name__)
        annot = (
            f"{tag}:{shadow_policy}="
            f"{shadow.action.value}:{shadow.state.value}:{shadow.confidence:.3f}"
        )
        return MidloopDecision(
            intervene=primary.intervene,
            action=primary.action,
            confidence=primary.confidence,
            reason=f"{primary.reason}|{annot}",
            state=primary.state,
            policy=self.name,
            signals=primary.signals,
        )


# ---------------------------------------------------------------- persistence filter

def is_persistable_step(
    decision: MidloopDecision,
    prev_decision: MidloopDecision | None,
    is_last_step: bool,
    *,
    confidence_change_threshold: float = 0.2,
) -> bool:
    """Should this step's audit row be written to merken?

    Per Jay (2026-04-19): persisting EVERY step floods the audit log
    with on_track noise. The training signal lives at the moments
    when something changed. Persist only when:

    1. The decision is non-on_track (state != ON_TRACK), OR
    2. The state changed vs the previous decision, OR
    3. The confidence moved by >= threshold vs previous, OR
    4. This is the last step before task outcome.

    All other steps are dropped. The full trajectory still lives in
    the in-process deque while the task is running; only what is
    "interesting" makes it to the persistent audit collection.
    """
    if is_last_step:
        return True
    if decision.state != CognitiveState.ON_TRACK:
        return True
    if prev_decision is None:
        return False  # first step, no signal-trigger context
    if decision.state != prev_decision.state:
        return True
    if abs(decision.confidence - prev_decision.confidence) >= confidence_change_threshold:
        return True
    return False


# ---------------------------------------------------------------- audit format

# format_midloop_audit_row lives in merken.audit alongside the other
# format_*_audit_row helpers. Re-exported here so existing
# `from merken.policies.midloop import format_midloop_audit_row`
# imports keep working.
from merken.audit import format_midloop_audit_row  # noqa: E402, F401
