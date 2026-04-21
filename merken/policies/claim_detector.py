"""Claim detection -- upstream stage of the retrieval-grounded midloop.

The generic midloop (see ``notes/midloop-architecture-v2.md``) fires
retrieval when one of two upstream signals says "this step deserves
fact-checking":

1. ``HeuristicMidloopDecider`` anomaly (LOOPING / DRIFTING / STUCK) --
   already in ``merken/policies/midloop.py``.
2. ``ClaimDetector`` (this module) -- "does the step make a factual
   assertion or express uncertainty?"

ClaimDetector is deliberately **content-agnostic**. It asks about the
LINGUISTIC SHAPE of the step text, not whether the claim is true --
verification happens downstream in ``ClaimVerifier`` against retrieved
memory events.

Three reference implementations ship here:

- ``NoopClaimDetector`` -- never fires. Safe baseline, useful as the
  ``primary`` in a shadow pairing while the candidate detector
  accumulates a labeled disagreement pool.
- ``HeuristicClaimDetector`` -- pure-regex baseline. Detects
  uncertainty markers ("I think"), prescriptive imperatives
  ("administer X"), reasoning connectives ("because"), and
  number-plus-unit factual assertions ("5 mg/kg"). Cheap, no API
  calls, no model load.
- ``LLMClaimDetector`` -- wraps a structured-output LLM client
  (Cerebras / Gemini / Anthropic). The LLM returns the claim list
  directly; no training needed. Primary v0 path per the v2 roadmap.

The scaffolding mirrors ``merken/policies/midloop.py`` (Protocol +
dataclasses + Noop/Heuristic/Shadow deciders) and the LLM client
shape used by ``merken/training/case_generator.py`` (structured
``Callable[[str, str], list[dict]]``). Training a small tagger
(``NanoGPTClaimDetector``) is a follow-up, gated behind a concept
test that proves the midloop end-to-end is worth the latency; see
the v2 note for the exact trigger.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

# ---------------------------------------------------------------- enums

class ClaimType(str, Enum):
    """Categories from the v2 architecture note.

    The enum is open-world friendly: an LLM that returns a string
    outside this set is mapped to ``FACTUAL_ASSERTION`` by default
    rather than rejected, since the downstream verifier does not
    branch on type. Fail-open here, fail-closed at verification.
    """
    FACTUAL_ASSERTION = "factual_assertion"
    UNCERTAINTY_MARKER = "uncertainty_marker"
    PRESCRIPTIVE = "prescriptive"
    REASONING = "reasoning"


# ---------------------------------------------------------------- dataclasses

@dataclass(frozen=True)
class ClaimSpan:
    """One claim located inside a step's text.

    ``start`` and ``end`` are character offsets into the original
    ``step_text`` (half-open, Python slice semantics). ``text`` is
    the substring itself, kept verbatim so downstream code does not
    have to re-slice if the step text is mutated.
    """
    start: int
    end: int
    type: ClaimType
    text: str


@dataclass
class ClaimDetectionInput:
    """Input to a ClaimDetector.

    ``prev_steps`` and ``task_description`` are optional context the
    detector MAY use to reduce false positives (e.g. "I think" in a
    planning step after a prev step full of speculation is less
    likely a verifiable claim than "I think X is Y" in isolation).
    Heuristic ignores them; LLMClaimDetector forwards them verbatim.
    """
    step_text: str
    prev_steps: list[str] = field(default_factory=list)
    task_description: str = ""


@dataclass(frozen=True)
class ClaimDetection:
    """Output of a ClaimDetector.

    ``has_claim`` is a convenience equal to ``bool(claims)``. Kept
    as a separate field so callers can short-circuit without having
    to inspect the list (mirrors the ``intervene`` field on
    ``MidloopDecision``).

    ``signals`` carries per-detector diagnostic values for the audit
    log -- regex match counts for Heuristic, token usage for LLM.
    """
    has_claim: bool
    claims: tuple[ClaimSpan, ...]
    confidence: float
    reason: str
    policy: str
    signals: dict[str, float] = field(default_factory=dict)


class ClaimDetector(Protocol):
    """Protocol for content-agnostic claim detection."""
    name: str

    def detect(self, inp: ClaimDetectionInput) -> ClaimDetection: ...


# ---------------------------------------------------------------- noop

class NoopClaimDetector:
    """Baseline: never reports a claim.

    Serves the same role ``NoopMidloopDecider`` does for the
    midloop: a safe primary during shadow-mode bootstrap. Runtime
    never triggers retrieval from this detector; the shadow's
    verdict is what fills the labeled disagreement pool.
    """

    name = "NoopClaimDetector"

    def detect(self, inp: ClaimDetectionInput) -> ClaimDetection:  # noqa: ARG002
        return ClaimDetection(
            has_claim=False,
            claims=(),
            confidence=1.0,
            reason="noop",
            policy=self.name,
        )


# ---------------------------------------------------------------- heuristic

# Patterns are deliberately small and well-commented. Adding a new
# marker is cheap but every addition is a potential false-positive
# contributor; keep the set curated, not exhaustive.

_UNCERTAINTY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("hedge_i_think", re.compile(
        r"\b(?:i\s+think|i\s+believe|i\s+guess|i\s+suspect)\b",
        re.IGNORECASE,
    )),
    ("hedge_modal", re.compile(
        r"\b(?:might|could|may|possibly|perhaps|probably|maybe)\b",
        re.IGNORECASE,
    )),
    ("hedge_seems", re.compile(
        r"\b(?:seems|appears|looks\s+like)\b",
        re.IGNORECASE,
    )),
)

# Prescriptive imperatives: sentence-initial verb with no explicit
# subject. Clinical bias in the seed list is intentional -- these
# are the shapes the MedLocal v1c-6L artifact detects, useful as a
# sanity anchor. Extend per-domain in a subclass, don't bloat here.
_PRESCRIPTIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("imperative_start", re.compile(
        r"(?:^|\.\s+)(administer|apply|use|avoid|give|prescribe|"
        r"refer|escalate|call|start|stop|check|monitor|continue|"
        r"discontinue|repeat|adjust)\b",
        re.IGNORECASE,
    )),
    ("should_must", re.compile(
        r"\b(?:should|must|need\s+to|has\s+to|have\s+to|ought\s+to)\b",
        re.IGNORECASE,
    )),
)

_REASONING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("causal", re.compile(
        r"\b(?:because|therefore|thus|hence|so\s+that|due\s+to)\b",
        re.IGNORECASE,
    )),
    ("conditional", re.compile(
        r"\b(?:if\s+\w+.+then|since\s+\w+)\b",
        re.IGNORECASE,
    )),
)

# Factual assertion: number + (unit or range) is the highest-signal
# shape. Pure copulas ("X is Y") are too noisy on their own.
_FACTUAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("number_unit", re.compile(
        r"\b\d+(?:\.\d+)?\s*"
        r"(?:mg|g|kg|ml|l|mcg|ug|IU|mmol|mol|mEq|%|"
        r"mg/kg|mg/day|ml/kg|mmHg|bpm|rpm|c|C|f|F|"
        r"hours?|minutes?|days?|weeks?|months?|years?)\b",
        re.IGNORECASE,
    )),
    ("range", re.compile(r"\b\d+(?:\.\d+)?\s*(?:to|-|and)\s*\d+(?:\.\d+)?\b")),
)


def _first_match(
    text: str, patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> tuple[str, re.Match[str]] | None:
    for name, pat in patterns:
        m = pat.search(text)
        if m is not None:
            return name, m
    return None


@dataclass
class HeuristicDetectorThresholds:
    """Knobs for ``HeuristicClaimDetector``.

    Defaults are PLACEHOLDERS until a labeled held-out exists. The
    same discipline as ``HeuristicMidloopDecider``: never promote
    from shadow without held-out calibration.
    """
    min_step_length: int = 8            # drop very short fragments
    uncertainty_confidence: float = 0.7
    prescriptive_confidence: float = 0.8
    reasoning_confidence: float = 0.6
    factual_confidence: float = 0.75


class HeuristicClaimDetector:
    """Regex baseline; no API calls, no model load.

    One span per triggered category (first match wins within a
    category). Scanning is ordered: factual > prescriptive >
    uncertainty > reasoning. This ordering reflects actionability
    at the verifier -- a number+unit mismatch is the highest-signal
    thing to surface; "because" alone rarely is.

    Explicitly NOT exhaustive. The baseline exists so:
      - tests have a non-LLM decider to exercise,
      - the runtime has a backstop when the LLM client is offline,
      - future training runs have a "beat this" bar.

    Per Silt's rule (CLAUDE.md): do not extend this regex set
    speculatively. Add patterns only when a measured false-negative
    in a real corpus justifies them.
    """

    name = "HeuristicClaimDetector"

    def __init__(
        self, thresholds: HeuristicDetectorThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or HeuristicDetectorThresholds()

    def detect(self, inp: ClaimDetectionInput) -> ClaimDetection:
        text = inp.step_text.strip()
        if len(text) < self.thresholds.min_step_length:
            return ClaimDetection(
                has_claim=False,
                claims=(),
                confidence=1.0,
                reason=f"too_short:{len(text)}<{self.thresholds.min_step_length}",
                policy=self.name,
            )

        claims: list[ClaimSpan] = []
        matched_categories: list[str] = []

        factual = _first_match(text, _FACTUAL_PATTERNS)
        if factual is not None:
            name, m = factual
            claims.append(ClaimSpan(
                start=m.start(), end=m.end(),
                type=ClaimType.FACTUAL_ASSERTION,
                text=m.group(0),
            ))
            matched_categories.append(f"factual:{name}")

        prescriptive = _first_match(text, _PRESCRIPTIVE_PATTERNS)
        if prescriptive is not None:
            name, m = prescriptive
            claims.append(ClaimSpan(
                start=m.start(), end=m.end(),
                type=ClaimType.PRESCRIPTIVE,
                text=m.group(0),
            ))
            matched_categories.append(f"prescriptive:{name}")

        uncertainty = _first_match(text, _UNCERTAINTY_PATTERNS)
        if uncertainty is not None:
            name, m = uncertainty
            claims.append(ClaimSpan(
                start=m.start(), end=m.end(),
                type=ClaimType.UNCERTAINTY_MARKER,
                text=m.group(0),
            ))
            matched_categories.append(f"uncertainty:{name}")

        reasoning = _first_match(text, _REASONING_PATTERNS)
        if reasoning is not None:
            name, m = reasoning
            claims.append(ClaimSpan(
                start=m.start(), end=m.end(),
                type=ClaimType.REASONING,
                text=m.group(0),
            ))
            matched_categories.append(f"reasoning:{name}")

        if not claims:
            return ClaimDetection(
                has_claim=False,
                claims=(),
                confidence=0.9,
                reason="no_pattern_match",
                policy=self.name,
                signals={"n_categories": 0.0},
            )

        # Confidence = max of per-category priors that fired. Crude
        # but honest -- a proper calibration needs held-out labels.
        type_to_conf = {
            ClaimType.FACTUAL_ASSERTION: self.thresholds.factual_confidence,
            ClaimType.PRESCRIPTIVE: self.thresholds.prescriptive_confidence,
            ClaimType.UNCERTAINTY_MARKER: self.thresholds.uncertainty_confidence,
            ClaimType.REASONING: self.thresholds.reasoning_confidence,
        }
        confidence = max(type_to_conf[c.type] for c in claims)

        return ClaimDetection(
            has_claim=True,
            claims=tuple(claims),
            confidence=confidence,
            reason=",".join(matched_categories),
            policy=self.name,
            signals={"n_categories": float(len(claims))},
        )


# ---------------------------------------------------------------- llm

# Structured client contract: a callable that takes (system_prompt,
# user_prompt) and returns a parsed list of dicts. Matches the shape
# used by ``case_generator.LLMStructuredClient`` so a single Cerebras
# / Gemini / Anthropic client can be shared across generators and
# detectors without an adapter layer.
LLMStructuredClient = Callable[[str, str], list[dict]]

_LLM_SYSTEM_PROMPT = (
    "You are a claim classifier for a retrieval-grounded memory "
    "system. Given a single step of model output, identify any "
    "spans that make a factual assertion, express uncertainty, "
    "give a prescriptive instruction, or state a reasoning link.\n\n"
    "Claim types:\n"
    "- factual_assertion: 'X is Y', numeric values with units, "
    "statements of fact that can be verified against a source.\n"
    "- uncertainty_marker: hedging words like 'I think', 'possibly', "
    "'might', 'could be'.\n"
    "- prescriptive: imperative instructions like 'administer X', "
    "'avoid Y', 'escalate to Z'.\n"
    "- reasoning: causal links like 'because X, therefore Y'.\n\n"
    "Return a JSON list. Each element MUST have keys: start (int, "
    "character offset), end (int, exclusive), type (one of the four "
    "strings above), text (the exact substring). Return [] if the "
    "step contains no claims. Do not invent spans; only report what "
    "is in the step_text."
)

_LLM_USER_TEMPLATE = (
    "task_description: {task_description}\n"
    "prev_steps (most recent last):\n{prev_steps}\n\n"
    "step_text:\n```\n{step_text}\n```\n\n"
    "Return the JSON list now."
)


def _coerce_claim_type(raw: str) -> ClaimType:
    """Map an LLM-returned string to a ClaimType, fail-open.

    An LLM that returns an unexpected category (typo, neologism)
    should not crash the pipeline. Default to FACTUAL_ASSERTION --
    the downstream verifier retrieves + compares regardless of
    type, so over-reporting assertions is the safe direction.
    """
    try:
        return ClaimType(raw.strip().lower())
    except ValueError:
        return ClaimType.FACTUAL_ASSERTION


class LLMClaimDetector:
    """Wrap a structured LLM client as a ClaimDetector.

    Works with any client matching ``LLMStructuredClient``:
    Cerebras (concept-test choice -- ~50ms per call at 2200 tok/s),
    Gemini 2.5 Flash (batch labeling for dataset generation),
    Anthropic Haiku (fallback). The client is injected, not
    constructed here, so tests can pass a fake that returns canned
    spans without hitting the network.

    The detector is stateless aside from the client reference. Safe
    to call concurrently if the underlying client is thread-safe.
    """

    name = "LLMClaimDetector"

    def __init__(
        self,
        llm_client: LLMStructuredClient,
        *,
        system_prompt: str | None = None,
        min_step_length: int = 8,
    ) -> None:
        self._llm = llm_client
        self._system_prompt = system_prompt or _LLM_SYSTEM_PROMPT
        self._min_step_length = min_step_length

    def detect(self, inp: ClaimDetectionInput) -> ClaimDetection:
        text = inp.step_text
        if len(text.strip()) < self._min_step_length:
            return ClaimDetection(
                has_claim=False,
                claims=(),
                confidence=1.0,
                reason=f"too_short:{len(text.strip())}<{self._min_step_length}",
                policy=self.name,
            )

        prev_rendered = (
            "\n".join(f"- {s}" for s in inp.prev_steps)
            if inp.prev_steps else "(none)"
        )
        user = _LLM_USER_TEMPLATE.format(
            task_description=inp.task_description or "(none)",
            prev_steps=prev_rendered,
            step_text=text,
        )

        try:
            rows = self._llm(self._system_prompt, user)
        except Exception as exc:
            return ClaimDetection(
                has_claim=False,
                claims=(),
                confidence=0.0,
                reason=f"llm_error:{exc.__class__.__name__}",
                policy=self.name,
            )

        if not isinstance(rows, list):
            return ClaimDetection(
                has_claim=False,
                claims=(),
                confidence=0.0,
                reason=f"llm_bad_shape:{type(rows).__name__}",
                policy=self.name,
            )

        claims: list[ClaimSpan] = []
        dropped = 0
        for row in rows:
            if not isinstance(row, dict):
                dropped += 1
                continue
            try:
                start = int(row["start"])
                end = int(row["end"])
                span_text = str(row.get("text", ""))
                claim_type = _coerce_claim_type(str(row.get("type", "")))
            except (KeyError, TypeError, ValueError):
                dropped += 1
                continue
            # Guard against out-of-range offsets. We trust the LLM
            # but not blindly: span bounds outside the step text
            # would corrupt any downstream verifier that slices on
            # them.
            if start < 0 or end > len(text) or end <= start:
                dropped += 1
                continue
            # Realign span_text to the source text if the LLM
            # returned a paraphrase. Fail-closed: drop rather than
            # keep a non-verbatim span.
            if text[start:end] != span_text and span_text:
                dropped += 1
                continue
            claims.append(ClaimSpan(
                start=start, end=end,
                type=claim_type,
                text=span_text if span_text else text[start:end],
            ))

        signals = {
            "n_llm_rows": float(len(rows)),
            "n_dropped": float(dropped),
            "n_kept": float(len(claims)),
        }
        if not claims:
            return ClaimDetection(
                has_claim=False,
                claims=(),
                confidence=0.8 if not rows else 0.5,
                reason="llm_empty" if not rows else f"llm_all_dropped:{dropped}",
                policy=self.name,
                signals=signals,
            )

        # LLM doesn't emit a calibrated confidence; use a flat 0.85
        # until a held-out calibration pass produces a real curve.
        return ClaimDetection(
            has_claim=True,
            claims=tuple(claims),
            confidence=0.85,
            reason=f"llm_detected:{len(claims)}",
            policy=self.name,
            signals=signals,
        )


# ---------------------------------------------------------------- shadow

class ShadowClaimDetector:
    """Run two ClaimDetectors side by side; primary is authoritative.

    Mirror of ``ShadowMidloopDecider`` for the claim-detection
    primitive. Primary drives runtime behavior; shadow's verdict is
    annotated onto the reason string as ``shadow_agree`` /
    ``shadow_disagree`` so the audit log can be grepped for the
    labeled-disagreement pool.

    Typical bootstrap: ``ShadowClaimDetector(NoopClaimDetector(),
    LLMClaimDetector(client))``. Noop is the safe primary (no
    retrieval fires). The LLM's disagreements accumulate as the
    labeling pool for a future trained detector.
    """

    def __init__(
        self,
        primary: ClaimDetector,
        shadow: ClaimDetector,
    ) -> None:
        self._primary = primary
        self._shadow = shadow
        primary_name = getattr(primary, "name", type(primary).__name__)
        shadow_name = getattr(shadow, "name", type(shadow).__name__)
        self.name = f"Shadow({primary_name}|{shadow_name})"

    def detect(self, inp: ClaimDetectionInput) -> ClaimDetection:
        primary = self._primary.detect(inp)
        try:
            shadow = self._shadow.detect(inp)
        except Exception as exc:
            return ClaimDetection(
                has_claim=primary.has_claim,
                claims=primary.claims,
                confidence=primary.confidence,
                reason=f"{primary.reason}|shadow_error:{exc.__class__.__name__}",
                policy=self.name,
                signals=primary.signals,
            )

        agree = shadow.has_claim == primary.has_claim
        tag = "shadow_agree" if agree else "shadow_disagree"
        shadow_policy = getattr(shadow, "policy", type(self._shadow).__name__)
        annot = (
            f"{tag}:{shadow_policy}=has_claim:{shadow.has_claim}"
            f":n_claims:{len(shadow.claims)}:conf:{shadow.confidence:.3f}"
        )
        return ClaimDetection(
            has_claim=primary.has_claim,
            claims=primary.claims,
            confidence=primary.confidence,
            reason=f"{primary.reason}|{annot}",
            policy=self.name,
            signals=primary.signals,
        )
