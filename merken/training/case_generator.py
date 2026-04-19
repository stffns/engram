"""Generate (prompt, truth) clinical-case pairs from Type B protocol clauses.

The large model is the ground-truth oracle: it sees the authoritative
protocol clause and produces N realistic clinical scenarios whose
correct answer follows STRICTLY from the clause. The smaller model
(``response_generator.py``) will later attempt the same prompts
WITHOUT the protocol; the gap between truth and small-model response
is the midloop training signal.

Mock-friendly design: ``CaseGenerator`` takes a ``LLMClient``
callable. The default implementation wraps the anthropic SDK, but
tests pass a fake that returns canned JSON. This keeps the suite
offline + zero-cost while preserving the real-world path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# (system_prompt, user_prompt) -> raw model output text.
LLMClient = Callable[[str, str], str]


@dataclass
class ProtocolClause:
    """One Type B authoritative source clause to expand into cases.

    ``protocol_id`` is the canonical identifier (e.g.
    ``who_pneumonia_amoxicillin_dose``). Used in case_id derivation
    and downstream traceability so a generated case can always be
    linked back to its source clause.
    """
    protocol_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedCase:
    """One synthetic clinical case ready for the response generator."""
    case_id: str
    prompt: str
    truth: str
    metadata: dict[str, Any] = field(default_factory=dict)


_SYSTEM_PROMPT = (
    "You generate realistic clinical scenarios for training a memory-"
    "intervention model. Given an authoritative protocol clause, you "
    "produce N scenarios where a clinician would need exactly this "
    "knowledge.\n\n"
    "Each scenario must have:\n"
    "- prompt: a realistic clinician question or situation, "
    "1-3 sentences. Concrete details (age, weight, signs).\n"
    "- truth: the correct answer DERIVED STRICTLY from the protocol "
    "clause. Do not add information not present in the clause. If the "
    "clause is silent on a detail, do not invent it.\n\n"
    "Respond ONLY with a JSON array of objects with `prompt` and "
    "`truth` keys. No prose, no markdown fences, no commentary."
)

_USER_TEMPLATE = (
    "Protocol id: {protocol_id}\n"
    "Protocol clause:\n```\n{protocol_text}\n```\n\n"
    "Generate {n} distinct clinical scenarios. Vary patient details "
    "(age, weight, presentation) but keep the protocol-derived "
    "answer correct in every case. Output the JSON array now."
)


def _extract_json_array(raw: str) -> list[dict]:
    """Tolerant JSON extraction.

    LLMs sometimes wrap responses in ```json ... ``` fences or add
    a leading sentence. We strip those by scanning for the first
    `[` and the matching final `]`. If parsing fails we re-raise
    with the original text in the message so the caller can debug.
    """
    text = raw.strip()
    # Strip code fences.
    fence_match = re.match(r"```(?:json)?\s*", text)
    if fence_match:
        text = text[fence_match.end():]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    # If the model added a sentence before the array, find the array.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"could not locate JSON array in LLM response: {raw[:200]!r}"
        )
    candidate = text[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON array parse failed: {e}; candidate={candidate[:200]!r}"
        ) from e
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON array, got {type(parsed).__name__}")
    return parsed


class CaseGenerator:
    """Turn one ProtocolClause into N GeneratedCase rows via an LLM.

    Construct once per pipeline run; reuse across clauses. Stateless
    aside from the LLM client reference.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def generate(self, clause: ProtocolClause, n: int = 5) -> list[GeneratedCase]:
        if n <= 0:
            return []
        user = _USER_TEMPLATE.format(
            protocol_id=clause.protocol_id,
            protocol_text=clause.text,
            n=n,
        )
        raw = self._llm(_SYSTEM_PROMPT, user)
        rows = _extract_json_array(raw)

        cases: list[GeneratedCase] = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            prompt = row.get("prompt")
            truth = row.get("truth")
            if not prompt or not truth:
                continue
            case_id = f"{clause.protocol_id}__case_{i:03d}"
            cases.append(GeneratedCase(
                case_id=case_id,
                prompt=str(prompt).strip(),
                truth=str(truth).strip(),
                metadata={"protocol_id": clause.protocol_id, **clause.metadata},
            ))
        return cases


def default_anthropic_client(
    api_key: str,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
) -> LLMClient:
    """Build an LLMClient backed by the anthropic SDK.

    Lazy-imports anthropic so the module remains importable without
    the SDK installed (tests use fake clients). When the SDK is
    needed it is imported once at construction time.

    The default model is Claude Sonnet 4.6 -- per project preference,
    use the latest Sonnet for case generation (capable enough to
    follow the strict-derivation instruction without leaking
    out-of-clause info).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def _fn(system: str, user: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if not resp.content:
            return ""
        block = resp.content[0]
        return getattr(block, "text", "") or ""

    return _fn
