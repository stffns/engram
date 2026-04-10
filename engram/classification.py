"""Heuristic content-type classification — no LLM, no dependencies.

Auto-classifies text into content types for the ``ContentTypePriorDecider``
when the caller hasn't provided an explicit ``type:<kind>`` tag. The
classifier is a fallback, not a replacement: explicit tags always win.

Tier 1 (>85% precision on synthetic + real scenarios):
    - ``tool_echo`` — CLI tool output (npm, git, pytest, docker, etc.)
    - ``ack`` — acknowledgment phrases ("got it", "understood", "sure")

Tier 2 (70-85% precision):
    - ``status`` — build/deploy/monitoring status reports
    - ``ambient_chat`` — thinking-out-loud, self-talk

Unclassified text returns ``"unknown"`` and gets the default prior (1.0),
which means it passes through to the heuristic rules unchanged. The
classifier is intentionally conservative: a false negative (unknown)
just means the event gets treated normally; a false positive (wrong type)
could drop a valuable event.

See ``notes/research-2026-04-09.md`` for the A-MAC paper analysis that
motivated content-type priors.
"""

from __future__ import annotations

import re

# ------------------------------------------------------------------ patterns

# CLI tool names that appear in tool output
_TOOL_NAMES = re.compile(
    r"\b(npm|yarn|pnpm|pip|pip3|python|python3|node|cargo|go"
    r"|pytest|jest|vitest|mocha"
    r"|docker|docker-compose|kubectl|terraform"
    r"|git|gh|rg|find|grep|curl|wget|make|bash|sh|zsh)\b",
    re.IGNORECASE,
)

# Output markers that accompany tool names
_TOOL_OUTPUT = re.compile(
    r"\b(passed|failed|skipped|error|warning|modified|deleted|created"
    r"|installed|vulnerabilities|coverage|lint|compiled|built"
    r"|healthy|unhealthy|running|stopped|exited)\b",
    re.IGNORECASE,
)

# Standalone tool output patterns (no tool name needed)
_TOOL_STANDALONE = re.compile(
    r"(^\s*(Running |Fetching |Building |Compiling |Installing )"
    r"|Done in \d+\.\d+ seconds"
    r"|\d+ packages? installed"
    r"|\d+ passed,? \d+ failed"
    r"|Changes not staged for commit"
    r"|On branch \S+)",
    re.MULTILINE,
)

# Ack starters — case-insensitive, at the beginning of text
_ACK_STARTERS = (
    "got it",
    "understood",
    "sure,",
    "sure ",
    "ok,",
    "ok ",
    "okay,",
    "okay ",
    "noted",
    "confirmed",
    "will do",
    "sounds good",
    "makes sense",
    "right,",
    "right ",
    "alright",
)

# Ack follow-up phrases (present anywhere, but text must be short)
_ACK_PHRASES = (
    "i'll ",
    "i've noted",
    "i've updated",
    "updating the",
    "let me set that up",
)

_ACK_MAX_CHARS = 150

# Status report keywords
_STATUS_KEYWORDS = (
    "health check",
    "monitoring alert",
    "deployment completed",
    "deployment to",
    "deploy completed",
    "smoke test",
    "tests green",
    "all checks passed",
    "build passed",
    "build failed",
    "merged into",
    "branch deleted",
    "alert resolved",
    "pod",
)

# Status metric patterns (numbers + units that signal automated output)
_STATUS_METRICS = re.compile(
    r"(\b\d+(\.\d+)?%"  # percentages
    r"|\bBuild #\d+"  # build numbers
    r"|\b\d{2}:\d{2}\s*(UTC|CEST|EST|PST)"  # timestamps with TZ
    r"|\b\d+ pods?\b"  # pod counts
    r"|\bPR #\d+)",  # PR numbers
    re.IGNORECASE,
)

# Ambient chat starters
_AMBIENT_STARTERS = (
    "let me think",
    "let me re-read",
    "let me look",
    "hmm,",
    "hmm ",
    "hmmm",
    "interesting —",
    "interesting,",
    "hold on,",
    "hold on ",
    "wait,",
    "wait —",
)

# Ambient chat phrases (present anywhere)
_AMBIENT_PHRASES = (
    "i hadn't considered",
    "i hadn't thought",
    "yeah, i think",
    "yeah, that",
    "actually,",
    "let me think about",
    "let me re-read",
    "not sure if",
    "i want to make sure",
)


# ------------------------------------------------------------------ public API


def classify_content_type(text: str) -> str:
    """Classify text into a content type by heuristic rules.

    Returns one of: ``tool_echo``, ``ack``, ``status``,
    ``ambient_chat``, or ``unknown``.

    The classifier is conservative — ``unknown`` is the safe default.
    """
    lower = text.lower().strip()

    # --- Tier 1: high confidence ---

    # Tool echo: tool name + output marker, or standalone pattern
    if _TOOL_STANDALONE.search(text):
        return "tool_echo"
    if _TOOL_NAMES.search(text) and _TOOL_OUTPUT.search(text):
        return "tool_echo"

    # Ack: formulaic opening phrase + short
    if any(lower.startswith(s) for s in _ACK_STARTERS):
        if len(text) < _ACK_MAX_CHARS:
            return "ack"

    if len(text) < _ACK_MAX_CHARS:
        if any(p in lower for p in _ACK_PHRASES):
            return "ack"

    # --- Tier 2: good confidence ---

    # Status: keyword match + optional metric pattern
    if any(kw in lower for kw in _STATUS_KEYWORDS):
        return "status"
    if _STATUS_METRICS.search(text) and len(text) < 200:
        # Short text with metrics but no decision language → likely status
        if any(w in lower for w in ("passed", "failed", "completed", "resolved", "merged")):
            return "status"

    # Ambient chat: thinking phrases
    if any(lower.startswith(s) for s in _AMBIENT_STARTERS):
        return "ambient_chat"
    if any(p in lower for p in _AMBIENT_PHRASES):
        return "ambient_chat"

    return "unknown"
