"""Audit log — every decision the loop makes, written through vstash.

CONSTITUTION §4.2: glass box. Every ``should_remember`` (and later
``should_recall``, ``should_consolidate``, ``should_forget``) decision lands
here, with inputs, the policy that fired, and the resulting write.

Storage choice (CONSTITUTION §10 #7): the same vstash backend, but isolated
in its own collection so audit rows can never leak into normal recall.

Audit writes deliberately bypass the engram loop — they call
``vstash.Memory.remember`` directly. Routing audit through the loop would be
recursive and would let a buggy decider silence itself.
"""

from __future__ import annotations

from datetime import datetime, timezone

from engram.policies.types import Decision, Event

AUDIT_COLLECTION = "engram_audit"
AUDIT_LAYER = "audit"

_PREVIEW_CHARS = 160


def format_audit_row(event: Event, decision: Decision) -> tuple[str, str]:
    """Build a (title, body) pair for one audit entry.

    The body is plain text on purpose: it has to be human-readable when an
    operator runs ``mem.audit("why was this dropped")``, and it has to be
    indexable by both vstash's vector and FTS paths.
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    preview = " ".join(event.text.split())[:_PREVIEW_CHARS]

    body = (
        f"timestamp: {ts}\n"
        f"decision: should_remember\n"
        f"write: {decision.write}\n"
        f"reason: {decision.reason}\n"
        f"policy: {decision.policy}\n"
        f"confidence: {decision.confidence}\n"
        f"event_layer: {event.layer}\n"
        f"event_title: {event.title or ''}\n"
        f"event_tags: {event.tags or ''}\n"
        f"event_text_preview: {preview}\n"
    )

    title = f"audit:should_remember:{decision.reason}:{ts}"
    return title, body
