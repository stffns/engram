"""The Memory class — engram's only public surface for now.

Phase 1 wiring (CONSTITUTION §11 + ultra-plan):

- ``remember`` runs the configured ``should_remember`` policy first, writes an
  audit row for every decision (write *or* skip), and only then calls
  ``vstash.Memory.remember`` if the decision says so.
- ``recall`` is still a thin pass-through over ``vstash.Memory.search``. It
  is naturally isolated from the audit log because audit lives in its own
  vstash collection (``engram_audit``).
- ``audit`` lets you query the decision log directly.

The boundary with vstash is sacred (CONSTITUTION §6): every storage call
goes through vstash's public API. Engram never reads the SQLite tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import vstash

from engram.audit import AUDIT_COLLECTION, AUDIT_LAYER, format_audit_row
from engram.policies.should_remember import HeuristicWriteDecider
from engram.policies.types import Decision, Event, WriteContext, WriteDecider

if TYPE_CHECKING:
    from vstash import IngestResult, SearchResult

DEFAULT_LAYER = "episodic"
DEFAULT_COLLECTION = "default"


@dataclass(frozen=True)
class RememberResult:
    """The outcome of a ``Memory.remember`` call.

    ``written`` reflects whether the event actually landed in vstash.
    ``decision`` is the policy verdict (always populated, even on skips).
    ``ingest`` is the underlying ``vstash.IngestResult`` when written, else
    ``None``.
    """

    written: bool
    decision: Decision
    ingest: IngestResult | None


class Memory:
    """Agent-loop memory, backed by a single ``vstash.Memory`` instance.

    Parameters
    ----------
    project:
        Logical project name. Becomes the vstash ``project`` tag on every
        write and the default filter on every read.
    db:
        Optional path to the vstash SQLite file. When omitted, vstash uses
        its default location.
    config:
        Optional path to a vstash config file/profile.
    write_decider:
        Optional custom ``should_remember`` policy. Defaults to
        ``HeuristicWriteDecider()``. Pass ``AlwaysWrite()`` for the
        store-everything baseline used in benchmarks.
    """

    def __init__(
        self,
        project: str,
        *,
        db: str | Path | None = None,
        config: str | Path | None = None,
        collection: str = DEFAULT_COLLECTION,
        write_decider: WriteDecider | None = None,
    ) -> None:
        self.project = project
        self.collection = collection
        self._vstash = vstash.Memory(
            config=config,
            project=project,
            db=db,
            collection=collection,
        )
        self._write_decider: WriteDecider = write_decider or HeuristicWriteDecider()

    # ------------------------------------------------------------------ write

    def remember(
        self,
        text: str,
        *,
        layer: str = DEFAULT_LAYER,
        title: str | None = None,
        tags: str | None = None,
    ) -> RememberResult:
        """Submit an event to memory.

        The configured ``should_remember`` policy decides whether the event
        is written. Both outcomes (write and skip) produce an audit row.
        """
        event = Event(text=text, layer=layer, title=title, tags=tags)
        ctx = WriteContext(project=self.project, recall=self._policy_recall)

        decision = self._write_decider.decide(event, ctx)
        self._write_audit(event, decision)

        if not decision.write:
            return RememberResult(written=False, decision=decision, ingest=None)

        ingest = self._vstash.remember(
            text,
            title=title,
            collection=self.collection,
            layer=layer,
            tags=tags,
        )
        return RememberResult(written=True, decision=decision, ingest=ingest)

    # ------------------------------------------------------------------- read

    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        layer: str | None = None,
    ) -> list[SearchResult]:
        """Read from memory via vstash hybrid search.

        Audit rows live in their own collection and are never returned here
        — only the ``audit`` method exposes them. We pass ``collection``
        explicitly because ``vstash.Memory.search`` does not auto-scope to
        the instance's collection when the argument is omitted.
        """
        return self._vstash.search(
            query,
            top_k=top_k,
            collection=self.collection,
            layer=layer,
        )

    def audit(
        self,
        query: str = "should_remember",
        *,
        top_k: int = 20,
    ) -> list[SearchResult]:
        """Query the audit log.

        Use this to answer "why was X kept / dropped?" The audit collection
        is searched in isolation from normal memory.
        """
        return self._vstash.search(
            query,
            top_k=top_k,
            collection=AUDIT_COLLECTION,
            layer=AUDIT_LAYER,
        )

    # --------------------------------------------------------------- internals

    def _policy_recall(
        self,
        query: str,
        top_k: int,
        layer: str | None,
    ) -> list[Any]:
        """Recall callable handed to write deciders.

        Kept here so policies never touch the vstash instance directly.
        Scoped to the engram collection so deduplication never sees audit
        rows.
        """
        return self._vstash.search(
            query,
            top_k=top_k,
            collection=self.collection,
            layer=layer,
        )

    def _write_audit(self, event: Event, decision: Decision) -> None:
        """Write one audit row, bypassing the loop.

        Audit failures must never break the user's call. We swallow the
        exception so a flaky audit write can't take down a good ``remember``.
        The cost is that audit gaps are silent — acceptable for Phase 1, to
        be revisited if it ever bites in practice.
        """
        title, body = format_audit_row(event, decision)
        try:
            self._vstash.remember(
                body,
                title=title,
                collection=AUDIT_COLLECTION,
                layer=AUDIT_LAYER,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------- misc

    def close(self) -> None:
        """Release the underlying vstash handle."""
        self._vstash.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
