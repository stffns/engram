"""The Memory class — engram's only public surface for now.

Phase 0 wiring (CONSTITUTION §11): a thin wrapper over ``vstash.Memory`` with a
single write path (``remember``) and a single read path (``recall``). No
decision policies, no consolidation, no audit log. Those land in later phases.

The boundary with vstash is sacred (CONSTITUTION §6): we go through its public
API, never through internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import vstash

if TYPE_CHECKING:
    from vstash import IngestResult, SearchResult

DEFAULT_LAYER = "episodic"


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
    """

    def __init__(
        self,
        project: str,
        *,
        db: str | Path | None = None,
        config: str | Path | None = None,
    ) -> None:
        self.project = project
        self._vstash = vstash.Memory(config=config, project=project, db=db)

    def remember(
        self,
        text: str,
        *,
        layer: str = DEFAULT_LAYER,
        title: str | None = None,
        tags: str | None = None,
    ) -> IngestResult:
        """Write an event to memory.

        Phase 0: every call writes. Decision policies (``should_remember``)
        arrive in Phase 1.
        """
        return self._vstash.remember(text, title=title, layer=layer, tags=tags)

    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        layer: str | None = None,
    ) -> list[SearchResult]:
        """Read from memory via vstash hybrid search.

        Phase 0: pure pass-through. Recall policy (``should_recall``) and
        layer-aware routing arrive in Phase 1.
        """
        return self._vstash.search(query, top_k=top_k, layer=layer)

    def close(self) -> None:
        """Release the underlying vstash handle."""
        self._vstash.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
