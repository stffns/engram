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

from engram.audit import (
    AUDIT_COLLECTION,
    AUDIT_LAYER,
    TOMBSTONE_COLLECTION,
    TOMBSTONE_LAYER,
    format_audit_row,
    format_consolidate_audit_row,
    format_forget_audit_row,
    format_recall_audit_row,
    format_tombstone_row,
)
from engram.consolidation import (
    ConsolidationResult,
    cluster_by_embedding,
    cluster_by_jaccard,
    cluster_by_recall,
    fact_fingerprint,
    materialize_fact,
)
from engram.policies.should_consolidate import (
    ConsolidateContext,
    ConsolidateDecider,
    ConsolidationDecision,
    PeriodicConsolidator,
)
from engram.policies.should_forget import (
    ForgetContext,
    ForgetDecider,
    ForgetDecision,
    NeverForget,
)
from engram.policies.should_recall import (
    LayeredRecaller,
    RecallContext,
    RecallDecider,
    RecallPlan,
)
from engram.policies.should_remember import HeuristicWriteDecider
from engram.policies.types import Decision, Event, WriteContext, WriteDecider

if TYPE_CHECKING:
    from vstash import IngestResult, SearchResult

DEFAULT_LAYER = "episodic"
DEFAULT_COLLECTION = "default"


def _resolve_vstash_embed_model(vstash_memory: vstash.Memory) -> str:
    """Return the embedding model this vstash Memory is actually using.

    Priority:

    1. ``store_meta.embedding_model`` from the vstash SQLite DB. This is
       authoritative for an existing store because it records the model
       vstash used to *ingest* the chunks that are now sitting in the
       vector index. Reading any other model at clustering time creates
       a silent vector-space mismatch between "how engram groups" and
       "how vstash retrieves."
    2. ``vstash.config.EmbeddingsConfig().model`` — vstash's current
       factory default. Used when the store is fresh (no ingests yet,
       so no ``store_meta`` row) and when reading the DB fails for any
       reason.

    The earlier implementation hardcoded ``BAAI/bge-small-en-v1.5`` as
    ``DEFAULT_EMBED_MODEL``, which was wrong for any user who had
    configured vstash with a different model. On Jay's live vstash
    (which uses ``paraphrase-multilingual-MiniLM-L12-v2``), engram
    consolidation was re-embedding chunks with bge-small and then
    writing facts that vstash indexed back with multilingual. Clusters
    were internally coherent but misaligned with the actual retrieval
    vector space. Caught 2026-04-09 while verifying the store wasn't
    mixed (it wasn't — but engram was pretending it was bge-small).
    """
    import sqlite3

    from vstash.config import EmbeddingsConfig

    fallback = EmbeddingsConfig().model

    db_path = getattr(getattr(vstash_memory, "_store", None), "db_path", None)
    if db_path is None:
        return fallback

    try:
        con = sqlite3.connect(str(db_path))
        try:
            row = con.execute(
                "SELECT value FROM store_meta WHERE key = ?",
                ("embedding_model",),
            ).fetchone()
        finally:
            con.close()
    except Exception:
        return fallback

    if row and row[0]:
        return row[0]
    return fallback


@dataclass(frozen=True)
class ForgetResult:
    """The outcome of a ``Memory.forget`` call.

    ``tombstoned`` is the list of event paths that were tombstoned
    in this call. ``skipped`` pairs each surviving event path with
    the reason the decider gave for keeping it.
    """

    tombstoned: list[str]
    skipped: list[tuple[str, str]]
    events_examined: int
    decider: str


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
        consolidate_decider: ConsolidateDecider | None = None,
        recall_decider: RecallDecider | None = None,
        forget_decider: ForgetDecider | None = None,
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
        self._consolidate_decider: ConsolidateDecider = (
            consolidate_decider or PeriodicConsolidator()
        )
        self._recall_decider: RecallDecider = recall_decider or LayeredRecaller()
        self._forget_decider: ForgetDecider = forget_decider or NeverForget()

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
        """Read from memory.

        When ``layer`` is not specified, routes through the configured
        ``should_recall`` policy: the decider returns a ``RecallPlan``
        with a priority-ordered list of per-layer requests, we query
        each in turn, dedupe by vstash path, and truncate to the
        caller's ``top_k``.

        When ``layer`` IS specified, the decider is bypassed and the
        call is a pass-through to ``vstash.Memory.search`` with that
        layer. This is the escape hatch for benchmarks and for callers
        that already know exactly which layer they want.

        Either way, recall is scoped to the engram collection and
        never returns rows from the ``engram_audit`` collection.
        """
        if layer is not None:
            return self._vstash.search(
                query,
                top_k=top_k,
                collection=self.collection,
                layer=layer,
            )

        ctx = RecallContext(project=self.project, top_k=top_k)
        plan = self._recall_decider.decide(query, ctx)
        self._write_recall_audit(query, plan)

        # Fetch from every layer first, then interleave round-robin.
        # The earlier implementation drained each layer sequentially,
        # which meant that when the first layer returned ``top_k``
        # hits, later layers were never consulted. On a smoke test
        # against real vstash content (2026-04-09), that made the
        # "episodic fallback" a fallback only in name: a Kafka
        # meeting note (singleton in episodic) never surfaced for
        # the "Kafka merchant pipeline" query because four semantic
        # facts filled the top_k=3 budget first.
        #
        # Round-robin interleave guarantees every requested layer
        # gets at least one slot in the final list (until the
        # user's top_k is reached), which is what "layered recall"
        # was supposed to mean all along.
        per_layer_hits: list[list[SearchResult]] = []
        for req in plan.layers:
            layer_hits = self._vstash.search(
                query,
                top_k=req.top_k,
                collection=self.collection,
                layer=req.layer,
            )
            per_layer_hits.append(list(layer_hits))

        seen_paths: set[str] = set()
        merged: list[SearchResult] = []
        max_len = max((len(hs) for hs in per_layer_hits), default=0)
        for i in range(max_len):
            for layer_hits in per_layer_hits:
                if i >= len(layer_hits):
                    continue
                h = layer_hits[i]
                path = getattr(h, "path", None)
                if path is not None and path in seen_paths:
                    continue
                if path is not None:
                    seen_paths.add(path)
                merged.append(h)
                if len(merged) >= top_k:
                    return merged

        return merged[:top_k]

    # ----------------------------------------------------------- consolidate

    def consolidate(
        self,
        *,
        method: str = "embedding_v1",
        min_cluster: int = 2,
        embedding_threshold: float = 0.70,
        embedding_linkage: str = "complete",
        recall_top_k: int = 5,
        jaccard_threshold: float = 0.5,
        force: bool = False,
    ) -> ConsolidationResult:
        """Cluster episodic events into semantic facts. Phase 2, no LLM.

        Pulls every ``layer="episodic"`` document in this engram
        collection, reassembles the text from its chunks, clusters
        them, and writes one ``layer="semantic"`` fact per cluster of
        size ≥ ``min_cluster``. Singletons are skipped (they're
        already findable as episodic).

        Every run, write or skip, logs a ``should_consolidate`` audit
        row. Fact titles are a stable hash of ``derived_from`` so
        re-running is idempotent — no duplicate semantic rows.

        Parameters
        ----------
        method:
            Clustering strategy. ``"embedding_v1"`` (default) embeds
            each event via the vstash embedder and clusters by raw
            cosine similarity above ``embedding_threshold``. This is
            the only method that reliably handles paraphrased natural
            language. ``"recall_v1"`` delegates to vstash's hybrid
            search but is brittle on small corpora — see
            ``cluster_by_recall`` for the limitation. ``"jaccard_v1"``
            is near-duplicate only; see ``cluster_by_jaccard``.
        min_cluster:
            Minimum cluster size for a fact to be written. ``2`` means
            "require corroboration across at least two episodic
            events." Singletons always stay in the episodic layer.
        embedding_threshold:
            Cosine similarity cutoff when ``method="embedding_v1"``.
            ``0.70`` was picked by a grid search across the three
            loop_quality scenarios (2026-04-09). It achieves 100%
            query pass rate and 100% cluster purity on all three.
            See ``experiments/loop_quality/RESULTS.md`` for the
            full grid and trade-offs.
        embedding_linkage:
            ``"complete"`` (default) or ``"single"``. Complete-link
            merges only when every cross-cluster pair is above
            ``embedding_threshold``; single-link merges on any one
            above-threshold edge. Complete avoids the cascade where
            one weak-but-genuine edge contaminates a transitive
            cluster. See ``cluster_by_embedding`` for the trade-off.
        recall_top_k:
            Neighbors per event when ``method="recall_v1"``.
        jaccard_threshold:
            Token-set Jaccard similarity when ``method="jaccard_v1"``.
        force:
            Bypass the ``should_consolidate`` decider and always run.
            Useful in tests and when the caller has already decided.
        """
        docs = self._vstash.list(
            collection=self.collection,
            layer="episodic",
        )

        events: list[tuple[str, str]] = []
        for doc in docs:
            chunks = self._vstash.get_document_chunks(
                doc.path,
                collection=self.collection,
            )
            full_text = " ".join(chunks).strip()
            if full_text:
                events.append((doc.path, full_text))

        ctx = ConsolidateContext(project=self.project)
        decision = self._consolidate_decider.decide(len(events), ctx)
        self._write_consolidate_audit(decision, len(events))

        if not decision.proceed and not force:
            return ConsolidationResult(
                events_examined=len(events),
                facts_written=0,
                facts=[],
                skipped=True,
                reason=decision.reason,
                decider=decision.policy,
                method=method,
            )

        if method == "embedding_v1":
            model_name = _resolve_vstash_embed_model(self._vstash)

            def _embed(texts: list[str]) -> list[Any]:
                from vstash.embed import embed_texts
                return embed_texts(
                    texts,
                    model_name=model_name,
                    backend="auto",
                )
            clusters = cluster_by_embedding(
                events,
                embed_fn=_embed,
                threshold=embedding_threshold,
                linkage=embedding_linkage,
            )
            reason = (
                f"clustered_embedding>={embedding_threshold}"
                f"_linkage={embedding_linkage}"
                f"_mincluster={min_cluster}"
            )
        elif method == "jaccard_v1":
            clusters = cluster_by_jaccard(events, threshold=jaccard_threshold)
            reason = f"clustered_jaccard>={jaccard_threshold}_mincluster={min_cluster}"
        elif method == "recall_v1":
            def _cluster_recall(query: str, top_k: int) -> list[Any]:
                return self._vstash.search(
                    query,
                    top_k=top_k,
                    collection=self.collection,
                    layer="episodic",
                )
            clusters = cluster_by_recall(
                events,
                recall_fn=_cluster_recall,
                top_k=recall_top_k,
            )
            reason = f"clustered_recall_topk={recall_top_k}_mincluster={min_cluster}"
        else:
            raise ValueError(
                f"unknown consolidation method {method!r}; "
                f"expected 'embedding_v1', 'recall_v1', or 'jaccard_v1'"
            )

        facts_written = []
        for cluster in clusters:
            if len(cluster) < min_cluster:
                continue
            fact = materialize_fact(cluster)
            fp = fact_fingerprint(fact)
            self._vstash.remember(
                fact.text,
                title=f"fact_{fp}",
                collection=self.collection,
                layer="semantic",
                tags=f"derived_from:{','.join(fact.derived_from)}",
            )
            facts_written.append(fact)

        return ConsolidationResult(
            events_examined=len(events),
            facts_written=len(facts_written),
            facts=facts_written,
            skipped=False,
            reason=reason,
            decider=decision.policy,
            method=method,
        )

    # --------------------------------------------------------------- forget

    def forget(self, *, force: bool = False) -> ForgetResult:
        """Tombstone episodic events whose content is preserved in facts.

        Walks the semantic layer to build a reverse map
        ``event_path → [fact_paths that cite it in derived_from]``,
        then asks the configured ``should_forget`` decider about
        each episodic event. When the decider says yes (or ``force``
        is set), the event is:

        1. Copied to ``engram_tombstones`` with full text + metadata
           + provenance to the facts that preserve it. This is the
           authoritative forgetting record — reversible via
           ``unforget`` (not implemented yet; future slice).
        2. Removed from the engram collection via ``vstash.remove``
           so it no longer surfaces in recall.

        Every decision (tombstone or skip) writes a ``should_forget``
        audit row. Forget operations are intentionally slow and
        loud on purpose — losing a memory is a big deal, even with
        the tombstone safety net.

        Parameters
        ----------
        force:
            Tombstone *every* episodic event regardless of the
            decider. Useful for ``mem.forget(force=True)`` as a
            "wipe the episodic layer" operation after a known-good
            consolidation pass. Still writes tombstones and audit
            rows for each — nothing is destroyed, only moved.
        """
        facts = self._vstash.list(
            collection=self.collection,
            layer="semantic",
        )
        derived_in: dict[str, list[str]] = {}
        for fact in facts:
            tags = fact.tags or ""
            if tags.startswith("derived_from:"):
                paths = tags[len("derived_from:"):].split(",")
                for p in paths:
                    p = p.strip()
                    if p:
                        derived_in.setdefault(p, []).append(fact.path)

        episodic = self._vstash.list(
            collection=self.collection,
            layer="episodic",
        )

        tombstoned: list[str] = []
        skipped: list[tuple[str, str]] = []

        for event in episodic:
            chunks = self._vstash.get_document_chunks(
                event.path,
                collection=self.collection,
            )
            full_text = " ".join(chunks).strip()

            event_derived = derived_in.get(event.path, [])
            ctx = ForgetContext(
                project=self.project,
                derived_in_facts=event_derived,
            )
            decision = self._forget_decider.decide(
                event.path,
                full_text,
                ctx,
            )

            self._write_forget_audit(event.path, decision, event_derived)

            if not (decision.tombstone or force):
                skipped.append((event.path, decision.reason))
                continue

            self._write_tombstone(
                event_path=event.path,
                event_text=full_text,
                event_title=event.title,
                event_layer=event.layer,
                event_tags=event.tags,
                derived_in_facts=event_derived,
                reason=decision.reason if decision.tombstone else "forced",
                policy=decision.policy if decision.tombstone else "force",
            )
            try:
                self._vstash.remove(event.path)
                tombstoned.append(event.path)
            except Exception:
                # If remove fails, the tombstone still exists — the
                # event is in both places until the next forget run.
                # Better than silently losing the tombstone.
                skipped.append((event.path, "vstash_remove_failed"))

        return ForgetResult(
            tombstoned=tombstoned,
            skipped=skipped,
            events_examined=len(episodic),
            decider=self._forget_decider.name,
        )

    def tombstones(
        self,
        query: str = "tombstone",
        *,
        top_k: int = 20,
    ) -> list[SearchResult]:
        """Query the tombstone collection.

        Use this to find "what did I forget?" The tombstone
        collection is searched in isolation from normal memory and
        from the audit log.
        """
        return self._vstash.search(
            query,
            top_k=top_k,
            collection=TOMBSTONE_COLLECTION,
            layer=TOMBSTONE_LAYER,
        )

    # --------------------------------------------------------------- audit

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

    def _write_consolidate_audit(
        self,
        decision: ConsolidationDecision,
        n_events: int,
    ) -> None:
        """Audit one ``should_consolidate`` decision. Same fail-open policy."""
        title, body = format_consolidate_audit_row(decision, n_events)
        try:
            self._vstash.remember(
                body,
                title=title,
                collection=AUDIT_COLLECTION,
                layer=AUDIT_LAYER,
            )
        except Exception:
            pass

    def _write_recall_audit(self, query: str, plan: RecallPlan) -> None:
        """Audit one ``should_recall`` decision. Fail-open."""
        title, body = format_recall_audit_row(query, plan)
        try:
            self._vstash.remember(
                body,
                title=title,
                collection=AUDIT_COLLECTION,
                layer=AUDIT_LAYER,
            )
        except Exception:
            pass

    def _write_forget_audit(
        self,
        event_path: str,
        decision: ForgetDecision,
        derived_in_facts: list[str],
    ) -> None:
        """Audit one ``should_forget`` decision. Fail-open."""
        title, body = format_forget_audit_row(
            event_path, decision, derived_in_facts
        )
        try:
            self._vstash.remember(
                body,
                title=title,
                collection=AUDIT_COLLECTION,
                layer=AUDIT_LAYER,
            )
        except Exception:
            pass

    def _write_tombstone(
        self,
        *,
        event_path: str,
        event_text: str,
        event_title: str | None,
        event_layer: str | None,
        event_tags: str | None,
        derived_in_facts: list[str],
        reason: str,
        policy: str,
    ) -> None:
        """Persist the full-text tombstone so the event can be unforgotten.

        Unlike the audit row (which is a decision log), this stores
        everything needed to reconstruct the event: full text, title,
        layer, tags. Lives in ``engram_tombstones`` collection so a
        user can query "what did I forget?" without touching audit.

        NOT fail-open. If the tombstone write fails, we raise — we
        must not remove the event from the main collection without a
        tombstone to restore from.
        """
        title, body = format_tombstone_row(
            event_path=event_path,
            event_text=event_text,
            event_title=event_title,
            event_layer=event_layer,
            event_tags=event_tags,
            derived_in_facts=derived_in_facts,
            reason=reason,
            policy=policy,
        )
        self._vstash.remember(
            body,
            title=title,
            collection=TOMBSTONE_COLLECTION,
            layer=TOMBSTONE_LAYER,
        )

    # ------------------------------------------------------------------- misc

    def close(self) -> None:
        """Release the underlying vstash handle."""
        self._vstash.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
