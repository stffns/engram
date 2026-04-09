"""Consolidation — turning episodic events into semantic facts.

Phase 2 skeleton. **No LLM.** The v1 pipeline is deliberately crude:

1. Pull every ``layer="episodic"`` document in the engram collection.
2. Cluster them by Jaccard token overlap (single-link, greedy).
3. For each cluster of size ≥ ``min_cluster``, materialize one fact
   and write it to ``layer="semantic"`` with ``derived_from`` pointing
   back at the source document paths.
4. Singletons stay as-is in the episodic layer.

The LLM version slots in later as a different ``method`` tag on the
``Fact`` — swap the body of ``materialize_fact`` and keep everything
else. The crude version exists so we can measure whether *any*
consolidation adds value on a real scenario before we spend an LLM
budget on it.

Design invariants:

- Consolidation is **additive**. Episodic documents are never deleted.
  Forgetting is a separate decision primitive (``should_forget``) that
  lands later.
- Every fact records its provenance via ``derived_from``. No orphan
  semantic docs.
- Fact titles are derived from a stable hash of ``derived_from`` so
  repeated runs over the same episodic set produce the same fact id,
  not duplicates. (vstash's ``remember`` uses ``text://{title}`` as
  the doc path — collisions overwrite.)
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

# Lowercase alphanumeric word tokens. Strips punctuation so "writes."
# and "writes" collapse to the same token, which is what Jaccard at
# this granularity should care about.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: A recall callable used by ``cluster_by_recall``. Takes a query text
#: and a ``top_k`` and returns an iterable of hits that at least expose
#: a ``.path`` attribute. Abstracts over ``vstash.Memory.search`` so
#: tests can inject fake hits without a live vstash.
RecallFn = Callable[[str, int], list[Any]]

#: An embedding callable used by ``cluster_by_embedding``. Takes a list
#: of texts and returns a list of vectors (each a sequence of floats).
#: Abstracts over ``vstash.embed.embed_texts`` so tests can inject
#: deterministic vectors without loading a real model.
EmbedFn = Callable[[list[str]], list[Sequence[float]]]


@dataclass(frozen=True)
class Fact:
    """One consolidated memory, derived from one or more episodic events."""

    text: str
    derived_from: list[str]  # vstash doc paths for the source episodic rows
    cluster_size: int
    method: str  # "passthrough" (singleton) | "concat_v1" (cluster)


@dataclass(frozen=True)
class ConsolidationResult:
    """The outcome of a ``Memory.consolidate`` call."""

    events_examined: int
    facts_written: int
    facts: list[Fact] = field(default_factory=list)
    skipped: bool = False
    reason: str = ""
    decider: str = ""
    method: str = "jaccard_v1"


# ------------------------------------------------------------------ clustering


def _tokens(text: str) -> set[str]:
    """Lowercase alphanumeric token bag. Good enough for Jaccard v1."""
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity over token sets. Empty ∩ empty → 1.0."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def cluster_by_jaccard(
    items: list[tuple[str, str]],
    *,
    threshold: float = 0.5,
) -> list[list[tuple[str, str]]]:
    """Greedy single-link clustering on ``(id, text)`` pairs.

    Two items land in the same cluster iff their token sets exceed
    ``threshold``. Order-sensitive by design — re-clustering with a
    shuffled input may yield different groupings.

    **Known limitation (2026-04-09 prueba del vaso):** Jaccard over raw
    tokens only captures lexical overlap, not meaning. Paraphrased
    sentences that say the same thing with different vocabulary
    (normal human writing) score around 0.2 and never cluster. This
    method is genuinely useful only for near-duplicate text — copied
    snippets, template-filled strings, bot-generated lines. For
    consolidation over natural language, use ``cluster_by_recall``,
    which delegates to vstash's embedding similarity.
    """
    if not items:
        return []

    enriched = [(id_, text, _tokens(text)) for id_, text in items]
    assigned = [False] * len(enriched)
    clusters: list[list[tuple[str, str]]] = []

    for i in range(len(enriched)):
        if assigned[i]:
            continue
        cluster: list[tuple[str, str]] = [(enriched[i][0], enriched[i][1])]
        assigned[i] = True
        for j in range(i + 1, len(enriched)):
            if assigned[j]:
                continue
            if jaccard(enriched[i][2], enriched[j][2]) >= threshold:
                cluster.append((enriched[j][0], enriched[j][1]))
                assigned[j] = True
        clusters.append(cluster)

    return clusters


def cluster_by_recall(
    items: list[tuple[str, str]],
    *,
    recall_fn: RecallFn,
    top_k: int = 5,
) -> list[list[tuple[str, str]]]:
    """Cluster ``(id, text)`` items by the neighbors recall returns.

    For each item, ``recall_fn(text, top_k)`` returns hits whose
    ``.path`` is taken as a symmetric link into the same cluster.
    Connected components (via union-find) are the output clusters.

    **Known limitation (2026-04-09 prueba del vaso):** vstash's default
    hybrid search returns pure RRF ranks (``1/(60+rank)``) rather than
    absolute similarity. On tiny corpora this means *every* doc ranks
    in the top_k regardless of actual similarity, and the clustering
    collapses everything into one group. This primitive is useful
    when the neighbor source actually exposes a meaningful similarity
    signal (or when ``recall_fn`` pre-filters by a quality threshold).
    For consolidation over raw text, prefer ``cluster_by_embedding``.
    """
    if not items:
        return []

    paths = [path for path, _ in items]
    path_idx = {path: i for i, path in enumerate(paths)}
    items_by_path = dict(items)

    # Union-find with path compression.
    parent = list(range(len(paths)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for path_i, text_i in items:
        try:
            hits = recall_fn(text_i, top_k)
        except Exception:
            hits = []
        for hit in hits:
            hit_path = getattr(hit, "path", None)
            if hit_path is None or hit_path == path_i:
                continue
            if hit_path not in path_idx:
                continue
            union(path_idx[path_i], path_idx[hit_path])

    groups: dict[int, list[tuple[str, str]]] = {}
    for i, path in enumerate(paths):
        root = find(i)
        groups.setdefault(root, []).append((path, items_by_path[path]))

    return list(groups.values())


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors. Empty → 0."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def cluster_by_embedding(
    items: list[tuple[str, str]],
    *,
    embed_fn: EmbedFn,
    threshold: float = 0.65,
) -> list[list[tuple[str, str]]]:
    """Cluster ``(id, text)`` items by raw embedding cosine similarity.

    This is the v2 consolidation primitive and the engram default.
    Unlike ``cluster_by_recall``, which depends on vstash's RRF
    rankings (meaningful only on large corpora), this function asks
    the embedder for raw vectors and computes pairwise cosine
    similarity directly. Two items are linked iff their cosine
    exceeds ``threshold``; connected components are clusters.

    **Threshold calibration on the engram session corpus
    (2026-04-09):**

    - ``0.90+`` — near-paraphrase (same sentence, different wording).
    - ``0.65 – 0.90`` — same topic, different framing. Default here.
    - ``0.50 – 0.65`` — loosely related; inviting false positives.
    - ``< 0.50`` — unrelated.

    ``0.65`` is chosen as the v1 default because it clusters obvious
    paraphrases of real session content while rejecting unrelated
    topics. It is a configuration knob, not a constant — tune on the
    ``experiments/loop_quality/`` scenario that's most representative
    of the caller's stream.

    Cost model: one ``embed_fn`` call on all texts (batched), then
    O(N²) pairwise cosine. Dominated by the embedder. For N in the
    low thousands this is fine; beyond that an ANN index would help
    but consolidation is a batch operation that runs occasionally, so
    the quadratic is defensible for now.

    The embedder is an injection point: tests can pass fake vectors
    to exercise the clustering logic without loading a real model.
    """
    if not items:
        return []

    paths = [path for path, _ in items]
    texts = [text for _, text in items]
    path_idx = {path: i for i, path in enumerate(paths)}
    items_by_path = dict(items)

    try:
        vectors = embed_fn(texts)
    except Exception:
        # Embedding failure → every item is its own cluster. The
        # caller can catch the ``facts_written == 0`` signal from the
        # audit log and escalate.
        return [[(p, t)] for p, t in items]

    if len(vectors) != len(items):
        raise ValueError(
            f"embed_fn returned {len(vectors)} vectors for {len(items)} items"
        )

    parent = list(range(len(paths)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _cosine(vectors[i], vectors[j]) >= threshold:
                union(i, j)

    groups: dict[int, list[tuple[str, str]]] = {}
    for i, path in enumerate(paths):
        root = find(i)
        groups.setdefault(root, []).append((path, items_by_path[path]))

    return list(groups.values())


# ----------------------------------------------------------------- fact build


def materialize_fact(cluster: list[tuple[str, str]]) -> Fact:
    """Turn a cluster of ``(id, text)`` items into one ``Fact``.

    Singletons pass through unchanged (there is nothing to summarize).
    Clusters of two or more keep the longest text as the anchor and
    prefix it with an observation count. This is a placeholder — the
    real job belongs to an LLM — but it produces something testable
    and it never hallucinates information the episodic events didn't
    contain.
    """
    if not cluster:
        raise ValueError("cannot materialize an empty cluster")

    ids = [id_ for id_, _ in cluster]

    if len(cluster) == 1:
        return Fact(
            text=cluster[0][1],
            derived_from=ids,
            cluster_size=1,
            method="passthrough",
        )

    anchor = max((text for _, text in cluster), key=len)
    summary = f"[observed {len(cluster)}×] {anchor}"
    return Fact(
        text=summary,
        derived_from=ids,
        cluster_size=len(cluster),
        method="concat_v1",
    )


def fact_fingerprint(fact: Fact) -> str:
    """Stable short id for a fact, based on its ``derived_from`` set.

    Two consolidations over the same episodic set produce the same
    fingerprint, so writing the fact twice is idempotent (vstash keys
    by title and overwrites). The fingerprint is order-independent
    because we sort before hashing.
    """
    key = ",".join(sorted(fact.derived_from))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
