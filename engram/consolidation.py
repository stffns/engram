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
    linkage: str = "complete",
) -> list[list[tuple[str, str]]]:
    """Cluster ``(id, text)`` items by raw embedding cosine similarity.

    The v2 consolidation primitive. Computes pairwise cosine from
    raw embedder vectors and groups items whose similarity exceeds
    ``threshold``.

    Parameters
    ----------
    items:
        The ``(path, text)`` pairs to cluster.
    embed_fn:
        Injection point for the embedder. Tests can pass fake
        vectors; in production, ``vstash.embed.embed_texts``.
    threshold:
        Cosine cutoff. ``0.65`` was calibrated on engram's own
        session content (see
        ``experiments/loop_quality/RESULTS.md``).
    linkage:
        Agglomerative linkage strategy.

        - ``"complete"`` (default) — two clusters merge only when
          **every** cross-cluster pair exceeds ``threshold``. Stricter;
          prevents a single weak-but-above-threshold edge from
          cascading contamination through a transitive cluster.
          O(N³) worst case; fine for consolidation batch sizes.
        - ``"single"`` — two clusters merge if **any** cross-cluster
          pair exceeds ``threshold``. Cheaper (union-find, O(N²)) but
          vulnerable to cascade: one false positive edge can merge
          two otherwise unrelated clusters. Retained for the case
          where the caller has strong confidence in the threshold
          and wants transitive behavior.

    **Why complete is the default (2026-04-09):** on the
    ``session_2026_04_09`` loop_quality scenario, single-link
    cascaded two genuine-but-cross-topic edges
    (``longmemeval_a ~ dedup_fix_a = 0.663`` and
    ``vstash_bug_a ~ dedup_fix_a = 0.652``) into a single impure
    4-event cluster. Complete-link refuses the second merge because
    the weakest pair (``vstash_bug_a ~ longmemeval_a = 0.616``) is
    below threshold. The outcome on that scenario doubles the
    query pass rate without changing the threshold. See
    ``experiments/loop_quality/RESULTS.md``.

    **Ceiling we know about:** neither linkage fixes the fundamental
    overlap of same-topic and cross-topic edges around the threshold
    on meta-discussion content. Above ~50% pass rate on that
    scenario requires either a richer signal (LLM-based
    consolidation) or a different embedder.

    The embedder callable is called exactly once on all input texts.
    Embedding failure → each item becomes its own cluster (graceful
    degradation; the caller will see ``facts_written == 0``).
    """
    if not items:
        return []

    paths = [path for path, _ in items]
    texts = [text for _, text in items]
    items_by_path = dict(items)

    try:
        vectors = embed_fn(texts)
    except Exception:
        return [[(p, t)] for p, t in items]

    if len(vectors) != len(items):
        raise ValueError(
            f"embed_fn returned {len(vectors)} vectors for {len(items)} items"
        )

    n = len(items)

    # Precompute the pairwise cosine matrix once. O(N² × dim).
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sim[i][i] = 1.0
        for j in range(i + 1, n):
            c = _cosine(vectors[i], vectors[j])
            sim[i][j] = sim[j][i] = c

    if linkage == "single":
        groups = _single_link(n, sim, threshold)
    elif linkage == "complete":
        groups = _complete_link(n, sim, threshold)
    else:
        raise ValueError(
            f"unknown linkage {linkage!r}; expected 'single' or 'complete'"
        )

    return [
        [(paths[i], items_by_path[paths[i]]) for i in sorted(group)]
        for group in groups
    ]


def _single_link(
    n: int,
    sim: list[list[float]],
    threshold: float,
) -> list[set[int]]:
    """Union-find single-link: any above-threshold edge joins clusters."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i][j] >= threshold:
                union(i, j)

    groups: dict[int, set[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, set()).add(i)
    return list(groups.values())


def _complete_link(
    n: int,
    sim: list[list[float]],
    threshold: float,
) -> list[set[int]]:
    """Agglomerative complete-link: merge only when every cross pair passes.

    Iteratively finds the cluster pair whose *weakest* cross-edge is
    the highest. If that weakest edge is above ``threshold``, the
    two clusters merge. Otherwise no more merges are possible and
    we stop. The weakest-edge criterion is equivalent to maximum
    distance in distance-space (the textbook "complete linkage").
    """
    clusters: list[set[int]] = [{i} for i in range(n)]

    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_min_sim = -1.0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Complete-link distance = min similarity across all pairs.
                min_sim = min(
                    sim[a][b] for a in clusters[i] for b in clusters[j]
                )
                if min_sim > best_min_sim:
                    best_min_sim = min_sim
                    best_pair = (i, j)

        if best_pair is None or best_min_sim < threshold:
            break

        i, j = best_pair
        clusters[i] = clusters[i] | clusters[j]
        clusters.pop(j)

    return clusters


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
