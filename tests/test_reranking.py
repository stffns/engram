"""Unit tests for merken.reranking — temporal reranking of recall results."""

from __future__ import annotations

from types import SimpleNamespace

from merken.reranking import rerank_by_recency


def _hit(score: float, added_at: str | None = None, path: str = "p") -> SimpleNamespace:
    return SimpleNamespace(score=score, added_at=added_at, path=path)


def test_weight_zero_is_noop() -> None:
    hits = [_hit(0.9, "2026-01-01T00:00:00"), _hit(0.8, "2026-06-01T00:00:00")]
    result = rerank_by_recency(hits, temporal_weight=0.0)
    assert result == hits


def test_single_result_unchanged() -> None:
    hits = [_hit(0.5, "2026-01-01T00:00:00")]
    result = rerank_by_recency(hits, temporal_weight=0.5)
    assert result == hits


def test_no_timestamps_preserves_order() -> None:
    hits = [_hit(0.9), _hit(0.8), _hit(0.7)]
    result = rerank_by_recency(hits, temporal_weight=0.3)
    assert [r.score for r in result] == [0.9, 0.8, 0.7]


def test_equal_scores_prefers_newer() -> None:
    old = _hit(0.5, "2026-01-01T00:00:00", path="old")
    new = _hit(0.5, "2026-06-01T00:00:00", path="new")
    result = rerank_by_recency([old, new], temporal_weight=0.2)
    assert result[0].path == "new"


def test_strong_old_beats_weak_new() -> None:
    old = _hit(0.9, "2026-01-01T00:00:00", path="old")
    new = _hit(0.3, "2026-06-01T00:00:00", path="new")
    result = rerank_by_recency([old, new], temporal_weight=0.2)
    # 0.9 * 1.0 = 0.9 vs 0.3 * 1.2 = 0.36 — old wins
    assert result[0].path == "old"


def test_mixed_none_timestamps() -> None:
    a = _hit(0.5, None, path="a")
    b = _hit(0.5, "2026-01-01T00:00:00", path="b")
    c = _hit(0.5, "2026-06-01T00:00:00", path="c")
    result = rerank_by_recency([a, b, c], temporal_weight=0.2)
    # c is newest → boosted most, b is oldest → no boost, a has no ts → no boost
    assert result[0].path == "c"


def test_equal_timestamps_preserves_order() -> None:
    hits = [
        _hit(0.9, "2026-01-01T00:00:00", path="a"),
        _hit(0.8, "2026-01-01T00:00:00", path="b"),
    ]
    result = rerank_by_recency(hits, temporal_weight=0.5)
    assert [r.path for r in result] == ["a", "b"]


def test_close_scores_recency_breaks_tie() -> None:
    """The knowledge_update scenario: v1 and v3 have similar scores
    but v3 is newer and should rank first."""
    v1 = _hit(0.500, "2026-01-01T00:00:00", path="v1")
    v3 = _hit(0.499, "2026-04-01T00:00:00", path="v3")
    result = rerank_by_recency([v1, v3], temporal_weight=0.2)
    # v1: 0.500 * 1.0 = 0.500, v3: 0.499 * 1.2 = 0.5988 → v3 wins
    assert result[0].path == "v3"
