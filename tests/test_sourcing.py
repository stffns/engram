"""Tests for `merken.sourcing` -- Type A vs Type B fail-closed predicate.

The fail-closed semantics are load-bearing for the MedLocal use
case: an event with an unknown ``source:<value>`` tag must be
treated as IF it were authoritative for mutation purposes, even
though we cannot prove it. Better one extra event in the store
than a tombstoned protocol.
"""

from __future__ import annotations

import pytest

from merken.sourcing import (
    KNOWN_AUTHORITATIVE,
    KNOWN_DERIVED,
    is_authoritative,
    is_safely_mutable,
)


class TestIsSafelyMutable:
    """The predicate that consolidate / forget / future decay use."""

    def test_no_tags_defaults_to_mutable(self) -> None:
        # Legacy / untagged events stay Type A so existing flows
        # continue to work after the predicate ships.
        assert is_safely_mutable(None) is True
        assert is_safely_mutable("") is True

    def test_known_derived_source_is_mutable(self) -> None:
        for source in KNOWN_DERIVED:
            assert is_safely_mutable(f"source:{source}") is True, source

    def test_known_authoritative_source_is_immutable(self) -> None:
        for source in KNOWN_AUTHORITATIVE:
            assert is_safely_mutable(f"source:{source}") is False, source

    def test_unknown_source_value_fails_closed(self) -> None:
        # Critical: an unrecognized source value (typo, deprecated
        # category, format change) must NOT mutate. Asymmetric cost:
        # one extra row vs. an irreversible protocol drop.
        assert is_safely_mutable("source:autoritative") is False
        assert is_safely_mutable("source:foo_made_up") is False
        assert is_safely_mutable("source:") is False  # empty after colon

    def test_other_tags_alongside_source_dont_interfere(self) -> None:
        assert is_safely_mutable("source:session,topic:nanogpt") is True
        assert is_safely_mutable("topic:nanogpt,source:session") is True
        assert is_safely_mutable("topic:nanogpt,source:protocol_who") is False

    def test_no_source_tag_at_all_defaults_to_mutable(self) -> None:
        # Tags exist but no `source:<value>` -> legacy untagged.
        assert is_safely_mutable("topic:foo,decision:bar") is True

    def test_mixed_sources_immutable_wins(self) -> None:
        # If ANY source tag is non-derived, fail closed.
        assert is_safely_mutable("source:session,source:authoritative") is False
        assert is_safely_mutable("source:authoritative,source:session") is False

    def test_whitespace_tolerant(self) -> None:
        assert is_safely_mutable("source:session ") is True
        assert is_safely_mutable(" source:session") is True
        assert is_safely_mutable("source:authoritative ") is False

    def test_case_insensitive_value(self) -> None:
        # Tags from different ingest paths may capitalize differently.
        assert is_safely_mutable("source:AUTHORITATIVE") is False
        assert is_safely_mutable("source:Session") is True


class TestIsAuthoritative:
    """The convenience predicate for explicit citation preference."""

    def test_no_tags_is_not_authoritative(self) -> None:
        assert is_authoritative(None) is False
        assert is_authoritative("") is False

    def test_known_authoritative_returns_true(self) -> None:
        for source in KNOWN_AUTHORITATIVE:
            assert is_authoritative(f"source:{source}") is True, source

    def test_known_derived_returns_false(self) -> None:
        for source in KNOWN_DERIVED:
            assert is_authoritative(f"source:{source}") is False, source

    def test_unknown_returns_false_distinct_from_immutable(self) -> None:
        # Unknown values are immutable (fail-closed) but NOT
        # explicitly authoritative -- they don't get citation
        # preference. This asymmetry is intentional.
        assert is_safely_mutable("source:foo_unknown") is False
        assert is_authoritative("source:foo_unknown") is False


@pytest.mark.parametrize("legacy_tag", [
    "source:precompact,session:abc",
    "source:backfill,session:def",
    "source:session,decision:next-move,hypothesis:H2",
    "source:transcript,project:engram",
])
def test_existing_engram_tag_patterns_remain_mutable(legacy_tag: str) -> None:
    # These are tag strings observed in the live engram DB at the
    # time the predicate shipped (2026-04-19). The fail-closed
    # design must NOT regress them; otherwise consolidate / forget
    # silently break for everyone.
    assert is_safely_mutable(legacy_tag) is True
