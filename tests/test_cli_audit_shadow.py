"""Tests for the `merken audit --shadow-*` CLI helpers.

The E2E smoke test for the subcommand lives in test_cli.py; these
tests pin the parsing helpers (``_parse_audit_row`` and
``_shadow_tag_from_reason``) that do the real work.
"""

from __future__ import annotations

from merken.cli import _parse_audit_row, _shadow_tag_from_reason


def test_parse_audit_row_extracts_fields() -> None:
    block = (
        "timestamp: 2026-04-17T08:07:18+00:00\n"
        "decision: should_remember\n"
        "write: True\n"
        "reason: novel|shadow_disagree:nanogpt-classifier=skip:1.000\n"
        "policy: Shadow(HeuristicWriteDecider|nanogpt-classifier)\n"
        "event_title: t2\n"
        "event_text_preview: Sprint planning: infra team prioritized CI.\n"
    )
    out = _parse_audit_row(block)
    assert out["decision"] == "should_remember"
    assert out["write"] == "True"
    assert out["event_title"] == "t2"
    assert (
        out["reason"]
        == "novel|shadow_disagree:nanogpt-classifier=skip:1.000"
    )


def test_parse_audit_row_empty_input() -> None:
    assert _parse_audit_row("") == {}


def test_shadow_tag_disagree() -> None:
    tag = _shadow_tag_from_reason(
        "novel|shadow_disagree:nanogpt-classifier=skip:0.850"
    )
    assert tag == ("shadow_disagree", "skip", "0.850")


def test_shadow_tag_agree() -> None:
    tag = _shadow_tag_from_reason(
        "novel|shadow_agree:nanogpt-classifier=write:0.997"
    )
    assert tag == ("shadow_agree", "write", "0.997")


def test_shadow_tag_error_has_no_label() -> None:
    tag = _shadow_tag_from_reason("novel|shadow_error:RuntimeError")
    assert tag is not None
    marker, label, conf = tag
    assert marker == "shadow_error"
    assert label == ""
    assert "RuntimeError" in conf


def test_shadow_tag_absent_returns_none() -> None:
    assert _shadow_tag_from_reason("novel") is None
    assert _shadow_tag_from_reason("too_short:<8") is None
