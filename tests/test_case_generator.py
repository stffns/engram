"""Tests for `merken.training.case_generator`.

Uses a fake LLMClient so the suite stays offline + deterministic.
The real anthropic-backed path is exercised manually via the CLI.
"""

from __future__ import annotations

import pytest

from merken.training.case_generator import (
    CaseGenerator,
    GeneratedCase,
    LLMClient,
    ProtocolClause,
    _extract_json_array,
)


# ----------------------------------------------------------------- json extract

def test_extract_clean_json_array() -> None:
    raw = '[{"prompt": "p1", "truth": "t1"}, {"prompt": "p2", "truth": "t2"}]'
    out = _extract_json_array(raw)
    assert len(out) == 2
    assert out[0]["prompt"] == "p1"


def test_extract_strips_markdown_fences() -> None:
    raw = '```json\n[{"prompt": "p", "truth": "t"}]\n```'
    out = _extract_json_array(raw)
    assert out == [{"prompt": "p", "truth": "t"}]


def test_extract_strips_unlabeled_fences() -> None:
    raw = '```\n[{"prompt": "p", "truth": "t"}]\n```'
    out = _extract_json_array(raw)
    assert out == [{"prompt": "p", "truth": "t"}]


def test_extract_skips_leading_prose() -> None:
    raw = 'Here are the cases:\n[{"prompt": "p", "truth": "t"}]'
    out = _extract_json_array(raw)
    assert len(out) == 1


def test_extract_raises_on_missing_brackets() -> None:
    with pytest.raises(ValueError, match="could not locate JSON array"):
        _extract_json_array("just some prose with no JSON here")


def test_extract_raises_on_invalid_json() -> None:
    with pytest.raises(ValueError, match="JSON array parse failed"):
        _extract_json_array('[{"prompt": "p", broken}]')


def test_extract_raises_on_object_not_array() -> None:
    with pytest.raises(ValueError, match="locate JSON array"):
        _extract_json_array('{"prompt": "p", "truth": "t"}')


# ----------------------------------------------------------------- generation

def _fake_client(canned_response: str) -> LLMClient:
    """Build an LLMClient that ignores prompts and returns a canned string."""
    def _fn(system: str, user: str) -> str:
        return canned_response
    return _fn


def test_generate_emits_n_cases_with_stable_ids() -> None:
    canned = (
        '[{"prompt": "child 5y/o, fever, cough", '
        '  "truth": "amoxicillin 50 mg/kg/day for 5 days"},'
        ' {"prompt": "child 3y/o, productive cough", '
        '  "truth": "amoxicillin 50 mg/kg/day for 5 days"}]'
    )
    gen = CaseGenerator(_fake_client(canned))
    clause = ProtocolClause(
        protocol_id="who_pneumonia_amox",
        text="...",
        metadata={"source": "WHO"},
    )
    cases = gen.generate(clause, n=2)
    assert len(cases) == 2
    assert all(isinstance(c, GeneratedCase) for c in cases)
    assert cases[0].case_id == "who_pneumonia_amox__case_000"
    assert cases[1].case_id == "who_pneumonia_amox__case_001"
    # Metadata carries protocol_id + clause metadata.
    assert cases[0].metadata["protocol_id"] == "who_pneumonia_amox"
    assert cases[0].metadata["source"] == "WHO"


def test_generate_respects_n_zero() -> None:
    gen = CaseGenerator(_fake_client("[]"))
    cases = gen.generate(ProtocolClause("p", "..."), n=0)
    assert cases == []


def test_generate_skips_malformed_rows() -> None:
    canned = (
        '[{"prompt": "ok", "truth": "ok"},'
        ' {"prompt": "missing truth"},'   # incomplete
        ' "not even a dict",'             # wrong type
        ' {"truth": "missing prompt"},'   # incomplete
        ' {"prompt": "valid two", "truth": "valid"}]'
    )
    gen = CaseGenerator(_fake_client(canned))
    cases = gen.generate(ProtocolClause("p", "..."), n=5)
    assert len(cases) == 2
    assert cases[0].prompt == "ok"
    assert cases[1].prompt == "valid two"


def test_generate_strips_whitespace_in_fields() -> None:
    canned = '[{"prompt": "  query  ", "truth": "\\n\\nanswer\\n  "}]'
    gen = CaseGenerator(_fake_client(canned))
    cases = gen.generate(ProtocolClause("p", "..."), n=1)
    assert cases[0].prompt == "query"
    assert cases[0].truth == "answer"


def test_generate_propagates_extraction_errors() -> None:
    gen = CaseGenerator(_fake_client("totally not json"))
    with pytest.raises(ValueError):
        gen.generate(ProtocolClause("p", "..."), n=1)
