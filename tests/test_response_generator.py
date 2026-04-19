"""Tests for `merken.training.response_generator`.

Uses a fake GenerateFn. The real HF transformers path is exercised
manually via the CLI when a model is available.
"""

from __future__ import annotations

from merken.training.case_generator import GeneratedCase
from merken.training.response_generator import (
    CaseWithResponse,
    GenerateFn,
    ResponseGenerator,
)


def _fake_gen(prefix: str = "model says: ") -> GenerateFn:
    def _fn(prompt: str) -> str:
        return f"{prefix}{prompt[:30]}"
    return _fn


def test_respond_attaches_model_response_to_case() -> None:
    gen = ResponseGenerator(_fake_gen())
    case = GeneratedCase(
        case_id="c1", prompt="what dose for pneumonia?",
        truth="50 mg/kg/day", metadata={"protocol_id": "p1"},
    )
    out = gen.respond(case)
    assert isinstance(out, CaseWithResponse)
    assert out.case_id == "c1"
    assert out.prompt == case.prompt
    assert out.truth == case.truth
    assert out.model_response.startswith("model says:")
    assert out.metadata == {"protocol_id": "p1"}


def test_respond_strips_whitespace_from_response() -> None:
    def gen(_: str) -> str:
        return "   hello   \n"
    out = ResponseGenerator(gen).respond(GeneratedCase(
        case_id="c", prompt="p", truth="t",
    ))
    assert out.model_response == "hello"


def test_respond_many_preserves_order() -> None:
    gen = ResponseGenerator(_fake_gen())
    cases = [
        GeneratedCase(case_id=f"c{i}", prompt=f"p{i}", truth=f"t{i}")
        for i in range(3)
    ]
    out = gen.respond_many(cases)
    assert len(out) == 3
    for i, r in enumerate(out):
        assert r.case_id == f"c{i}"


def test_respond_does_not_mutate_input_metadata() -> None:
    case = GeneratedCase(
        case_id="c", prompt="p", truth="t",
        metadata={"k": "v"},
    )
    out = ResponseGenerator(_fake_gen()).respond(case)
    out.metadata["new"] = "value"
    assert "new" not in case.metadata  # input untouched
