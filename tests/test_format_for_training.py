"""Tests for the Phase 3 dataset format converter.

Verifies the two label-shape options match what
``notes/midloop-training-plan.md`` documents and what the
NanoGPTMidloopDecider trainer will expect.
"""

from __future__ import annotations

from experiments.midloop_pilot.format_for_training import (
    convert_case,
    labels_for_response,
    tokenize,
)


def test_tokenize_matches_aligner_contract() -> None:
    """The converter MUST tokenize the same way the aligner did,
    otherwise token indices in intervention spans are nonsense.
    """
    from merken.training.midloop_dataset import tokenize as aligner_tokenize
    s = "Administer IV fluids and obtain urgent abdominal ultrasound"
    assert tokenize(s) == aligner_tokenize(s)


def test_span_labeling_marks_every_token_inside_span() -> None:
    tokens = ["a", "b", "c", "d", "e"]
    spans = [{"model_token_start": 1, "model_token_end": 4}]  # b,c,d
    labels = labels_for_response(tokens, spans, mode="span")
    assert labels == [0, 1, 1, 1, 0]


def test_boundary_labeling_marks_only_start() -> None:
    tokens = ["a", "b", "c", "d", "e"]
    spans = [{"model_token_start": 1, "model_token_end": 4}]
    labels = labels_for_response(tokens, spans, mode="boundary")
    # Only index 1 (start of [1,4)) gets the label.
    assert labels == [0, 1, 0, 0, 0]


def test_multi_span_boundary_one_label_per_span() -> None:
    tokens = ["a", "b", "c", "d", "e", "f"]
    spans = [
        {"model_token_start": 0, "model_token_end": 2},
        {"model_token_start": 4, "model_token_end": 6},
    ]
    labels = labels_for_response(tokens, spans, mode="boundary")
    assert labels == [1, 0, 0, 0, 1, 0]


def test_unknown_mode_raises() -> None:
    import pytest
    with pytest.raises(ValueError, match="unknown labeling mode"):
        labels_for_response(["a"], [], mode="bogus")


def test_convert_case_full_round_trip() -> None:
    row = {
        "case_id": "test_001",
        "prompt": "what dose for kid pneumonia?",
        "model_response": "Administer IV fluids urgently",
        "interventions": [
            {
                "model_token_start": 0,
                "model_token_end": 4,
                "truth_text": "REFER URGENTLY to hospital",
                "model_text": "Administer IV fluids urgently",
                "cosine_sim": 0.55,
            }
        ],
        "metadata": {"protocol_id": "pneumonia"},
    }
    out = convert_case(row, labeling="boundary")
    assert out is not None
    assert out["case_id"] == "test_001"
    assert out["prompt"] == "what dose for kid pneumonia?"
    assert out["response_tokens"] == [
        "Administer", "IV", "fluids", "urgently",
    ]
    # boundary -> only first token labeled
    assert out["intervene_labels"] == [1, 0, 0, 0]
    assert out["labeling_mode"] == "boundary"
    assert out["metadata"] == {"protocol_id": "pneumonia"}
    # spans preserved for traceability
    assert len(out["intervention_spans"]) == 1
    assert out["intervention_spans"][0]["truth_text"] == "REFER URGENTLY to hospital"


def test_convert_case_skips_empty_response() -> None:
    row = {"case_id": "x", "model_response": "", "interventions": []}
    assert convert_case(row) is None


def test_convert_case_skips_whitespace_only_response() -> None:
    row = {"case_id": "x", "model_response": "   \n\t  ", "interventions": []}
    assert convert_case(row) is None


def test_convert_case_no_interventions_all_zero_labels() -> None:
    row = {
        "case_id": "perfect",
        "prompt": "trivial",
        "model_response": "exactly the right answer",
        "interventions": [],
        "metadata": {},
    }
    out = convert_case(row, labeling="boundary")
    assert out is not None
    assert out["intervene_labels"] == [0, 0, 0, 0]  # 4 tokens


def test_span_labeling_clamps_to_token_bounds() -> None:
    # Span end > len(tokens) should not crash; just clamps.
    tokens = ["a", "b", "c"]
    spans = [{"model_token_start": 1, "model_token_end": 99}]
    labels = labels_for_response(tokens, spans, mode="span")
    assert labels == [0, 1, 1]


def test_boundary_labeling_skips_out_of_range_start() -> None:
    tokens = ["a", "b", "c"]
    spans = [{"model_token_start": 99, "model_token_end": 100}]
    labels = labels_for_response(tokens, spans, mode="boundary")
    assert labels == [0, 0, 0]


def test_accepts_dataclass_direct_field_names() -> None:
    """JSONL writers can use either dialect:
       - model_token_start / model_token_end (CLI rename)
       - model_start / model_end (direct from Region dataclass)
    Per PR #24 review (Gemini); both must work for forward compat.
    """
    tokens = ["a", "b", "c", "d", "e"]
    # dataclass-direct shape
    spans_direct = [{"model_start": 1, "model_end": 4}]
    assert labels_for_response(tokens, spans_direct, mode="boundary") == [0, 1, 0, 0, 0]
    assert labels_for_response(tokens, spans_direct, mode="span") == [0, 1, 1, 1, 0]


def test_token_keys_take_precedence_when_both_present() -> None:
    """If a span has BOTH key shapes, the explicit ``model_token_*``
    keys win -- consistent with the upstream CLI being the canonical
    JSONL writer."""
    tokens = ["a", "b", "c", "d", "e"]
    spans = [{
        "model_token_start": 1, "model_token_end": 2,
        "model_start": 3, "model_end": 4,
    }]
    assert labels_for_response(tokens, spans, mode="boundary") == [0, 1, 0, 0, 0]
