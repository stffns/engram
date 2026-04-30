"""Smoke test for the reasoning-capture variant of cerebras_chat.

Live test (paid API). Excluded from the default suite via the
``cerebras_live`` marker -- run explicitly with::

    pytest -m cerebras_live tests/test_cerebras_chat_capture.py

Tests the load-bearing claim of the Phase 2 distillation plan:
gpt-oss-120b on Cerebras populates ``message.reasoning`` with text
that the SDK exposes as a field on the message object. The probe
under ``experiments/phase2_distillation/`` confirmed this on N=10;
this test pins the contract so it cannot regress silently when the
SDK or model is upgraded.

Cost: 1 paid call per test invocation, ~$0.05.
"""

from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.cerebras_live


@pytest.fixture(autouse=True)
def _require_api_key():
    if not os.environ.get("CEREBRAS_API_KEY"):
        pytest.skip("CEREBRAS_API_KEY not set; skipping live Cerebras test")


def test_capture_returns_reasoning_for_gpt_oss():
    """gpt-oss-120b on Cerebras must populate message.reasoning."""
    from experiments.midloop_concept.medlocal.cerebras_midloop import (
        cerebras_chat_capture,
    )

    out = cerebras_chat_capture(
        model="gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You answer questions briefly."},
            {"role": "user", "content": "What is 17 + 25? Reason step by step."},
        ],
        max_tokens=512,
        temperature=0.0,
    )

    assert set(out.keys()) >= {
        "content",
        "reasoning",
        "reasoning_present",
        "wall_s",
        "usage",
    }, f"missing required keys: {set(out.keys())}"

    assert out["reasoning_present"] is True, (
        "Cerebras gpt-oss-120b did not expose message.reasoning. "
        "If this is a Cerebras SDK upgrade, the Phase 2 distillation "
        "pipeline assumption is broken; see "
        "notes/2026-04-30-phase2-distillation-probe-go.md."
    )
    assert isinstance(out["reasoning"], str) and out["reasoning"].strip(), (
        "reasoning_present=True but reasoning text is empty -- model "
        "chose not to populate the channel on a question explicitly "
        "asking for step-by-step reasoning. This is a regression in "
        "the model's reasoning behavior."
    )
    assert isinstance(out["content"], str) and out["content"].strip(), (
        "content empty; gpt-oss-120b returned no visible answer."
    )
    assert isinstance(out["usage"], dict)
    assert out["usage"].get("total_tokens", 0) > 0
