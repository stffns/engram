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
    """gpt-oss-120b populates ``message.reasoning`` even on a
    reasoning-neutral prompt (no explicit "think step by step").

    The probe under ``experiments/phase2_distillation/`` showed
    10/10 reasoning-neutral prompts populated the channel; if the
    only contract test biases the prompt with an explicit ask, a
    future model regression that requires the explicit ask would
    pass green. Use a plain factual question.
    """
    from experiments.midloop_concept.medlocal.cerebras_midloop import (
        cerebras_chat_capture,
    )

    out = cerebras_chat_capture(
        model="gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You answer questions briefly."},
            {"role": "user", "content": "What is 17 + 25?"},
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
        "reasoning_present=True but reasoning text is empty on a "
        "reasoning-neutral prompt. The probe showed gpt-oss-120b "
        "populates the channel without explicit asks; a regression "
        "to ask-required reasoning would pass a biased contract "
        "test but break Phase 2 SFT data generation."
    )
    assert isinstance(out["content"], str) and out["content"].strip(), (
        "content empty; gpt-oss-120b returned no visible answer."
    )
    assert isinstance(out["usage"], dict)
    assert out["usage"].get("total_tokens", 0) > 0


def test_capture_handles_non_reasoning_model():
    """Empirical: Cerebras SDK exposes ``message.reasoning`` on
    all models, but the value is ``None`` on non-reasoning models.
    So the actual reasoning-vs-not discriminator is
    ``reasoning is not None``, not ``reasoning_present``.

    The ``reasoning_present`` flag remains a defensive guard for
    a future SDK schema change that drops the attribute entirely
    -- if Cerebras ever ships a build where ``msg.reasoning``
    raises AttributeError, this helper degrades to
    ``reasoning_present=False`` rather than crashing.

    This test pins the empirical observation so a future SDK
    upgrade that *does* start populating reasoning on llama3.1-8b
    surfaces as a test failure for review (it would not be a bug,
    but it would be a contract change).
    """
    from experiments.midloop_concept.medlocal.cerebras_midloop import (
        cerebras_chat_capture,
    )

    out = cerebras_chat_capture(
        model="llama3.1-8b",
        messages=[
            {"role": "system", "content": "You answer questions briefly."},
            {"role": "user", "content": "What is 17 + 25?"},
        ],
        max_tokens=64,
        temperature=0.0,
    )

    # SDK currently always sets the attribute (so present=True on
    # both reasoning and non-reasoning models).
    assert out["reasoning_present"] is True, (
        "Cerebras SDK no longer exposes message.reasoning on "
        "llama3.1-8b. The helper's sentinel logic still works -- "
        "this assert just pins the empirical SDK behavior."
    )
    # Non-reasoning model does not populate the channel.
    assert out["reasoning"] is None, (
        "llama3.1-8b populated message.reasoning. Either Cerebras "
        "rolled out reasoning-on-by-default, the model has been "
        "retrained, or upstream behavior changed. Re-evaluate the "
        "Phase 2 distillation plan -- a non-reasoning teacher with "
        "reasoning traces shifts the cost / quality picture."
    )
    assert isinstance(out["content"], str) and out["content"].strip()
