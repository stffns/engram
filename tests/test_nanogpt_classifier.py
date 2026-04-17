"""Smoke tests for NanoGPTWriteDecider.

These tests only run when a trained checkpoint is available on disk.
Both char-level (v3b/out-merken-v3) and BPE (v4/out-merken-bpe) checkpoints
live outside the merken repo, under ../nanoGPT/. Skip cleanly if missing
so CI without the artifacts still passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.nanogpt

pytest.importorskip("torch")

from merken.classifiers.nanogpt import NanoGPTWriteDecider, extract_verb_marker  # noqa: E402
from merken.policies import Event, WriteContext  # noqa: E402

_NANOGPT = Path("/Users/jaysonsteffens/Desktop/Personal/Projects/nanoGPT")
_BPE_CKPT = _NANOGPT / "out-merken-bpe" / "ckpt.pt"
_BPE_META = _NANOGPT / "data" / "merken_bpe" / "meta.pkl"
_CHAR_CKPT = _NANOGPT / "out-merken" / "ckpt.pt"
_CHAR_META = _NANOGPT / "data" / "merken" / "meta.pkl"


def _ctx() -> WriteContext:
    return WriteContext(project="test")


def test_verb_marker_extraction() -> None:
    assert extract_verb_marker("Replaced Redis with Caffeine.") == "[VERB:replaced]"
    assert extract_verb_marker("Discussed switching caches.") == "[VERB:discussed]"
    assert extract_verb_marker("Sprint planning: infra team.") == "[CTX:routine]"
    assert extract_verb_marker("Redis memory usage spiked.") == "[CTX:unknown]"


@pytest.mark.skipif(
    not (_BPE_CKPT.exists() and _BPE_META.exists()),
    reason="BPE checkpoint not present",
)
def test_bpe_decider_separates_decision_from_noise() -> None:
    pytest.importorskip("tokenizers")
    decider = NanoGPTWriteDecider(_BPE_CKPT, _BPE_META)
    assert decider._bpe is not None, "should auto-detect BPE mode"
    assert decider._use_markers is True, "BPE default should enable verb markers"

    decision = decider.decide(
        Event(text="Replaced Redis with Caffeine for session caching."),
        _ctx(),
    )
    assert decision.write is True, decision.reason

    noise = decider.decide(
        Event(text="Sprint planning: infra team prioritized CI pipeline."),
        _ctx(),
    )
    assert noise.write is False, noise.reason


@pytest.mark.skipif(
    not (_CHAR_CKPT.exists() and _CHAR_META.exists()),
    reason="char-level v2 checkpoint not present",
)
def test_char_decider_still_works() -> None:
    decider = NanoGPTWriteDecider(_CHAR_CKPT, _CHAR_META)
    assert decider._bpe is None, "char mode should not load a BPE tokenizer"
    assert decider._use_markers is False, "char-level v2 was trained without markers"

    decision = decider.decide(
        Event(text="Replaced Redis with Caffeine for session caching."),
        _ctx(),
    )
    assert decision.write is True, decision.reason

    noise = decider.decide(
        Event(text="Sprint planning: infra team prioritized CI pipeline."),
        _ctx(),
    )
    assert noise.write is False, noise.reason
