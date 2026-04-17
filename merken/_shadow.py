"""Env-var activation for shadow mode and graduation.

Two roles for a classifier (nanoGPT / LLM) via env:

- ``MERKEN_SHADOW=nanogpt|llm`` -- auditing-only. The default
  ``HeuristicWriteDecider`` stays primary and controls writes; the
  classifier only annotates the audit reason with its prediction and
  an agree/disagree tag. Bootstrap mode.

- ``MERKEN_PRIMARY=nanogpt|llm`` -- graduation. The classifier becomes
  authoritative, chained *after* ``HeuristicWriteDecider`` so that the
  hygiene gates (empty, too-short, too-long, exact-dup) still fire
  before the classifier sees the event. Only use this once the shadow
  logs + oracular labels have shown the classifier beats the gate-only
  baseline on held-out content.

Only one of ``MERKEN_SHADOW`` / ``MERKEN_PRIMARY`` should be set at a
time. If both are set the primary role wins (ignores SHADOW) -- a
deliberately conservative tie-breaker: whoever typed ``PRIMARY`` meant
it.

Torch is imported **at module evaluation** (not inside a function) when
either role is enabled, so that ``from merken import ...`` causes torch
to load BEFORE vstash/fastembed. That's the Mistake #10 order
requirement (notes/nanogpt-training-log.md): the first native library
to load wins the macOS process; later ones can segfault. Keep this
module as the first import in ``merken/__init__.py``.

Supported backends (shared between SHADOW and PRIMARY):

- ``nanogpt``
    + ``MERKEN_<role>_NANOGPT_CKPT=/path/to/ckpt.pt``
    + ``MERKEN_<role>_NANOGPT_META=/path/to/meta.pkl``
- ``llm``
    + ``MERKEN_<role>_LLM_MODEL=google/gemma-3-270m-it``
    + ``MERKEN_<role>_LLM_DEVICE=cpu`` (optional, default ``cpu``)

where ``<role>`` is ``SHADOW`` or ``PRIMARY``.

Anything else (empty / unset / unknown) -> no classifier wired.
"""

from __future__ import annotations

import os

_ENABLED_VALUES = {"nanogpt", "llm"}
_SHADOW_KIND = os.environ.get("MERKEN_SHADOW", "").strip().lower()
_PRIMARY_KIND = os.environ.get("MERKEN_PRIMARY", "").strip().lower()

# Eagerly import torch if either role is enabled. This must happen
# before any other merken module that transitively imports vstash
# (fastembed/ONNX). See notes/nanogpt-training-log.md Mistake #10.
if _SHADOW_KIND in _ENABLED_VALUES or _PRIMARY_KIND in _ENABLED_VALUES:
    import torch  # noqa: F401


def _build_classifier(kind: str, role: str):
    """Construct the classifier for ``role`` (``SHADOW`` / ``PRIMARY``)."""
    if kind == "nanogpt":
        from merken.classifiers.nanogpt import NanoGPTWriteDecider

        ckpt = os.environ.get(f"MERKEN_{role}_NANOGPT_CKPT")
        meta = os.environ.get(f"MERKEN_{role}_NANOGPT_META")
        if not (ckpt and meta):
            raise RuntimeError(
                f"MERKEN_{role}=nanogpt requires MERKEN_{role}_NANOGPT_CKPT "
                f"and MERKEN_{role}_NANOGPT_META env vars."
            )
        return NanoGPTWriteDecider(ckpt, meta)

    if kind == "llm":
        from merken.classifiers.llm import LLMWriteDecider

        model = os.environ.get(f"MERKEN_{role}_LLM_MODEL")
        if not model:
            raise RuntimeError(
                f"MERKEN_{role}=llm requires MERKEN_{role}_LLM_MODEL env var."
            )
        device = os.environ.get(f"MERKEN_{role}_LLM_DEVICE", "cpu")
        return LLMWriteDecider(model_name=model, device=device)

    raise RuntimeError(f"unknown MERKEN_{role} backend: {kind!r}")


def load_shadow_from_env():
    """Return a shadow ``WriteDecider`` constructed from env, or ``None``.

    Only returns a classifier when ``MERKEN_SHADOW`` is set AND
    ``MERKEN_PRIMARY`` is NOT set. When PRIMARY wins, there is no
    "shadow" decider -- graduation mode is strictly about putting the
    classifier on the write path.
    """
    if _PRIMARY_KIND in _ENABLED_VALUES:
        return None
    if _SHADOW_KIND not in _ENABLED_VALUES:
        return None
    return _build_classifier(_SHADOW_KIND, "SHADOW")


def load_primary_from_env():
    """Return the graduated classifier constructed from env, or ``None``.

    When this returns a non-None value, ``Memory`` will chain it AFTER
    a ``HeuristicWriteDecider`` gate so hygiene rules still fire, and
    will NOT wrap it in shadow mode.
    """
    if _PRIMARY_KIND not in _ENABLED_VALUES:
        return None
    return _build_classifier(_PRIMARY_KIND, "PRIMARY")
