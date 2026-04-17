"""Env-var shadow-mode wiring.

If ``MERKEN_SHADOW`` is set, ``Memory`` auto-wraps its default decider
in a ``ShadowWriteDecider`` whose secondary is constructed from the
other ``MERKEN_SHADOW_*`` vars. This lets CLI, MCP, and SDK consumers
opt into shadow mode with zero code changes.

Torch is imported **at module evaluation** (not inside a function) when
shadow is enabled, so that ``from merken import ...`` causes torch to
load BEFORE vstash/fastembed. That's the Mistake #10 order requirement
(notes/nanogpt-training-log.md): the first native library to load wins
the macOS process; later ones can segfault. Keep this module as the
first import in ``merken/__init__.py``.

Supported env vars:

- ``MERKEN_SHADOW=nanogpt``
    + ``MERKEN_SHADOW_NANOGPT_CKPT=/path/to/ckpt.pt``
    + ``MERKEN_SHADOW_NANOGPT_META=/path/to/meta.pkl``
- ``MERKEN_SHADOW=llm``
    + ``MERKEN_SHADOW_LLM_MODEL=google/gemma-3-270m-it`` (HF id or local path)
    + ``MERKEN_SHADOW_LLM_DEVICE=cpu`` (optional, default ``cpu``)

Anything else (empty / unset / unknown) -> shadow disabled.
"""

from __future__ import annotations

import os

_ENABLED_VALUES = {"nanogpt", "llm"}
_KIND = os.environ.get("MERKEN_SHADOW", "").strip().lower()

# Eagerly import torch if shadow is enabled. This must happen before any
# other merken module that transitively imports vstash (fastembed/ONNX)
# -- see notes/nanogpt-training-log.md Mistake #10.
if _KIND in _ENABLED_VALUES:
    import torch  # noqa: F401


def load_shadow_from_env():
    """Return a shadow ``WriteDecider`` constructed from env, or ``None``."""
    if _KIND not in _ENABLED_VALUES:
        return None

    if _KIND == "nanogpt":
        from merken.classifiers.nanogpt import NanoGPTWriteDecider

        ckpt = os.environ.get("MERKEN_SHADOW_NANOGPT_CKPT")
        meta = os.environ.get("MERKEN_SHADOW_NANOGPT_META")
        if not (ckpt and meta):
            raise RuntimeError(
                "MERKEN_SHADOW=nanogpt requires MERKEN_SHADOW_NANOGPT_CKPT "
                "and MERKEN_SHADOW_NANOGPT_META env vars."
            )
        return NanoGPTWriteDecider(ckpt, meta)

    if _KIND == "llm":
        from merken.classifiers.llm import LLMWriteDecider

        model = os.environ.get("MERKEN_SHADOW_LLM_MODEL")
        if not model:
            raise RuntimeError(
                "MERKEN_SHADOW=llm requires MERKEN_SHADOW_LLM_MODEL env var."
            )
        device = os.environ.get("MERKEN_SHADOW_LLM_DEVICE", "cpu")
        return LLMWriteDecider(model_name=model, device=device)

    return None
