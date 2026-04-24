"""Monkey-patch vstash.embed to route a specific model_name to v5-lora.

Temporary bridge until vstash supports per-collection encoder plug-in
(see pending vstash issue). Activates on import by calling `install()`.

The patched functions still delegate to the original vstash
implementation for any model_name that isn't the v5-lora sentinel, so
mixed runs (bge + v5-lora in the same process) are supported.

Sentinel model_name: `v5-lora:<adapter_out_dir>` -- any model_name
starting with "v5-lora:" is routed to this shim.

Usage:
    import vstash  # must be imported first
    from experiments.retrieval.bge_lme_ft.vstash_v5lora_shim import install
    install()
    # Now any vstash call that uses embed_texts("v5-lora:...", ...) will
    # use the LoRA-adapted encoder.
"""

from __future__ import annotations

import threading
from pathlib import Path

import vstash.embed as _ve

_SENTINEL = "v5-lora:"
_models: dict[str, object] = {}
_lock = threading.Lock()
_force_model: str | None = None  # when set via install(auto_set_model=...), ALL embed calls route through this

_orig_embed_texts = _ve.embed_texts
_orig_embed_query = _ve.embed_query
_orig_get_dim = _ve.get_embedding_dim


def _get_model(model_name: str):
    """Lazy-load and cache the LoRA-adapted SentenceTransformer."""
    if model_name in _models:
        return _models[model_name]
    with _lock:
        if model_name in _models:
            return _models[model_name]
        assert model_name.startswith(_SENTINEL)
        path = model_name[len(_SENTINEL):]
        if not Path(path).exists():
            raise FileNotFoundError(
                f"v5-lora adapter dir not found: {path}. model_name={model_name!r}"
            )
        from experiments.retrieval.bge_lme_ft.load_v5_lora import load_v5_lora
        m = load_v5_lora(path)
        m.eval()
        _models[model_name] = m
        print(
            f"[shim] loaded v5-lora from {path} dim={m.get_sentence_embedding_dimension()}",
            flush=True,
        )
        return m


def _shim_embed_texts(texts, model_name, backend="auto"):
    if _force_model is not None:
        model_name = _force_model
    if not model_name.startswith(_SENTINEL):
        return _orig_embed_texts(texts, model_name, backend)
    m = _get_model(model_name)
    arr = m.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return arr.tolist()


def _shim_embed_query(text, model_name, backend="auto"):
    if _force_model is not None:
        model_name = _force_model
    if not model_name.startswith(_SENTINEL):
        return _orig_embed_query(text, model_name, backend)
    m = _get_model(model_name)
    arr = m.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return arr[0].tolist()


def _shim_get_dim(model_name):
    if _force_model is not None:
        model_name = _force_model
    if not model_name.startswith(_SENTINEL):
        return _orig_get_dim(model_name)
    m = _get_model(model_name)
    return int(m.get_sentence_embedding_dimension())


_orig_memory_init = None


def install(auto_set_model: str | None = None) -> None:
    """Patch vstash.embed entry points in-place. Idempotent.

    If `auto_set_model` is given (e.g. 'v5-lora:<adapter_out>'), also
    patches `vstash.Memory.__init__` so every freshly-constructed Memory
    has its store meta `embedding_model` set to that value. This makes
    it possible to route a script like pipeline_runner.py through
    v5-lora without modifying its source: just install() with the
    sentinel name, then run the original main().
    """
    _ve.embed_texts = _shim_embed_texts
    _ve.embed_query = _shim_embed_query
    _ve.get_embedding_dim = _shim_get_dim

    if auto_set_model is not None:
        # Global force: any embed call (regardless of what model_name
        # the caller passed -- including vstash's own config resolution
        # path for queries) routes through this model. Necessary because
        # vstash's query path reads `cfg.embeddings.model` from its own
        # config, bypassing the per-Memory store meta we set below.
        global _force_model
        _force_model = auto_set_model

        import vstash as _vs
        global _orig_memory_init
        if _orig_memory_init is None:
            _orig_memory_init = _vs.Memory.__init__

        def patched_init(self, *args, **kwargs):
            _orig_memory_init(self, *args, **kwargs)
            try:
                self._store.set_meta("embedding_model", auto_set_model)
            except Exception as exc:  # noqa: BLE001
                # Surface clearly if vstash internals drift; never swallow.
                raise RuntimeError(
                    f"vstash_v5lora_shim: failed to set embedding_model "
                    f"meta on Memory._store: {exc}"
                ) from exc

        _vs.Memory.__init__ = patched_init


def uninstall() -> None:
    _ve.embed_texts = _orig_embed_texts
    _ve.embed_query = _orig_embed_query
    _ve.get_embedding_dim = _orig_get_dim
    global _orig_memory_init, _force_model
    if _orig_memory_init is not None:
        import vstash as _vs
        _vs.Memory.__init__ = _orig_memory_init
        _orig_memory_init = None
    _force_model = None
