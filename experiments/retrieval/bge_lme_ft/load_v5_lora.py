"""Loader for v5-ft bge LoRA adapter.

Reconstitutes a SentenceTransformer whose underlying HF transformer is
wrapped by the trained peft adapter, so encode(...) includes the
fine-tuned deltas.

Usage:

    from experiments.retrieval.bge_lme_ft.load_v5_lora import load_v5_lora
    model = load_v5_lora("experiments/retrieval/bge_lme_ft/adapters/v5-lora")
    emb = model.encode(["a sentence"])

The loader reads `load_hint.json` produced by train_lora.py to find the
canonical base model id (not the HF cache path embedded in peft's
adapter_config). This keeps adapters portable across machines and HF
cache locations.
"""

from __future__ import annotations

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


def load_v5_lora(adapter_out: str | Path, device: str | None = None) -> SentenceTransformer:
    from peft import PeftModel

    adapter_out = Path(adapter_out)
    hint_path = adapter_out / "load_hint.json"
    if not hint_path.exists():
        raise FileNotFoundError(
            f"missing {hint_path}. Did train_lora.py finish cleanly?"
        )
    hint = json.loads(hint_path.read_text())
    base = hint["base_model"]
    adapter_dir = adapter_out / hint["adapter_subdir"]
    if not adapter_dir.exists():
        raise FileNotFoundError(f"missing adapter dir {adapter_dir}")

    model = SentenceTransformer(base, device=device)
    if "max_seq_length" in hint:
        model.max_seq_length = int(hint["max_seq_length"])
    # See train_lora.py::_apply_lora for the `.model` vs `.auto_model` note.
    transformer = model[0]
    hf_model = transformer.model
    transformer.model = PeftModel.from_pretrained(hf_model, str(adapter_dir))
    return model
