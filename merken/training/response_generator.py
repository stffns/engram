"""Generate small-model responses to clinical case prompts.

The small model (Gemma 4 E4B in the spec, but configurable) attempts
each case WITHOUT seeing the source protocol. Its response is what
the aligner compares against truth -- divergences become midloop
intervention training labels.

Mock-friendly: ``ResponseGenerator`` takes a ``GenerateFn``
callable. Default implementation wraps HuggingFace transformers
but is lazy-imported so the module is testable without a model on
disk.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from merken.training.case_generator import GeneratedCase

# prompt -> raw response string.
GenerateFn = Callable[[str], str]


@dataclass
class CaseWithResponse:
    """Case + small-model response, ready for the aligner."""
    case_id: str
    prompt: str
    truth: str
    model_response: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ResponseGenerator:
    """Apply a GenerateFn to each GeneratedCase.

    Stateless aside from the generate_fn reference. Construct once
    per pipeline run; iterate over cases.
    """

    def __init__(self, generate_fn: GenerateFn) -> None:
        self._gen = generate_fn

    def respond(self, case: GeneratedCase) -> CaseWithResponse:
        response = self._gen(case.prompt)
        return CaseWithResponse(
            case_id=case.case_id,
            prompt=case.prompt,
            truth=case.truth,
            model_response=response.strip(),
            metadata=dict(case.metadata),
        )

    def respond_many(
        self, cases: Iterable[GeneratedCase],
    ) -> list[CaseWithResponse]:
        return [self.respond(c) for c in cases]


def default_hf_client(
    model_name: str = "google/gemma-3-1b-it",
    *,
    device: str = "auto",
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.0,
) -> GenerateFn:
    """Build a GenerateFn backed by HuggingFace transformers.

    Lazy-imports transformers + torch. First load is slow (model
    download to ~/.cache/huggingface). Subsequent calls reuse the
    in-process model.

    ``do_sample=False`` (greedy) is the default because we want
    deterministic responses for a given prompt -- the midloop
    training data is more useful when reproducible. If you need
    diversity, pass ``do_sample=True, temperature=0.7``.

    The default model ``google/gemma-3-1b-it`` is conservative; the
    spec mentions Gemma 4 E4B but that model identifier shifts
    across HF releases. Override via ``--model`` on the CLI.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device if device != "auto" else "auto",
    )
    model.eval()

    def _fn(prompt: str) -> str:
        # Use the model's chat template if available so instruction-
        # tuned models behave correctly.
        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            messages = [{"role": "user", "content": prompt}]
            input_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            input_text = prompt

        inputs = tok(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tok.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
            out = model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs.input_ids.shape[-1]:]
        return tok.decode(new_tokens, skip_special_tokens=True)

    return _fn
