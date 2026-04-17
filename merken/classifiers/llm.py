"""Generic LLM-backed write decider.

Uses any HuggingFace causal LM as a zero-shot DECISION/NOISE classifier
by scoring two single-token continuations ("DECISION" vs "NOISE") after
a short prompt. One forward pass per event.

Designed for small local models (Gemma 3 270M-instruct, Qwen 0.5B,
SmolLM 360M) where CPU inference is a few hundred ms per event. Bigger
models work too but the write-path latency adds up.

Why this instead of ``NanoGPTWriteDecider``: a pretrained LM has seen
markdown tables, code blocks, prose, transcripts in its training
corpus. It gets the format-sensitivity for free, without the
3-session-iteration-on-synthetic-data treadmill that the custom
800K-param classifier went through. The trade-off is size and latency.
"""

from __future__ import annotations

from merken.policies.types import Decision, Event, WriteContext

_PROMPT_TEMPLATE = (
    "You classify memory events as DECISION or NOISE.\n"
    "- DECISION: a lasting choice, design commitment, resolved open question, "
    "or reference material worth keeping.\n"
    "- NOISE: routine status, attendance, transient update, or ephemeral log "
    "with no lasting value.\n\n"
    "Event:\n{text}\n\n"
    "Label:"
)


class LLMWriteDecider:
    """Score two single-token continuations after a classification prompt.

    The model runs once per event; we compare the logit of the first
    token of " DECISION" vs " NOISE" at the label position. This is
    cheaper and more stable than generating an answer and parsing it.
    """

    name = "llm-classifier"

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        max_input_chars: int = 3000,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._device = device
        self._threshold = confidence_threshold
        self._max_input_chars = max_input_chars
        self._model_name = model_name

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._model.eval()
        self._model.to(device)

        # Find the token ids for the first subword of " DECISION" and
        # " NOISE". Leading space matters -- after "Label:" the natural
        # continuation starts with a space token in most SentencePiece
        # and BPE tokenizers.
        self._d_id = self._first_token(" DECISION")
        self._n_id = self._first_token(" NOISE")

    def _first_token(self, text: str) -> int:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise RuntimeError(
                f"tokenizer produced no ids for {text!r} "
                f"(model={self._model_name!r})"
            )
        return int(ids[0])

    def decide(self, event: Event, ctx: WriteContext) -> Decision:  # noqa: ARG002
        text = event.text[: self._max_input_chars]
        prompt = _PROMPT_TEMPLATE.format(text=text)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        last_logits = outputs.logits[0, -1, :]
        probs = self._torch.nn.functional.softmax(last_logits, dim=-1)

        p_d = float(probs[self._d_id])
        p_n = float(probs[self._n_id])

        # Normalize over the two-choice subspace so the confidence
        # number is interpretable even when the LM distributes mass
        # across other continuations.
        total = p_d + p_n
        if total > 0:
            rel_p_d = p_d / total
            rel_p_n = p_n / total
        else:
            rel_p_d = rel_p_n = 0.5

        is_signal = rel_p_d > rel_p_n and rel_p_d >= self._threshold

        return Decision(
            write=is_signal,
            reason=(
                f"P(D)={rel_p_d:.3f} P(N)={rel_p_n:.3f} "
                f"raw_mass={total:.3f}"
            ),
            confidence=max(rel_p_d, rel_p_n),
            policy=self.name,
        )
