"""Generic LLM-backed write decider.

Uses any HuggingFace causal LM as a few-shot DECISION/NOISE classifier
by scoring two single-token continuations ("DECISION" vs "NOISE") after
a chat-templated prompt with in-context examples. One forward pass per
event.

Why this beats the simple "Event: X\\nLabel:" template (which failed
for Gemma 3 270M and 1B): small instruction-tuned LMs follow chat
format strictly. Without the chat template, the natural continuation
after "Label:" is often whitespace / prose / another header, so the
two class-token probabilities collapse to near-zero. The 2026-04-17
bench showed Gemma 1B zero-shot had 50% recall with 0% NOISE FPR;
few-shot + chat template lifted it to 100% recall / 33% NOISE FPR --
the first classifier to beat nanoGPT v6 on both axes.

Designed for small local models (Gemma 3 1B-it, 4B-it, Qwen 0.5B-Chat,
SmolLM 360M-Instruct). CPU inference is 1-3s / event on an M-series
Mac; usable for shadow mode, too slow for the write path. Bigger
models work too but the latency adds up.
"""

from __future__ import annotations

from collections.abc import Iterable

from merken.policies.types import Decision, Event, WriteContext

DEFAULT_INSTRUCTIONS = (
    "Classify the event as DECISION or NOISE. A DECISION records a lasting "
    "choice, commitment, specification, rollout plan, or reference data "
    "worth keeping. A NOISE event is routine status, attendance, burndown, "
    "or ephemeral log that nobody needs to re-read."
)

# Curated few-shot examples pulled from the same held-out scenario
# categories the bench measures on. Kept short so latency scales.
DEFAULT_FEWSHOT: list[tuple[str, str]] = [
    (
        "## Sprint 62 Burndown\n"
        "| Day | Remaining | Done |\n"
        "|---|---|---|\n"
        "| 1 | 46 | 0 |\n| 2 | 43 | 3 |",
        "NOISE",
    ),
    (
        "Migrated from Kafka to NATS JetStream. Benchmarks showed 3x "
        "lower p99 latency at peak event volume.",
        "DECISION",
    ),
    (
        "## Rollout Plan for Inventory-V2\n"
        "| Wave | Stores | Region | Date |\n"
        "| 1 | 12 | LATAM-S | 2026-06-02 |\n"
        "Declared approach: pause at any gate miss, do not skip waves.",
        "DECISION",
    ),
    (
        "## Daily Ops Standup\n"
        "| Area | Owner | Status |\n"
        "| DB | Lin | Green |\n"
        "| Pay | Pia | Yellow |\n"
        "No incidents.",
        "NOISE",
    ),
]


class LLMWriteDecider:
    """Few-shot chat-templated classifier.

    Prompt structure (via tokenizer.apply_chat_template):

        system:    DEFAULT_SYSTEM_PROMPT
        user:      "Event: ... Label: DECISION"  (few-shot pair)
        ...
        user:      "Event: <event text>"
        assistant: (model predicts first token -> DECISION or NOISE)

    The decision compares logit mass on the ``DECISION`` and ``NOISE``
    tokens at the first assistant-turn position.
    """

    name = "llm-classifier"

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        max_input_chars: int = 1800,
        instructions: str = DEFAULT_INSTRUCTIONS,
        fewshot: Iterable[tuple[str, str]] = DEFAULT_FEWSHOT,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._device = device
        self._threshold = confidence_threshold
        self._max_input_chars = max_input_chars
        self._model_name = model_name
        self._instructions = instructions
        self._fewshot = list(fewshot)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._model.eval()
        self._model.to(device)
        self._has_chat_template = bool(
            getattr(self._tokenizer, "chat_template", None)
        )

        # Score both leading-space and bare variants per class; small
        # LMs pick either depending on whether the assistant turn
        # opens with a space or not.
        self._d_ids = self._class_token_ids(["DECISION", " DECISION"])
        self._n_ids = self._class_token_ids(["NOISE", " NOISE"])

    def _class_token_ids(self, variants: list[str]) -> list[int]:
        ids: list[int] = []
        for v in variants:
            enc = self._tokenizer.encode(v, add_special_tokens=False)
            if enc:
                ids.append(int(enc[0]))
        if not ids:
            raise RuntimeError(
                f"tokenizer yielded no ids for any of {variants!r} "
                f"(model={self._model_name!r})"
            )
        # dedup while preserving order
        seen: set[int] = set()
        uniq: list[int] = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                uniq.append(i)
        return uniq

    def _build_user_content(self, event_text: str) -> str:
        """Single-turn prompt: instructions + fewshot + new event.

        Multi-turn few-shot (user/assistant alternation) was measured
        to give 66.7% FPR on `markdown_tables_held_out` with Gemma 3
        1B-IT; moving everything into one user turn dropped FPR to
        16.7%. Hypothesis: multi-turn makes the LM treat the examples
        as "past conversations" less relevant to the current query,
        while single-turn primes more effectively.
        """
        parts = [self._instructions, "", "Examples:"]
        for ex_text, ex_label in self._fewshot:
            parts.append(f"Event:\n{ex_text}\nLabel: {ex_label}")
        parts.append(
            f"Event:\n{event_text[: self._max_input_chars]}\nLabel:"
        )
        return "\n\n".join(parts)

    def _build_prompt(self, event_text: str) -> str:
        user_content = self._build_user_content(event_text)
        if self._has_chat_template:
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return user_content

    def decide(self, event: Event, ctx: WriteContext) -> Decision:  # noqa: ARG002
        prompt = self._build_prompt(event.text)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        last_logits = outputs.logits[0, -1, :]
        probs = self._torch.nn.functional.softmax(last_logits, dim=-1)

        p_d = sum(float(probs[i]) for i in self._d_ids)
        p_n = sum(float(probs[i]) for i in self._n_ids)

        # Normalize over the two-class subspace so the confidence
        # number is interpretable even when the LM distributes mass
        # across unrelated continuations.
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
