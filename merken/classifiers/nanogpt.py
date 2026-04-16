"""nanoGPT-based event classifier for merken write decisions.

Loads a trained nanoGPT checkpoint and classifies events as NOISE or
DECISION (signal). Used as a WriteDecider to filter noise at ingestion
time, keeping the memory store clean for retrieval.

The model was trained on character-level event classification:
  <|event|>text<|label|>NOISE<|end|>
  <|event|>text<|label|>DECISION:topic:version<|end|>

At inference, we feed up to <|label|> and check if the model generates
'N' (NOISE) or 'D' (DECISION) as the first character. This single
character is enough for the write/skip decision -- we don't need the
full topic:version classification.

Requires: torch, a trained checkpoint, and the meta.pkl vocab file.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merken.policies import Decision, Event, WriteContext, WriteDecider


@dataclass
class NanoGPTConfig:
    """Mirrors nanoGPT's GPTConfig for inference only."""
    block_size: int = 256
    vocab_size: int = 81
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True


def _load_model(checkpoint_path: Path):
    """Load a nanoGPT checkpoint for inference."""
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    import math

    # Minimal model definition (inference only, no training code)
    class LayerNorm(nn.Module):
        def __init__(self, ndim, bias):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(ndim))
            self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

        def forward(self, input):
            return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

    class CausalSelfAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.attn_dropout = nn.Dropout(config.dropout)
            self.resid_dropout = nn.Dropout(config.dropout)
            self.n_head = config.n_head
            self.n_embd = config.n_embd
            self.dropout = config.dropout
            self.flash = hasattr(F, "scaled_dot_product_attention")

        def forward(self, x):
            B, T, C = x.size()
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            if self.flash:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=0, is_causal=True)
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(
                    torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T) == 0,
                    float("-inf"),
                )
                att = F.softmax(att, dim=-1)
                y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.resid_dropout(self.c_proj(y))

    class MLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

    class Block(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
            self.mlp = MLP(config)

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

    class GPT(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.transformer = nn.ModuleDict(dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            ))
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight

        def forward(self, idx):
            b, t = idx.size()
            pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
            x = self.transformer.drop(
                self.transformer.wte(idx) + self.transformer.wpe(pos)
            )
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)
            return self.lm_head(x[:, [-1], :])

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = NanoGPTConfig(**{
        k: v for k, v in ckpt["model_args"].items()
        if k in NanoGPTConfig.__dataclass_fields__
    })
    model = GPT(config)
    state_dict = ckpt["model"]
    prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            state_dict[k[len(prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, config


class NanoGPTWriteDecider:
    """WriteDecider that uses a trained nanoGPT to filter noise.

    Classifies events by generating the first character after <|label|>.
    If the model predicts 'D' (DECISION), the event is written.
    If the model predicts 'N' (NOISE) or anything else, it's skipped.
    """

    name = "nanogpt-classifier"

    def __init__(
        self,
        checkpoint_path: str | Path,
        meta_path: str | Path,
        *,
        confidence_threshold: float = 0.6,
    ) -> None:
        import torch
        self._torch = torch

        self._model, self._config = _load_model(Path(checkpoint_path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self._stoi: dict[str, int] = meta["stoi"]
        self._itos: dict[int, str] = meta["itos"]
        self._threshold = confidence_threshold

        # Find token ids for 'D' and 'N'
        self._d_id = self._stoi.get("D")
        self._n_id = self._stoi.get("N")

    def decide(self, event: Event, ctx: WriteContext) -> Decision:
        prompt = f"<|event|>{event.text}<|label|>"
        ids = [self._stoi.get(c, 0) for c in prompt]

        # Truncate to block_size
        if len(ids) > self._config.block_size:
            ids = ids[-self._config.block_size:]

        x = self._torch.tensor([ids], dtype=self._torch.long)

        with self._torch.no_grad():
            logits = self._model(x)

        probs = self._torch.nn.functional.softmax(logits[0, -1, :], dim=-1)

        p_decision = float(probs[self._d_id]) if self._d_id is not None else 0.0
        p_noise = float(probs[self._n_id]) if self._n_id is not None else 0.0

        is_signal = p_decision > p_noise and p_decision >= self._threshold

        return Decision(
            write=is_signal,
            reason=f"P(D)={p_decision:.3f} P(N)={p_noise:.3f}",
            confidence=max(p_decision, p_noise),
            policy=self.name,
        )
