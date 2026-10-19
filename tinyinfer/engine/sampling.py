"""Token sampling: greedy, temperature, top-k, top-p (nucleus) and stop tokens.

All randomness goes through a seeded torch.Generator so tests are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SamplingParams:
    """Decoding settings for one request. temperature == 0 means greedy."""

    temperature: float = 0.0
    top_k: int = 0  # 0 disables top-k
    top_p: float = 1.0  # 1.0 disables nucleus filtering
    max_tokens: int = 64
    stop_token_ids: list[int] = field(default_factory=list)
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0 or self.max_tokens < 1:
            raise ValueError("top_k must be >= 0 and max_tokens >= 1")


def filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Set logits outside the top-k / nucleus to -inf ([..., vocab] in, same out).

    top-p keeps the smallest set of tokens whose probability sums to >= top_p;
    the most likely token is always kept.
    """
    logits = logits.clone()
    if top_k > 0 and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits[logits < kth] = float("-inf")
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(-1)
        # Drop a token if the mass of the tokens ranked above it already reaches top_p.
        mass_before = probs.cumsum(-1) - probs
        drop_sorted = mass_before >= top_p
        drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)
        logits[drop] = float("-inf")
    return logits


def sample(
    logits: torch.Tensor, params: SamplingParams, generator: torch.Generator | None = None
) -> int:
    """Pick the next token id from 1-D logits [vocab]."""
    if params.temperature == 0:
        return int(logits.argmax())
    logits = filter_logits(logits.float() / params.temperature, params.top_k, params.top_p)
    probs = logits.softmax(-1)
    return int(torch.multinomial(probs, 1, generator=generator))
