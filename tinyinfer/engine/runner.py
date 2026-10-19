"""Builds model inputs for one scheduler step and runs the forward passes."""

from __future__ import annotations

import torch

from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.engine.sequence import Sequence
from tinyinfer.model.llama import Llama


class ModelRunner:
    """Prefill one sequence at a time; decode all running sequences in one call."""

    def __init__(self, model: Llama, cache: PagedKVCache) -> None:
        self.model = model
        self.cache = cache
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def prefill(self, seq: Sequence) -> torch.Tensor:
        """Process all of seq's tokens; return logits [vocab] for the next token."""
        ids = torch.tensor([seq.all_ids], device=self.device)
        pos = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
        self.cache.bind([seq.seq_id])
        return self.model(ids, pos, self.cache)[0, -1]

    @torch.no_grad()
    def decode(self, seqs: list[Sequence]) -> torch.Tensor:
        """Feed each sequence's last sampled token; return logits [B, vocab].

        The scheduler has already reserved the slot, so each token's position
        is its sequence's length minus one.
        """
        ids = torch.tensor([[s.all_ids[-1]] for s in seqs], device=self.device)
        pos = torch.tensor(
            [[self.cache.blocks.lens[s.seq_id] - 1] for s in seqs], device=self.device
        )
        self.cache.bind([s.seq_id for s in seqs])
        return self.model(ids, pos, self.cache)[:, -1]
