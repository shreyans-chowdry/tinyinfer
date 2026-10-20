"""Builds model inputs for one scheduler step and runs the forward passes."""

from __future__ import annotations

import torch

from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.engine.sequence import Sequence
from tinyinfer.model.llama import Llama


class ModelRunner:
    """One forward pass prefills all admitted sequences; one more decodes all running ones."""

    def __init__(self, model: Llama, cache: PagedKVCache) -> None:
        self.model = model
        self.cache = cache
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        """Process the prompts of all admitted sequences in one forward pass.

        Rows are right-padded to the longest prompt with position -1, which the
        cache skips and attention ignores. Returns next-token logits [B, vocab],
        computed only at each row's last real token.
        """
        lens = [len(s.all_ids) for s in seqs]
        width = max(lens)
        ids = torch.zeros(len(seqs), width, dtype=torch.long)
        pos = torch.full((len(seqs), width), -1, dtype=torch.long)
        for row, (seq, n) in enumerate(zip(seqs, lens, strict=True)):
            ids[row, :n] = torch.tensor(seq.all_ids)
            pos[row, :n] = torch.arange(n)
        self.cache.bind([s.seq_id for s in seqs], has_padding=min(lens) < width)
        last = torch.tensor([n - 1 for n in lens], device=self.device)
        logits = self.model(ids.to(self.device), pos.to(self.device), self.cache, last)
        return logits[:, 0]

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
