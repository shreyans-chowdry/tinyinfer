"""LLMEngine: single-request generation with a KV cache (milestone M2).

Generation is split into prefill (the whole prompt in one forward pass,
compute-bound) and decode (one token per step, memory-bound). M4 replaces this
loop with a continuous-batching scheduler; generate() stays as the simple path
that every optimisation is tested against.
"""

from __future__ import annotations

import torch

from tinyinfer.cache.contiguous import ContiguousKVCache
from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.engine.sampling import SamplingParams, sample
from tinyinfer.model.llama import Llama


class LLMEngine:
    """Wraps a model and runs prefill + decode for one prompt at a time."""

    def __init__(
        self,
        model: Llama,
        eos_token_id: int | None = None,
        paged: bool = False,
        block_size: int = 16,
        num_blocks: int | None = None,
    ) -> None:
        """paged=True stores KV in fixed-size blocks (M3) instead of one buffer per request."""
        self.model = model.eval()
        self.eos = model.cfg.eos_token_id if eos_token_id is None else eos_token_id
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.paged = paged
        self.paged_cache = None
        if paged:
            n = num_blocks or -(-model.cfg.max_position_embeddings // block_size)
            self.paged_cache = PagedKVCache(model.cfg, n, block_size, self.device, self.dtype)

    def _new_cache(self, prompt_len: int, max_len: int):
        """Return (cache, reserve_next_slot) for one request."""
        if not self.paged:
            cache = ContiguousKVCache(self.model.cfg, 1, max_len, self.device, self.dtype)
            return cache, lambda: None
        cache = self.paged_cache
        cache.blocks.allocate(0, prompt_len)
        cache.bind([0])
        return cache, lambda: cache.blocks.append_slot(0)

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], params: SamplingParams | None = None) -> list[int]:
        """Return generated token ids (stop/EOS token excluded) for one prompt."""
        params = params or SamplingParams()
        n = len(prompt_ids)
        max_len = n + params.max_tokens
        if max_len > self.model.cfg.max_position_embeddings:
            raise ValueError("prompt + max_tokens exceeds the model's context length")
        gen = None
        if params.seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(params.seed)
        cache, reserve_slot = self._new_cache(n, max_len)
        stops = set(params.stop_token_ids) | {self.eos}
        out: list[int] = []
        try:
            ids = torch.tensor([prompt_ids], device=self.device)
            positions = torch.arange(n, device=self.device).unsqueeze(0)
            logits = self.model(ids, positions, cache)[0, -1]  # prefill
            for step in range(params.max_tokens):
                token = sample(logits, params, gen)
                if token in stops:
                    break
                out.append(token)
                if step == params.max_tokens - 1:
                    break
                reserve_slot()
                pos = torch.tensor([[n + step]], device=self.device)
                tok = torch.tensor([[token]], device=self.device)
                logits = self.model(tok, pos, cache)[0, -1]
        finally:
            if self.paged:
                self.paged_cache.blocks.free(0)
        return out

    @torch.no_grad()
    def generate_uncached(self, prompt_ids: list[int], params: SamplingParams) -> list[int]:
        """Reference path: recompute the whole sequence every step (quadratic work).

        Used only by tests to prove the cached path produces identical tokens.
        """
        gen = None
        if params.seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(params.seed)
        stops = set(params.stop_token_ids) | {self.eos}
        seq = list(prompt_ids)
        out: list[int] = []
        for _ in range(params.max_tokens):
            logits = self.model(torch.tensor([seq], device=self.device))[0, -1]
            token = sample(logits, params, gen)
            if token in stops:
                break
            out.append(token)
            seq.append(token)
        return out
