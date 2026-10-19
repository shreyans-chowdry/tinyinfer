"""LLMEngine: single-request generation with a KV cache (milestone M2).

Generation is split into prefill (the whole prompt in one forward pass,
compute-bound) and decode (one token per step, memory-bound). M4 replaces this
loop with a continuous-batching scheduler; generate() stays as the simple path
that every optimisation is tested against.
"""

from __future__ import annotations

import torch

from tinyinfer.cache.contiguous import ContiguousKVCache
from tinyinfer.engine.sampling import SamplingParams, sample
from tinyinfer.model.llama import Llama


class LLMEngine:
    """Wraps a model and runs prefill + decode for one prompt at a time."""

    def __init__(self, model: Llama, eos_token_id: int | None = None) -> None:
        self.model = model.eval()
        self.eos = model.cfg.eos_token_id if eos_token_id is None else eos_token_id
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype

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
        cache = ContiguousKVCache(self.model.cfg, 1, max_len, self.device, self.dtype)
        stops = set(params.stop_token_ids) | {self.eos}

        ids = torch.tensor([prompt_ids], device=self.device)
        positions = torch.arange(n, device=self.device).unsqueeze(0)
        logits = self.model(ids, positions, cache)[0, -1]  # prefill
        out: list[int] = []
        for step in range(params.max_tokens):
            token = sample(logits, params, gen)
            if token in stops:
                break
            out.append(token)
            if step == params.max_tokens - 1:
                break
            pos = torch.tensor([[n + step]], device=self.device)
            logits = self.model(torch.tensor([[token]], device=self.device), pos, cache)[0, -1]
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
