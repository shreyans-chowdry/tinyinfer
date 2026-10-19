"""LLMEngine: single-request generate() (M2/M3) and continuous batching (M4).

Generation is split into prefill (the whole prompt in one forward pass,
compute-bound) and decode (one token per step, memory-bound).

generate() runs one request at a time and is the simple reference path.
add_request() / step() / has_unfinished() run many requests together: every
step, finished sequences leave the batch and new ones join (paged=True only).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count

import torch

from tinyinfer.cache.contiguous import ContiguousKVCache
from tinyinfer.cache.paged import PagedKVCache
from tinyinfer.engine.runner import ModelRunner
from tinyinfer.engine.sampling import SamplingParams, sample
from tinyinfer.engine.scheduler import Scheduler
from tinyinfer.engine.sequence import Sequence, Status
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
        max_num_batched_tokens: int = 2048,
        max_num_seqs: int = 64,
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
            self.scheduler = Scheduler(
                self.paged_cache.blocks, max_num_batched_tokens, max_num_seqs
            )
            self.runner = ModelRunner(self.model, self.paged_cache)
        self._ids = count(1)  # seq id 0 is reserved for the single-request path
        self.sequences: dict[int, Sequence] = {}

    # ---- continuous batching (M4) -------------------------------------------------

    def add_request(self, prompt_ids: list[int], params: SamplingParams | None = None) -> int:
        """Queue a request; returns its sequence id. Requires paged=True."""
        if not self.paged:
            raise RuntimeError("continuous batching needs LLMEngine(..., paged=True)")
        params = params or SamplingParams()
        total = len(prompt_ids) + params.max_tokens
        if total > self.model.cfg.max_position_embeddings:
            raise ValueError("prompt + max_tokens exceeds the model's context length")
        if self.paged_cache.blocks.blocks_needed(total) > self.paged_cache.blocks.num_blocks:
            raise ValueError("request can never fit in the KV cache")
        if len(prompt_ids) > self.scheduler.max_num_batched_tokens:
            raise ValueError("prompt longer than max_num_batched_tokens")
        seq = Sequence(next(self._ids), list(prompt_ids), params)
        if params.seed is not None:
            seq.generator = torch.Generator(device=self.device).manual_seed(params.seed)
        self.sequences[seq.seq_id] = seq
        self.scheduler.add(seq)
        return seq.seq_id

    def has_unfinished(self) -> bool:
        return self.paged and self.scheduler.has_unfinished()

    def abort(self, seq_id: int) -> bool:
        """Cancel a request (e.g. client disconnected) and free its blocks at once."""
        seq = self.sequences.get(seq_id)
        if seq is None or seq.status is Status.FINISHED:
            return False
        if seq.status is Status.WAITING:
            self.scheduler.waiting.remove(seq)
            seq.finish("abort")
        else:
            self.scheduler.finish(seq, "abort")
        return True

    def _accept(self, seq: Sequence, logits: torch.Tensor, events: list) -> None:
        token = sample(logits, seq.params, seq.generator)
        if token in set(seq.params.stop_token_ids) | {self.eos}:
            self.scheduler.finish(seq, "stop")
            events.append(StepEvent(seq.seq_id, None, True))
            return
        seq.append(token)
        done = len(seq.output_ids) >= seq.params.max_tokens
        if done:
            self.scheduler.finish(seq, "length")
        events.append(StepEvent(seq.seq_id, token, done))

    def step(self) -> list[StepEvent]:
        """Run one scheduler iteration: prefill admitted sequences, decode the rest."""
        plan = self.scheduler.schedule()
        events: list[StepEvent] = []
        for seq in plan.prefills:
            self._accept(seq, self.runner.prefill(seq), events)
        if plan.decodes:
            logits = self.runner.decode(plan.decodes)
            for row, seq in enumerate(plan.decodes):
                self._accept(seq, logits[row], events)
        return events

    def generate_many(
        self, prompts: list[list[int]], params: SamplingParams | list[SamplingParams] | None = None
    ) -> list[list[int]]:
        """Run all prompts through continuous batching; return outputs in input order."""
        if not isinstance(params, list):
            params = [params] * len(prompts)
        ids = [self.add_request(p, sp) for p, sp in zip(prompts, params, strict=True)]
        while self.has_unfinished():
            self.step()
        return [self.sequences[i].output_ids for i in ids]

    # ---- single request (M2/M3 reference path) ------------------------------------

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


@dataclass
class StepEvent:
    """A token produced (or a stop) for one sequence during step()."""

    seq_id: int
    token: int | None
    finished: bool
