"""A simple preallocated KV cache for one or more sequences (milestone M2).

Each layer gets K and V tensors of shape [batch, kv_heads, max_len, head_dim].
The model writes new keys/values at explicit positions and attends over the
filled prefix. M3 replaces this with a paged cache; tests prove both agree.
"""

from __future__ import annotations

import torch

from tinyinfer.model.config import LlamaConfig


class ContiguousKVCache:
    """Per-layer K/V buffers reserved up front for max_len tokens."""

    def __init__(
        self,
        cfg: LlamaConfig,
        batch: int,
        max_len: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (batch, cfg.num_key_value_heads, max_len, cfg.head_dim)
        layers = range(cfg.num_hidden_layers)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in layers]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in layers]
        self.max_len = max_len
        self.length = 0  # number of filled positions (same for every sequence in M2)

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store k, v ([B, kv_heads, T, D]) at positions ([B, T]); return the filled prefix.

        Positions must be contiguous and identical across the batch in M2.
        """
        start, end = int(positions[0, 0]), int(positions[0, -1]) + 1
        if end > self.max_len:
            raise ValueError(f"cache full: need {end} positions, have {self.max_len}")
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v
        if layer == len(self.k) - 1:
            self.length = max(self.length, end)
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]
