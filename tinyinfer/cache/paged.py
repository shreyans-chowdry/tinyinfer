"""Paged KV storage and attention reads through block tables (milestone M3).

One pre-allocated tensor per layer, shaped [num_blocks, block_size, 2,
kv_heads, head_dim] (index 0 of the third axis = keys, 1 = values), so a token
slot is one row of a [num_blocks * block_size, 2, kv_heads, head_dim] view.
Before a forward pass the engine reserves slots in the BlockManager and binds
the batch's sequence ids. At the first layer the cache builds a padded block
table [B, max_blocks] and the key positions once; every layer then does one
vectorised write and one gather for the whole batch.

The gather copies blocks, which costs memory bandwidth; a kernel that reads
blocks in place (M8) would avoid the copy. Correctness first.
"""

from __future__ import annotations

import torch

from tinyinfer.cache.block_manager import BlockManager
from tinyinfer.model.config import LlamaConfig

PAD_POSITION = torch.iinfo(torch.long).max  # padded keys are never attended to


def bytes_per_block(cfg: LlamaConfig, block_size: int, dtype: torch.dtype) -> int:
    """KV bytes one block occupies across all layers."""
    elem = torch.tensor([], dtype=dtype).element_size()
    per_layer = 2 * cfg.num_key_value_heads * block_size * cfg.head_dim * elem
    return cfg.num_hidden_layers * per_layer


def num_blocks_for_gpu(
    cfg: LlamaConfig, block_size: int, dtype: torch.dtype, utilization: float = 0.9
) -> int:
    """Blocks that fit in the GPU memory left after loading weights."""
    free, total = torch.cuda.mem_get_info()
    budget = free - (1 - utilization) * total
    return max(1, int(budget // bytes_per_block(cfg, block_size, dtype)))


class PagedKVCache:
    """Block-structured KV cache shared by every sequence in the engine."""

    def __init__(
        self,
        cfg: LlamaConfig,
        num_blocks: int,
        block_size: int = 16,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.cfg = cfg
        self.block_size = block_size
        self.blocks = BlockManager(num_blocks, block_size)
        shape = (num_blocks, block_size, 2, cfg.num_key_value_heads, cfg.head_dim)
        self.kv = [
            torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)
        ]
        self.device = torch.device(device)
        self.seq_ids: list[int] = []
        self.has_padding = False
        self._tables: torch.Tensor | None = None
        self._key_pos: torch.Tensor | None = None

    def bind(self, seq_ids: list[int], has_padding: bool = False) -> None:
        """Set which sequence each batch row belongs to for the next forward pass.

        has_padding: some rows contain padding tokens (position -1) that must
        not be written to the cache (batched prefill of unequal prompts).
        """
        self.seq_ids = list(seq_ids)
        self.has_padding = has_padding

    def _prepare(self) -> None:
        """Build the padded block table and key positions for the bound batch."""
        lens = [self.blocks.lens[s] for s in self.seq_ids]
        tables = [self.blocks.tables[s] for s in self.seq_ids]
        width = max(len(t) for t in tables)
        padded = [t + [0] * (width - len(t)) for t in tables]  # padding is masked below
        self._tables = torch.tensor(padded, device=self.device)
        s = max(lens)
        pos = torch.arange(s, device=self.device).expand(len(lens), s)
        n = torch.tensor(lens, device=self.device).unsqueeze(1)
        self._key_pos = torch.where(pos < n, pos, PAD_POSITION)
        self._lens = lens

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Write k, v ([B, kv_heads, T, D]) at positions [B, T] (-1 = padding, not written).

        Returns padded K, V and key positions.

        Returned K, V are [B, kv_heads, S, D] with S the longest sequence in the
        batch; key positions [B, S] are PAD_POSITION past each row's length so
        the causal mask hides the padding.
        """
        if len(self.seq_ids) != k.shape[0]:
            raise ValueError("bind() the batch's sequence ids before the forward pass")
        if layer == 0:
            self._prepare()
            limit = torch.tensor(self._lens, device=positions.device)
            if bool((positions.max(dim=1).values >= limit).any()):
                raise IndexError("write position beyond the slots reserved for a sequence")
        b, h, t, d = k.shape
        bs = self.block_size
        kv = self.kv[layer]
        flat = kv.view(-1, 2, h, d)  # one row per token slot
        pos = positions.clamp(min=0)
        slots = torch.gather(self._tables, 1, pos // bs) * bs + pos % bs  # [B, T]
        new = torch.stack((k, v), dim=2).permute(0, 3, 2, 1, 4).reshape(b * t, 2, h, d)
        if self.has_padding:  # padding tokens would overwrite position 0: skip them
            keep = (positions >= 0).reshape(-1)
            flat[slots.reshape(-1)[keep]] = new[keep]
        else:
            flat[slots.reshape(-1)] = new
        s = self._key_pos.shape[1]
        gathered = kv[self._tables].reshape(b, -1, 2, h, d)[:, :s]  # [B, S, 2, H, D]
        keys = gathered[:, :, 0].transpose(1, 2)
        values = gathered[:, :, 1].transpose(1, 2)
        return keys, values, self._key_pos
