"""Paged KV storage and attention reads through block tables (milestone M3).

One pre-allocated tensor per layer, shaped [num_blocks, 2, kv_heads, block_size,
head_dim] (index 0 = keys, 1 = values). Before a forward pass the engine
reserves slots in the BlockManager and binds the batch's sequence ids; update()
then writes each row's new K/V into its slots and gathers that row's blocks
into a contiguous view for attention.

The gather copies blocks, which costs memory bandwidth; M8 can replace it with
a kernel that reads blocks in place. Correctness first.
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
        shape = (num_blocks, 2, cfg.num_key_value_heads, block_size, cfg.head_dim)
        self.kv = [
            torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)
        ]
        self.device = torch.device(device)
        self.seq_ids: list[int] = []

    def bind(self, seq_ids: list[int]) -> None:
        """Set which sequence each batch row belongs to for the next forward pass."""
        self.seq_ids = list(seq_ids)

    def _table(self, seq_id: int) -> torch.Tensor:
        return torch.tensor(self.blocks.tables[seq_id], device=self.device)

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Write k, v ([B, kv_heads, T, D]) at positions [B, T]; return padded K, V, key positions.

        Returned K, V are [B, kv_heads, S, D] with S the longest sequence in the
        batch; key positions [B, S] are PAD_POSITION past each row's length so
        the causal mask hides the padding.
        """
        if len(self.seq_ids) != k.shape[0]:
            raise ValueError("bind() the batch's sequence ids before the forward pass")
        kv, bs = self.kv[layer], self.block_size
        gathered, lengths = [], []
        for row, seq in enumerate(self.seq_ids):
            n = self.blocks.lens[seq]
            pos = positions[row]
            if int(pos.max()) >= n:
                raise IndexError(f"sequence {seq}: write at {int(pos.max())} but only {n} slots")
            table = self._table(seq)
            blk, off = table[pos // bs], pos % bs
            # Advanced indices around a slice put the token axis first: [T, H, D].
            kv[blk, 0, :, off] = k[row].transpose(0, 1)
            kv[blk, 1, :, off] = v[row].transpose(0, 1)
            blocks = kv[table]  # [nb, 2, H, bs, D]
            flat = blocks.permute(1, 2, 0, 3, 4).reshape(2, k.shape[1], -1, k.shape[3])
            gathered.append(flat[:, :, :n])
            lengths.append(n)
        s = max(lengths)
        b, h, _, d = k.shape
        keys = k.new_zeros(b, h, s, d)
        values = k.new_zeros(b, h, s, d)
        key_pos = torch.full((b, s), PAD_POSITION, dtype=torch.long, device=k.device)
        for row, (g, n) in enumerate(zip(gathered, lengths, strict=True)):
            keys[row, :, :n], values[row, :, :n] = g[0], g[1]
            key_pos[row, :n] = torch.arange(n, device=k.device)
        return keys, values, key_pos
