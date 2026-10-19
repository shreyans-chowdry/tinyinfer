"""Block allocator for the paged KV cache (the PagedAttention idea, Kwon et al. 2023).

KV memory is split into fixed-size blocks of block_size tokens. Each sequence
owns a block table: the list of physical block ids holding its tokens in order.
Token at position p lives in block table[p // block_size], slot p % block_size.
A sequence wastes at most one partly filled block, instead of reserving memory
for its maximum length up front.

Reference counts are kept per block so prefix caching (M8) can share blocks.
"""

from __future__ import annotations

from collections import deque


class OutOfBlocks(RuntimeError):
    """Raised when an allocation needs more free blocks than exist."""


class BlockManager:
    """Free list, per-sequence block tables and token counts."""

    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks: deque[int] = deque(range(num_blocks))
        self.ref_counts = [0] * num_blocks
        self.tables: dict[int, list[int]] = {}
        self.lens: dict[int, int] = {}

    @property
    def num_free(self) -> int:
        """Blocks not owned by any sequence."""
        return len(self.free_blocks)

    def blocks_needed(self, n_tokens: int) -> int:
        """Blocks required to hold n_tokens (ceiling division)."""
        return -(-n_tokens // self.block_size)

    def can_allocate(self, n_tokens: int) -> bool:
        """True if a new sequence of n_tokens fits in the free blocks."""
        return self.blocks_needed(n_tokens) <= self.num_free

    def allocate(self, seq_id: int, n_tokens: int) -> None:
        """Reserve blocks for a new sequence's first n_tokens (its prompt)."""
        if seq_id in self.tables:
            raise ValueError(f"sequence {seq_id} already has blocks")
        need = self.blocks_needed(n_tokens)
        if need > self.num_free:
            raise OutOfBlocks(f"need {need} blocks, {self.num_free} free")
        blocks = [self.free_blocks.popleft() for _ in range(need)]
        for b in blocks:
            self.ref_counts[b] = 1
        self.tables[seq_id] = blocks
        self.lens[seq_id] = n_tokens

    def can_append(self, seq_id: int) -> bool:
        """True if one more token fits (room in the last block, or a free block)."""
        return self.lens[seq_id] % self.block_size != 0 or self.num_free > 0

    def append_slot(self, seq_id: int) -> int:
        """Reserve the slot for the sequence's next token; return its position."""
        pos = self.lens[seq_id]
        if pos % self.block_size == 0:  # last block is full (or there is none)
            if not self.free_blocks:
                raise OutOfBlocks(f"no free block for sequence {seq_id}")
            b = self.free_blocks.popleft()
            self.ref_counts[b] = 1
            self.tables[seq_id].append(b)
        self.lens[seq_id] = pos + 1
        return pos

    def free(self, seq_id: int) -> None:
        """Release a sequence's blocks; blocks return to the free list at ref count 0."""
        for b in self.tables.pop(seq_id):
            self.ref_counts[b] -= 1
            if self.ref_counts[b] == 0:
                self.free_blocks.append(b)
        del self.lens[seq_id]

    def slot(self, seq_id: int, pos: int) -> tuple[int, int]:
        """(physical block, offset) holding the token at position pos."""
        if pos >= self.lens[seq_id]:
            raise IndexError(f"position {pos} not allocated for sequence {seq_id}")
        return self.tables[seq_id][pos // self.block_size], pos % self.block_size

    def used_blocks(self) -> int:
        """Blocks currently owned by sequences."""
        return self.num_blocks - self.num_free
