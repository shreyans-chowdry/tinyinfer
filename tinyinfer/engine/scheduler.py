"""Iteration-level scheduling (continuous batching, as in Orca and vLLM).

Every step the scheduler:
  1. reserves one KV slot for each RUNNING sequence's next token; if blocks run
     out it preempts the most recently admitted sequences (frees their blocks,
     puts them back at the front of the queue to be recomputed later);
  2. if nothing was preempted, admits WAITING sequences in FIFO order while
     their tokens fit in free blocks and the padded prefill batch
     (rows x longest prompt) fits in the per-step token budget;
  3. returns which sequences to prefill and which to decode this step.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from tinyinfer.cache.block_manager import BlockManager
from tinyinfer.engine.sequence import Sequence, Status


@dataclass
class SchedulerOutput:
    prefills: list[Sequence] = field(default_factory=list)
    decodes: list[Sequence] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)


class Scheduler:
    """Decides which sequences run each step, limited by KV blocks and a token budget."""

    def __init__(
        self,
        blocks: BlockManager,
        max_num_batched_tokens: int = 2048,
        max_num_seqs: int = 64,
    ) -> None:
        self.blocks = blocks
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []  # in admission order
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def _preempt(self, seq: Sequence) -> None:
        self.blocks.free(seq.seq_id)
        seq.status = Status.WAITING
        seq.preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        # 1. One more slot for every running sequence, preempting newest-first.
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            while not self.blocks.can_append(seq.seq_id):
                victim = self.running.pop()
                self._preempt(victim)
                out.preempted.append(victim)
                if victim is seq:
                    break
            else:
                self.blocks.append_slot(seq.seq_id)
                out.decodes.append(seq)
                i += 1
                continue
            break  # seq preempted itself; it was the last running sequence
        if out.preempted:
            return out  # memory is tight: do not admit anything this step
        # 2. Admit waiting sequences (prefill) in FIFO order. All admitted prompts run
        #    in one padded forward pass, so the budget counts padded tokens:
        #    rows x longest prompt.
        width = 0
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            n = len(seq.all_ids)
            cost = (len(out.prefills) + 1) * max(width, n)
            if cost > self.max_num_batched_tokens or not self.blocks.can_allocate(n + 1):
                break
            self.waiting.popleft()
            self.blocks.allocate(seq.seq_id, n)
            seq.status = Status.RUNNING
            self.running.append(seq)
            out.prefills.append(seq)
            width = max(width, n)
        return out

    def finish(self, seq: Sequence, reason: str) -> None:
        """Remove a finished sequence from the batch and free its blocks."""
        self.running.remove(seq)
        self.blocks.free(seq.seq_id)
        seq.finish(reason)
