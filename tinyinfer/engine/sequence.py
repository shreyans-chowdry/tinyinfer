"""Per-request state for the continuous-batching engine (milestone M4)."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

import torch

from tinyinfer.engine.sampling import SamplingParams


class Status(enum.Enum):
    WAITING = "waiting"  # queued, holds no KV blocks
    RUNNING = "running"  # in the batch, holds KV blocks
    FINISHED = "finished"


@dataclass
class Sequence:
    """One request: its tokens, sampling state and timestamps.

    Invariant while RUNNING: every token in all_ids except the most recently
    sampled one is stored in the KV cache. A preempted sequence loses its
    blocks and is recomputed from all_ids when it is admitted again.
    """

    seq_id: int
    prompt_ids: list[int]
    params: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    status: Status = Status.WAITING
    finish_reason: str | None = None
    preemptions: int = 0
    generator: torch.Generator | None = None
    arrival_time: float = field(default_factory=time.perf_counter)
    first_token_time: float | None = None
    token_times: list[float] = field(default_factory=list)
    finish_time: float | None = None

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    def append(self, token: int) -> None:
        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now
        self.token_times.append(now)
        self.output_ids.append(token)

    def finish(self, reason: str) -> None:
        self.status = Status.FINISHED
        self.finish_reason = reason
        self.finish_time = time.perf_counter()
