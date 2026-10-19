"""Runs the engine on one background thread and fans tokens out to asyncio streams.

The engine is not thread-safe, so every engine call (add, step, abort) happens
on the worker thread. HTTP handlers send commands through a queue and receive
tokens through per-request asyncio.Queues, filled with loop.call_soon_threadsafe.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass

from tinyinfer.engine.engine import LLMEngine
from tinyinfer.engine.sampling import SamplingParams


@dataclass
class TokenEvent:
    token: int | None
    finished: bool
    reason: str | None = None


class EngineWorker:
    """Owns the engine; loops step() while there is work."""

    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine
        self._cmds: queue.Queue = queue.Queue()
        self._streams: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._stop = threading.Event()
        self.tokens_total = 0
        self._recent: deque[tuple[float, int]] = deque()
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self._thread.start()

    # ---- called from the asyncio side ------------------------------------------
    async def submit(
        self, prompt_ids: list[int], params: SamplingParams
    ) -> tuple[int, asyncio.Queue]:
        """Queue a request; returns its sequence id and the queue its tokens arrive on."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        fut: Future = Future()
        self._cmds.put(("add", prompt_ids, params, loop, q, fut))
        seq_id = await asyncio.wrap_future(fut)
        return seq_id, q

    def abort(self, seq_id: int) -> None:
        self._cmds.put(("abort", seq_id))

    def stats(self) -> dict:
        sched = self.engine.scheduler
        now = time.perf_counter()
        window = [n for t, n in self._recent if now - t <= 10.0]
        return {
            "running": len(sched.running),
            "waiting": len(sched.waiting),
            "free_blocks": self.engine.paged_cache.blocks.num_free,
            "total_blocks": self.engine.paged_cache.blocks.num_blocks,
            "preemptions": sched.num_preemptions,
            "tokens_generated": self.tokens_total,
            "tokens_per_s_10s": round(sum(window) / 10.0, 2),
        }

    def shutdown(self) -> None:
        self._stop.set()
        self._cmds.put(("noop",))
        self._thread.join(timeout=5)

    # ---- worker thread ------------------------------------------------------------
    def _handle(self, cmd: tuple) -> None:
        if cmd[0] == "add":
            _, prompt_ids, params, loop, q, fut = cmd
            try:
                seq_id = self.engine.add_request(prompt_ids, params)
            except Exception as exc:  # report bad requests to the caller
                fut.set_exception(exc)
                return
            self._streams[seq_id] = (loop, q)
            fut.set_result(seq_id)
        elif cmd[0] == "abort":
            seq_id = cmd[1]
            if self.engine.abort(seq_id):
                self._emit(seq_id, TokenEvent(None, True, "abort"))

    def _emit(self, seq_id: int, event: TokenEvent) -> None:
        target = self._streams.get(seq_id)
        if target is None:
            return
        loop, q = target
        loop.call_soon_threadsafe(q.put_nowait, event)
        if event.finished:
            del self._streams[seq_id]

    def _run(self) -> None:
        while not self._stop.is_set():
            idle = not self.engine.has_unfinished()
            try:  # block while idle; otherwise just drain pending commands
                cmd = self._cmds.get(timeout=0.1) if idle else self._cmds.get_nowait()
                self._handle(cmd)
                while True:
                    self._handle(self._cmds.get_nowait())
            except queue.Empty:
                pass
            if not self.engine.has_unfinished():
                continue
            events = self.engine.step()
            produced = 0
            for ev in events:
                if ev.token is not None:
                    produced += 1
                reason = self.engine.sequences[ev.seq_id].finish_reason if ev.finished else None
                self._emit(ev.seq_id, TokenEvent(ev.token, ev.finished, reason))
            self.tokens_total += produced
            now = time.perf_counter()
            self._recent.append((now, produced))
            while self._recent and now - self._recent[0][0] > 10.0:
                self._recent.popleft()
