"""M5 tests: OpenAI-style API, SSE streaming, stop strings, cancellation, concurrency.

A real uvicorn server runs on a free local port so streaming and client
disconnects behave as in production.
"""

import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
import torch
import uvicorn

from tinyinfer.engine.engine import LLMEngine
from tinyinfer.model.config import tiny_config
from tinyinfer.model.llama import Llama
from tinyinfer.server.api import create_app, split_on_stop
from tinyinfer.server.tokenizer import IncrementalDetokenizer


class CharTokenizer:
    """Maps printable ASCII to ids 0..94 and any id back to a printable char."""

    def encode(self, text: str) -> list[int]:
        return [(ord(c) - 32) % 95 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(32 + i % 95) for i in ids)


class ByteTokenizer:
    """One token per UTF-8 byte, so multi-byte characters span tokens."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode())

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def server():
    torch.manual_seed(0)
    engine = LLMEngine(Llama(tiny_config()).eval(), eos_token_id=-1, paged=True, num_blocks=256)
    app = create_app(engine, CharTokenizer(), model_name="tiny")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    while not srv.started:
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}", engine
    srv.should_exit = True
    thread.join(timeout=5)


def wait_idle(url: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        m = httpx.get(f"{url}/metrics").json()
        if m["running"] == 0 and m["waiting"] == 0 and m["free_blocks"] == m["total_blocks"]:
            return m
        time.sleep(0.05)
    raise AssertionError(f"engine did not go idle: {m}")


def sse_events(lines) -> list:
    return [line[len("data: "):] for line in lines if line.startswith("data: ")]


def test_health(server):
    url, _ = server
    assert httpx.get(f"{url}/health").json() == {"status": "ok", "model": "tiny"}


def test_non_streaming_completion(server):
    url, _ = server
    r = httpx.post(f"{url}/v1/completions", json={"prompt": "hello", "max_tokens": 12,
                                                    "temperature": 0}).json()
    assert r["object"] == "text_completion"
    assert r["choices"][0]["finish_reason"] == "length"
    assert len(r["choices"][0]["text"]) == 12
    assert r["usage"] == {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}


def test_streaming_matches_non_streaming_and_ends_with_done(server):
    url, _ = server
    body = {"prompt": "stream me", "max_tokens": 20, "temperature": 0}
    full = httpx.post(f"{url}/v1/completions", json=body).json()["choices"][0]["text"]
    with httpx.stream("POST", f"{url}/v1/completions", json={**body, "stream": True}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = sse_events(r.iter_lines())
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert len(chunks) > 1  # tokens arrived incrementally
    assert "".join(c["choices"][0]["text"] for c in chunks) == full
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])


def test_stop_string_truncates_and_frees(server):
    url, _ = server
    body = {"prompt": "abc", "max_tokens": 30, "temperature": 0}
    full = httpx.post(f"{url}/v1/completions", json=body).json()["choices"][0]["text"]
    stop = full[8:10]
    r = httpx.post(f"{url}/v1/completions", json={**body, "stop": [stop]}).json()
    assert r["choices"][0]["text"] == full[: full.find(stop)]
    assert r["choices"][0]["finish_reason"] == "stop"
    wait_idle(url)


def test_client_disconnect_cancels_request_and_frees_blocks(server):
    url, engine = server
    body = {"prompt": "long", "max_tokens": 400, "temperature": 0, "stream": True}
    with httpx.stream("POST", f"{url}/v1/completions", json=body) as r:
        lines = r.iter_lines()
        got = 0
        for line in lines:
            if line.startswith("data: "):
                got += 1
            if got == 3:
                break  # leave the context: the connection closes mid-stream
    m = wait_idle(url)
    aborted = [s for s in engine.sequences.values() if s.finish_reason == "abort"]
    assert aborted and len(aborted[-1].output_ids) < 400
    assert m["free_blocks"] == m["total_blocks"]


def test_32_concurrent_streams_complete(server):
    url, _ = server

    async def one(client: httpx.AsyncClient, i: int) -> tuple[str, str]:
        body = {"prompt": f"req {i}", "max_tokens": 16, "temperature": 0, "stream": True}
        async with client.stream("POST", f"{url}/v1/completions", json=body) as r:
            events = [line async for line in r.aiter_lines() if line.startswith("data: ")]
        assert events[-1] == "data: [DONE]"
        chunks = [json.loads(e[6:]) for e in events[:-1]]
        return "".join(c["choices"][0]["text"] for c in chunks), chunks[-1]["choices"][0][
            "finish_reason"
        ]

    async def main():
        async with httpx.AsyncClient(timeout=60) as client:
            return await asyncio.gather(*(one(client, i) for i in range(32)))

    results = asyncio.run(main())
    assert all(len(text) == 16 and fin == "length" for text, fin in results)
    for i in (0, 17, 31):  # batched outputs equal the same request alone
        alone = httpx.post(f"{url}/v1/completions", json={"prompt": f"req {i}", "max_tokens": 16,
                                                           "temperature": 0}).json()
        assert alone["choices"][0]["text"] == results[i][0]
    wait_idle(url)


def test_metrics_report_progress(server):
    url, _ = server
    m = httpx.get(f"{url}/metrics").json()
    assert m["tokens_generated"] > 0 and m["total_blocks"] == 256
    assert set(m) >= {"running", "waiting", "free_blocks", "preemptions", "tokens_per_s_10s"}


def test_request_that_cannot_fit_returns_400(server):
    url, _ = server
    r = httpx.post(f"{url}/v1/completions", json={"prompt": "x" * 600, "max_tokens": 10})
    assert r.status_code == 400


def test_incremental_detokenizer_holds_back_partial_utf8():
    detok = IncrementalDetokenizer(ByteTokenizer())
    out = [detok.push(b) for b in "aé€".encode()]
    assert out == ["a", "", "é", "", "", "€"]


class SpaceTokenizer:
    """Mimics SentencePiece: a word-start token decodes with no space at text start."""

    vocab = {0: "_The", 1: "_capital", 2: "_is", 3: "_Paris", 4: "."}

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids).replace("_", " ").lstrip(" ")


def test_incremental_detokenizer_keeps_leading_space_after_prompt():
    detok = IncrementalDetokenizer(SpaceTokenizer(), prompt_ids=[0, 1, 2])
    assert [detok.push(t) for t in (3, 4)] == [" Paris", "."]
    assert [IncrementalDetokenizer(SpaceTokenizer()).push(3)] == ["Paris"]  # no context


def test_split_on_stop_holds_back_possible_prefix():
    assert split_on_stop("hello wor", ["world"]) == ("hello ", False, "wor")
    assert split_on_stop("hello world!", ["world"]) == ("hello ", True, "")
    assert split_on_stop("plain", []) == ("plain", False, "")
