"""OpenAI-style completions API with Server-Sent Events streaming (milestone M5).

    python -m tinyinfer.server.api --model-dir ~/.cache/tinyllama --port 8000
    curl -N localhost:8000/v1/completions -H 'content-type: application/json' \\
         -d '{"prompt": "The capital of France is", "max_tokens": 16, "stream": true}'

The HTTP layer only moves tokens: all model work happens on the engine thread
(server/worker.py). A client that disconnects mid-stream has its request
aborted, which frees its KV blocks immediately.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tinyinfer.engine.engine import LLMEngine
from tinyinfer.engine.sampling import SamplingParams
from tinyinfer.server.tokenizer import IncrementalDetokenizer, Tokenizer
from tinyinfer.server.worker import EngineWorker


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = Field(16, ge=1)
    temperature: float = Field(1.0, ge=0)
    top_p: float = Field(1.0, gt=0, le=1)
    top_k: int = Field(0, ge=0)
    stop: str | list[str] | None = None
    stream: bool = False
    seed: int | None = None


def split_on_stop(text: str, stops: list[str]) -> tuple[str, bool, str]:
    """Return (safe_to_emit, hit_stop, held_back) for accumulated unsent text.

    Text before a stop string is emitted and the stop string is dropped. Without
    a hit, a suffix that could still grow into a stop string is held back.
    """
    hits = [i for i in (text.find(s) for s in stops) if i != -1]
    if hits:
        return text[: min(hits)], True, ""
    hold = 0
    for s in stops:
        for k in range(min(len(s) - 1, len(text)), 0, -1):
            if text.endswith(s[:k]):
                hold = max(hold, k)
                break
    return text[: len(text) - hold], False, text[len(text) - hold :]


def create_app(engine: LLMEngine, tokenizer: Tokenizer, model_name: str = "tinyllama") -> FastAPI:
    """Build the app around an engine created with paged=True."""
    worker = EngineWorker(engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        worker.shutdown()

    app = FastAPI(title="TinyInfer", lifespan=lifespan)
    app.state.worker = worker

    def chunk(cid: str, created: int, text: str, finish: str | None) -> str:
        body = {
            "id": cid,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{"text": text, "index": 0, "finish_reason": finish}],
        }
        return f"data: {json.dumps(body)}\n\n"

    async def run(req: CompletionRequest, usage: dict):
        """Yield (text_delta, finish_reason) pairs; aborts the request if abandoned."""
        prompt_ids = tokenizer.encode(req.prompt)
        usage["prompt_tokens"], usage["completion_tokens"] = len(prompt_ids), 0
        params = SamplingParams(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_tokens=req.max_tokens,
            seed=req.seed,
        )
        try:
            seq_id, q = await worker.submit(prompt_ids, params)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        stops = [req.stop] if isinstance(req.stop, str) else list(req.stop or [])
        detok = IncrementalDetokenizer(tokenizer, prompt_ids)
        pending, done = "", False
        try:
            while True:
                ev = await q.get()
                if ev.token is not None:
                    usage["completion_tokens"] += 1
                    pending += detok.push(ev.token)
                    emit, hit, pending = split_on_stop(pending, stops)
                    if hit:
                        worker.abort(seq_id)
                        done = True
                        yield emit, "stop"
                        return
                    if emit:
                        yield emit, None
                if ev.finished:
                    done = True
                    reason = {"length": "length", "abort": "abort"}.get(ev.reason, "stop")
                    yield pending, reason
                    return
        finally:
            if not done:  # client went away or the handler was cancelled
                worker.abort(seq_id)

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request):
        cid, created = f"cmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        usage: dict = {}
        gen = run(req, usage)
        if req.stream:
            first = await anext(gen)  # surface 400 errors before streaming starts

            async def sse():
                try:
                    text, finish = first
                    while True:
                        if await request.is_disconnected():
                            return
                        if text or finish:
                            yield chunk(cid, created, text, finish)
                        if finish:
                            break
                        text, finish = await anext(gen)
                    yield "data: [DONE]\n\n"
                finally:
                    await gen.aclose()

            return StreamingResponse(sse(), media_type="text/event-stream")
        parts, finish = [], None
        async for text, fin in gen:
            parts.append(text)
            finish = fin
        return {
            "id": cid,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{"text": "".join(parts), "index": 0, "finish_reason": finish}],
            "usage": {**usage, "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"]},
        }

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model_name}

    @app.get("/metrics")
    async def metrics():
        return worker.stats()

    return app


def main() -> None:
    import torch
    import uvicorn

    from tinyinfer.cache.paged import num_blocks_for_gpu
    from tinyinfer.model.loader import load_model
    from tinyinfer.server.tokenizer import HFTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-blocks", type=int, default=None, help="default: fill free GPU memory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    model = load_model(args.model_dir, device=args.device)
    dtype = next(model.parameters()).dtype
    if args.num_blocks:
        n = args.num_blocks
    elif args.device.startswith("cuda"):
        n = num_blocks_for_gpu(model.cfg, 16, dtype)
    else:
        n = 512
    engine = LLMEngine(model, paged=True, num_blocks=n)
    print(f"KV cache: {n} blocks of 16 tokens")
    uvicorn.run(create_app(engine, HFTokenizer(args.model_dir)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
