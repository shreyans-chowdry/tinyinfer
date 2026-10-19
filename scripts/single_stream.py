"""Single-request speed: prefill latency and decode throughput (milestone M2).

    python scripts/single_stream.py --model-dir ~/.cache/tinyllama [--device cuda]

Prints prefill ms for 128- and 512-token prompts and decode tokens/s, as the
median of --repeats runs after one warm-up. Writes a JSON line to --out.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import torch

from tinyinfer.cache.contiguous import ContiguousKVCache
from tinyinfer.model.loader import load_model


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def prefill_ms(model, n: int, device: str) -> float:
    ids = torch.randint(100, 30000, (1, n), device=device)
    cache = ContiguousKVCache(model.cfg, 1, n, device, next(model.parameters()).dtype)
    sync(device)
    t = time.perf_counter()
    model(ids, torch.arange(n, device=device).unsqueeze(0), cache)
    sync(device)
    return (time.perf_counter() - t) * 1000


@torch.no_grad()
def decode_tps(model, prompt: int, steps: int, device: str) -> float:
    dtype = next(model.parameters()).dtype
    cache = ContiguousKVCache(model.cfg, 1, prompt + steps, device, dtype)
    ids = torch.randint(100, 30000, (1, prompt), device=device)
    logits = model(ids, torch.arange(prompt, device=device).unsqueeze(0), cache)
    tok = logits[:, -1:].argmax(-1)
    sync(device)
    t = time.perf_counter()
    for i in range(steps):
        logits = model(tok, torch.tensor([[prompt + i]], device=device), cache)
        tok = logits[:, -1:].argmax(-1)
    sync(device)
    return steps / (time.perf_counter() - t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--out", default="bench/results/single_stream.jsonl")
    args = ap.parse_args()

    model = load_model(args.model_dir, device=args.device)
    prefill_ms(model, 32, args.device)  # warm-up
    decode_tps(model, 16, 4, args.device)
    on_gpu = args.device.startswith("cuda")
    result = {
        "device": torch.cuda.get_device_name() if on_gpu else platform.processor() or "cpu",
        "dtype": str(next(model.parameters()).dtype),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
    }
    for n in (128, 512):
        result[f"prefill_{n}_ms"] = round(
            statistics.median(prefill_ms(model, n, args.device) for _ in range(args.repeats)), 1
        )
    result["decode_tokens_per_s"] = round(
        statistics.median(
            decode_tps(model, 128, args.decode_steps, args.device) for _ in range(args.repeats)
        ),
        2,
    )
    print(json.dumps(result))
    with open(args.out, "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
