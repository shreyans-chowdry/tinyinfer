"""Sequential generate() vs continuous batching on the same requests (milestone M4).

    python scripts/batching_speedup.py --model-dir ~/.cache/tinyllama --requests 8

Checks that both paths produce identical greedy outputs and prints the
throughput ratio. A development check; the full benchmark is M7.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch

from tinyinfer.engine.engine import LLMEngine
from tinyinfer.engine.sampling import SamplingParams
from tinyinfer.model.loader import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    model = load_model(args.model_dir, device=args.device)
    rng = random.Random(0)
    prompts = [
        [rng.randrange(100, 30000) for _ in range(args.prompt_len)] for _ in range(args.requests)
    ]
    params = SamplingParams(max_tokens=args.max_tokens)
    total = args.requests * args.max_tokens

    warm = LLMEngine(model, eos_token_id=-1, paged=True, num_blocks=512)
    warm.generate_many(prompts[:1], SamplingParams(max_tokens=2))

    single = LLMEngine(model, eos_token_id=-1, paged=True, num_blocks=512)
    t = time.perf_counter()
    sequential = [single.generate(p, params) for p in prompts]
    t_seq = time.perf_counter() - t

    batched_engine = LLMEngine(model, eos_token_id=-1, paged=True, num_blocks=512)
    t = time.perf_counter()
    batched = batched_engine.generate_many(prompts, params)
    t_bat = time.perf_counter() - t

    print(json.dumps({
        "device": args.device,
        "requests": args.requests,
        "sequential_tok_s": round(total / t_seq, 2),
        "batched_tok_s": round(total / t_bat, 2),
        "speedup": round(t_seq / t_bat, 2),
        "identical_outputs": sequential == batched,
    }))


if __name__ == "__main__":
    main()
