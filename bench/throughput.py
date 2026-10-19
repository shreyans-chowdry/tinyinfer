"""Throughput and latency: TinyInfer vs Hugging Face transformers (milestone M7).

    python bench/throughput.py --model-dir ~/.cache/tinyllama --concurrency 1 4 16 32

Every system serves the same requests: random token-id prompts of 64-512
tokens, each generating 64-256 tokens (seed 0). EOS is ignored everywhere so
every request produces exactly its max_tokens, and each system does the same
useful work.

Systems:
  tinyinfer       our engine, fp16, continuous batching, closed loop: C requests
                  are always in flight; a finished one is replaced at once.
  tinyinfer-int8  the same with int8 weights.
  hf-static       transformers generate(), batches of C left-padded prompts; a
                  batch runs until its longest request is done.
  hf-sequential   transformers generate(), one request at a time.

Latency for tinyinfer is measured in-process from add_request() to the first
token (TTFT, includes queueing) and between later tokens (TPOT).
Raw records go to --out as JSON lines; bench/report.py summarises them.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import random
import statistics
import time
from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class Request:
    prompt: list[int]
    max_tokens: int


def make_requests(n: int, seed: int = 0, vocab: tuple[int, int] = (100, 30000),
                  prompt_len: tuple[int, int] = (64, 512),
                  gen_len: tuple[int, int] = (64, 256)) -> list[Request]:
    rng = random.Random(seed)
    return [
        Request([rng.randrange(*vocab) for _ in range(rng.randint(*prompt_len))],
                rng.randint(*gen_len))
        for _ in range(n)
    ]


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_tinyinfer(engine, reqs: list[Request], concurrency: int) -> tuple[list[dict], float]:
    """Closed loop: keep `concurrency` requests in flight. Returns (records, wall seconds)."""
    from tinyinfer.engine.sampling import SamplingParams

    pending = deque(reqs)
    in_flight = 0
    records = []
    sync()
    t0 = time.perf_counter()
    while pending or engine.has_unfinished():
        while pending and in_flight < concurrency:
            r = pending.popleft()
            engine.add_request(r.prompt, SamplingParams(max_tokens=r.max_tokens))
            in_flight += 1
        for ev in engine.step():
            if ev.finished:
                in_flight -= 1
                seq = engine.sequences[ev.seq_id]
                gaps = [b - a for a, b in zip(seq.token_times, seq.token_times[1:], strict=False)]
                records.append({
                    "tokens": len(seq.output_ids),
                    "ttft_s": seq.first_token_time - seq.arrival_time,
                    "tpot_s": statistics.mean(gaps) if gaps else None,
                    "latency_s": seq.finish_time - seq.arrival_time,
                    "preemptions": seq.preemptions,
                })
    sync()
    return records, time.perf_counter() - t0


@torch.no_grad()
def run_hf(model, reqs: list[Request], batch: int, device: str) -> tuple[list[dict], float]:
    """Static batching with transformers generate(); batch=1 is sequential."""
    records = []
    sync()
    t0 = time.perf_counter()
    for i in range(0, len(reqs), batch):
        group = reqs[i : i + batch]
        width = max(len(r.prompt) for r in group)
        ids = torch.zeros(len(group), width, dtype=torch.long)
        mask = torch.zeros(len(group), width, dtype=torch.long)
        for row, r in enumerate(group):  # left padding, as generate() expects
            ids[row, width - len(r.prompt):] = torch.tensor(r.prompt)
            mask[row, width - len(r.prompt):] = 1
        new = max(r.max_tokens for r in group)
        start = time.perf_counter()
        model.generate(
            input_ids=ids.to(device), attention_mask=mask.to(device),
            max_new_tokens=new, min_new_tokens=new, do_sample=False, pad_token_id=0,
        )
        sync()
        end = time.perf_counter()
        for r in group:  # each request only needed its own max_tokens
            records.append({"tokens": r.max_tokens, "ttft_s": None, "tpot_s": None,
                            "latency_s": end - t0 if batch > 1 else end - start,
                            "preemptions": 0})
    sync()
    return records, time.perf_counter() - t0


def environment() -> dict:
    env = {"torch": torch.__version__, "python": platform.python_version()}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name()
        env["cuda"] = torch.version.cuda
    else:
        env["cpu"] = platform.processor() or platform.machine()
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    return env


def requests_for(concurrency: int, minimum: int) -> int:
    return max(minimum, 4 * concurrency)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 32])
    ap.add_argument("--systems", nargs="+",
                    default=["tinyinfer", "tinyinfer-int8", "hf-static", "hf-sequential"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--min-requests", type=int, default=16)
    ap.add_argument("--prompt-len", type=int, nargs=2, default=[64, 512])
    ap.add_argument("--gen-len", type=int, nargs=2, default=[64, 256])
    ap.add_argument("--num-blocks", type=int, default=None, help="default: fill GPU memory")
    ap.add_argument("--out", default="bench/results/raw.jsonl")
    args = ap.parse_args()

    env = environment()
    print(json.dumps(env))
    with open(f"{args.model_dir}/config.json") as f:
        vocab_size = json.load(f)["vocab_size"]
    vocab = (min(100, vocab_size // 10), min(30000, vocab_size))  # skip special-token ids

    def requests(n: int) -> list[Request]:
        return make_requests(n, seed=0, vocab=vocab, prompt_len=tuple(args.prompt_len),
                             gen_len=tuple(args.gen_len))
    out = open(args.out, "a")

    def emit(system: str, conc: int, rep: int, records: list[dict], wall: float, peak: int):
        row = {"system": system, "concurrency": conc, "repeat": rep, "wall_s": wall,
               "peak_mem_bytes": peak, "env": env, "records": records}
        out.write(json.dumps(row) + "\n")
        out.flush()
        toks = sum(r["tokens"] for r in records)
        print(f"{system:15s} C={conc:<3d} rep={rep} {toks / wall:8.1f} tok/s  "
              f"wall={wall:6.1f}s  peak={peak / 2**30:.2f} GiB", flush=True)

    def peak() -> int:
        return torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    def reset_peak() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    for system in args.systems:
        gc.collect()
        reset_peak()
        if system.startswith("tinyinfer"):
            from tinyinfer.cache.paged import num_blocks_for_gpu
            from tinyinfer.engine.engine import LLMEngine
            from tinyinfer.model.loader import load_model
            from tinyinfer.quant.int8 import quantize_model

            model = load_model(args.model_dir, device="cpu", dtype=torch.float16
                               if args.device.startswith("cuda") else torch.float32)
            if system == "tinyinfer-int8":
                quantize_model(model, skip=())
            model = model.to(args.device)
            dtype = next(model.parameters()).dtype
            n_blocks = args.num_blocks or (
                num_blocks_for_gpu(model.cfg, 16, dtype) if args.device.startswith("cuda")
                else 1024)
            print(f"{system}: {n_blocks} KV blocks", flush=True)
            for conc in args.concurrency:
                reqs = requests(requests_for(conc, args.min_requests))
                for rep in range(args.repeats + 1):  # repeat 0 is a warm-up, not reported
                    engine = LLMEngine(model, eos_token_id=-1, paged=True, num_blocks=n_blocks,
                                       max_num_seqs=max(64, conc))
                    reset_peak()
                    records, wall = run_tinyinfer(engine, reqs if rep else reqs[:2], conc)
                    if rep:
                        emit(system, conc, rep, records, wall, peak())
                    del engine
                    gc.collect()
            del model
        else:
            from transformers import AutoModelForCausalLM

            dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
            model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype)
            model = model.to(args.device).eval()
            levels = [1] if system == "hf-sequential" else args.concurrency
            for conc in levels:
                reqs = requests(requests_for(conc, args.min_requests))
                batch = 1 if system == "hf-sequential" else conc
                for rep in range(args.repeats + 1):
                    reset_peak()
                    records, wall = run_hf(model, reqs if rep else reqs[:2], batch, args.device)
                    if rep:
                        emit(system, conc, rep, records, wall, peak())
            del model
    out.close()


if __name__ == "__main__":
    main()

