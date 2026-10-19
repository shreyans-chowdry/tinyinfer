"""Weight memory of TinyLlama in fp16 vs int8, and what the saving buys in KV blocks.

    python bench/memory.py --model-dir ~/.cache/tinyllama [--device cuda]

Bytes are counted from the tensors actually held by the model (element size x
count). On CUDA it also reports torch.cuda.max_memory_allocated after loading.
"""

from __future__ import annotations

import argparse
import gc
import json

import torch

from tinyinfer.cache.paged import bytes_per_block
from tinyinfer.model.loader import load_model
from tinyinfer.quant.int8 import quantize_model, weight_bytes


def measure(model_dir: str, device: str, skip: tuple[str, ...] | None) -> dict:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = load_model(model_dir, device="cpu", dtype=torch.float16)
    if skip is not None:
        quantize_model(model, skip=skip)
    model = model.to(device)
    out = {"weight_bytes": weight_bytes(model)}
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        out["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
        out["cuda_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
    del model
    gc.collect()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="bench/results/memory.json")
    args = ap.parse_args()

    from tinyinfer.model.config import LlamaConfig

    cfg = LlamaConfig.from_json(f"{args.model_dir}/config.json")
    block = bytes_per_block(cfg, 16, torch.float16)
    rows = {
        "fp16": measure(args.model_dir, args.device, None),
        "int8 (lm_head fp16)": measure(args.model_dir, args.device, ("lm_head",)),
        "int8 (all linears)": measure(args.model_dir, args.device, ()),
    }
    base = rows["fp16"]["weight_bytes"]
    for r in rows.values():
        r["weight_gib"] = round(r["weight_bytes"] / 2**30, 3)
        r["reduction_pct"] = round(100 * (1 - r["weight_bytes"] / base), 1)
        r["extra_kv_blocks_vs_fp16"] = (base - r["weight_bytes"]) // block
        r["extra_kv_tokens_vs_fp16"] = r["extra_kv_blocks_vs_fp16"] * 16
    result = {"device": args.device, "kv_block_bytes_fp16": block, "rows": rows}
    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
