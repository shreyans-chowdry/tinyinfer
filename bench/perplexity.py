"""WikiText-2 perplexity of our forward pass, full precision vs int8 weights.

    python bench/perplexity.py --model-dir ~/.cache/tinyllama [--max-windows 16]

The test split is joined with blank lines, tokenised once, and cut into
non-overlapping windows of --window tokens. Perplexity = exp(mean negative
log-likelihood) over every predicted token. --max-windows limits the run for
CPU; leave it unset on a GPU to use the whole split. Variants: full precision,
int8 with lm_head kept in full precision, and int8 for every linear layer.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from tinyinfer.model.loader import load_model
from tinyinfer.quant.int8 import quantize_model
from tinyinfer.server.tokenizer import HFTokenizer


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, window: int, max_windows: int | None) -> tuple[float, int]:
    nll, count = 0.0, 0
    starts = range(0, ids.numel() - 1, window)
    if max_windows:
        starts = list(starts)[:max_windows]
    for s in starts:
        chunk = ids[s : s + window + 1]
        if chunk.numel() < 2:
            break
        inp, tgt = chunk[:-1][None], chunk[1:]
        logits = model(inp.to(next(model.parameters()).device))[0].float().cpu()
        nll += F.cross_entropy(logits, tgt, reduction="sum").item()
        count += tgt.numel()
    return math.exp(nll / count), count


def main() -> None:
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--out", default="bench/results/perplexity.json")
    args = ap.parse_args()

    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    text = "\n\n".join(rows)
    ids = torch.tensor(HFTokenizer(args.model_dir).encode(text))
    result = {"device": args.device, "window": args.window, "dataset_tokens": ids.numel()}
    variants = (("full", None), ("int8", ("lm_head",)), ("int8_all", ()))
    for name, skip in variants:
        model = load_model(args.model_dir, device=args.device)
        if skip is not None:
            quantize_model(model, skip=skip)
        t = time.perf_counter()
        ppl, n = perplexity(model, ids, args.window, args.max_windows)
        result[name] = {"perplexity": round(ppl, 4), "tokens": n,
                        "seconds": round(time.perf_counter() - t, 1)}
        print(name, result[name], flush=True)
        del model
    result["dtype"] = "fp16" if args.device.startswith("cuda") else "fp32"
    for name in ("int8", "int8_all"):
        inc = 100 * (result[name]["perplexity"] / result["full"]["perplexity"] - 1)
        result[name]["increase_pct"] = round(inc, 2)
        result[name]["within_2pct_budget"] = inc <= 2.0
    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
