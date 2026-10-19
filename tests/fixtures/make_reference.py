"""Build tests/fixtures/reference.pt from Hugging Face transformers.

For 5 fixed prompts (8 to ~300 tokens) this saves the input ids, the logits at
the last position and 32 greedy tokens. The tests compare our engine against
these numbers, so transformers is never needed at runtime.

Usage (fp32 on CPU needs ~5 GB RAM; fp16 on a T4 also works):
    python tests/fixtures/make_reference.py --model-dir ~/.cache/tinyllama --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Explain in one paragraph why the sky appears blue during the day.",
    (
        "Q: A train leaves at 3pm travelling 60 km/h. Another leaves at 4pm at 90 km/h on "
        "the same track. When does the second catch the first?\nA:"
    ),
    ("Paged attention stores the key-value cache in fixed-size blocks. " * 24).strip(),
]


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(Path(__file__).with_name("reference.pt")))
    ap.add_argument("--new-tokens", type=int, default=32)
    args = ap.parse_args()

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype)
    model = model.to(args.device).eval()

    cases = []
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
        with torch.no_grad():
            last = model(ids).logits[0, -1].float().cpu().clone()  # clone: drop the full logits
            gen = model.generate(ids, max_new_tokens=args.new_tokens, do_sample=False)
        cases.append({
            "prompt": prompt,
            "input_ids": ids[0].cpu(),
            "last_logits": last,
            "greedy_tokens": gen[0, ids.shape[1]:].cpu(),
        })
        print(f"{ids.shape[1]:>4} prompt tokens -> {tok.decode(cases[-1]['greedy_tokens'])[:60]!r}")

    torch.save({"model_dir": str(args.model_dir), "dtype": str(dtype), "cases": cases}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
