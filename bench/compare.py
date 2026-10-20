"""Compare two benchmark summaries (e.g. M7 baseline vs the current code).

    python bench/compare.py bench/results/m7_baseline/summary.json bench/results/summary.json

Prints a markdown table of output tokens/s, TTFT p99 and TPOT p50 for every
TinyInfer row present in both runs, plus the ratio to Hugging Face static
batching measured in the same run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict[tuple[str, int], dict]:
    return {(r["system"], r["concurrency"]): r for r in json.loads(Path(path).read_text())}


def fmt(v, digits: int = 1) -> str:
    return "–" if v is None else f"{v:,.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--out", default=None, help="also write the table to this file")
    args = ap.parse_args()
    a, b = load(args.before), load(args.after)
    lines = [
        "| System | Concurrency | tok/s before | tok/s after | Speed-up | x HF static before "
        "| x HF static after | TTFT p99 ms before | TTFT p99 ms after | TPOT p50 ms before "
        "| TPOT p50 ms after |",
        "|" + "---|" * 11,
    ]
    for key in sorted(set(a) & set(b), key=lambda k: (k[0], k[1])):
        if not key[0].startswith("tinyinfer"):
            continue
        x, y = a[key], b[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {fmt(x['tok_s'])} | {fmt(y['tok_s'])} | "
            f"{y['tok_s'] / x['tok_s']:.2f}x | {fmt(x.get('x_vs_hf_static'), 2)} | "
            f"{fmt(y.get('x_vs_hf_static'), 2)} | {fmt(x.get('ttft_p99_ms'), 0)} | "
            f"{fmt(y.get('ttft_p99_ms'), 0)} | {fmt(x.get('tpot_p50_ms'))} | "
            f"{fmt(y.get('tpot_p50_ms'))} |"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
