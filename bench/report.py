"""Summarise bench/results/raw.jsonl into summary.md and plots (milestone M7).

    python bench/report.py --raw bench/results/raw.jsonl --out bench/results

For each system and concurrency, metrics are computed per repeat and the
median across repeats is reported. Ratios are computed against hf-static at
the same concurrency (and hf-sequential where noted).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def pct(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; None if there are no values."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = max(0, min(len(vals) - 1, math.ceil(q / 100 * len(vals)) - 1))
    return vals[k]


def per_repeat(row: dict) -> dict:
    recs = row["records"]
    tokens = sum(r["tokens"] for r in recs)
    ttft = [r["ttft_s"] for r in recs]
    tpot = [r["tpot_s"] for r in recs]
    return {
        "tok_s": tokens / row["wall_s"],
        "req_s": len(recs) / row["wall_s"],
        "ttft_p50_ms": _ms(pct(ttft, 50)),
        "ttft_p99_ms": _ms(pct(ttft, 99)),
        "tpot_p50_ms": _ms(pct(tpot, 50)),
        "tpot_p99_ms": _ms(pct(tpot, 99)),
        "peak_mem_gib": row["peak_mem_bytes"] / 2**30,
        "preemptions": sum(r.get("preemptions", 0) for r in recs),
        "requests": len(recs),
    }


def _ms(v: float | None) -> float | None:
    return None if v is None else v * 1000


def summarise(rows: list[dict]) -> tuple[dict, dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    env = {}
    for row in rows:
        groups[(row["system"], row["concurrency"])].append(per_repeat(row))
        env = row.get("env", env)
    table = {}
    for key, reps in groups.items():
        med = {}
        for metric in reps[0]:
            vals = [r[metric] for r in reps if r[metric] is not None]
            med[metric] = statistics.median(vals) if vals else None
        med["repeats"] = len(reps)
        table[key] = med
    for (_system, conc), m in table.items():
        base = table.get(("hf-static", conc))
        m["x_vs_hf_static"] = m["tok_s"] / base["tok_s"] if base else None
        seq = table.get(("hf-sequential", 1))
        m["x_vs_hf_sequential"] = m["tok_s"] / seq["tok_s"] if seq else None
    return table, env


def fmt(v, digits=1) -> str:
    return "–" if v is None else f"{v:,.{digits}f}"


def write_markdown(table: dict, env: dict, path: Path) -> str:
    lines = ["# TinyInfer benchmark results", ""]
    lines.append("Environment: " + ", ".join(f"{k} {v}" for k, v in env.items()))
    lines += ["", "Median of repeats. Throughput = generated tokens / wall time. "
              "TTFT and TPOT are measured in-process for TinyInfer (TTFT includes queueing). "
              "Ratios are vs Hugging Face static batching at the same concurrency.", ""]
    lines.append("| System | Concurrency | Output tok/s | Req/s | x vs HF static | "
                 "x vs HF sequential | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms "
                 "| Peak mem GiB | Preemptions | Repeats |")
    lines.append("|" + "---|" * 13)
    for (system, conc), m in sorted(table.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        lines.append(
            f"| {system} | {conc} | {fmt(m['tok_s'])} | {fmt(m['req_s'], 2)} | "
            f"{fmt(m['x_vs_hf_static'], 2)} | {fmt(m['x_vs_hf_sequential'], 2)} | "
            f"{fmt(m['ttft_p50_ms'], 0)} | {fmt(m['ttft_p99_ms'], 0)} | "
            f"{fmt(m['tpot_p50_ms'], 1)} | {fmt(m['tpot_p99_ms'], 1)} | "
            f"{fmt(m['peak_mem_gib'], 2)} | {fmt(m['preemptions'], 0)} | {m['repeats']} |"
        )
    head = table.get(("tinyinfer", 16))
    if head and head.get("x_vs_hf_static"):
        lines += ["", "## Headline (concurrency 16, fp16)", "",
                  f"- Throughput: **{head['x_vs_hf_static']:.2f}x** Hugging Face static batching "
                  f"({fmt(head['tok_s'])} tok/s)",
                  f"- TTFT p99: **{fmt(head['ttft_p99_ms'], 0)} ms**"]
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def plot(table: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems = sorted({s for s, _ in table})
    for metric, label, fname in (("tok_s", "Output tokens/s", "throughput.png"),
                                 ("ttft_p99_ms", "TTFT p99 (ms)", "ttft_p99.png")):
        fig, ax = plt.subplots(figsize=(6, 4))
        for s in systems:
            pts = sorted((c, m[metric]) for (sy, c), m in table.items()
                         if sy == s and m[metric] is not None)
            if len(pts) > 1:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=s)
            elif pts:
                ax.scatter([pts[0][0]], [pts[0][1]], label=s)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Concurrent requests")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="bench/results/raw.jsonl")
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.raw) if line.strip()]
    table, env = summarise(rows)
    out = Path(args.out)
    print(write_markdown(table, env, out / "summary.md"))
    flat = [{"system": s, "concurrency": c, **m} for (s, c), m in table.items()]
    (out / "summary.json").write_text(json.dumps(flat, indent=2))
    if not args.no_plots:
        plot(table, out)


if __name__ == "__main__":
    main()
