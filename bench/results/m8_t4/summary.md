# TinyInfer M8 on a Tesla T4 (Kaggle, torch 2.11.0+cu128)

Copied from the Kaggle run log (version 2 of the notebook, 21 Oct 2026). The
notebook's `tinyinfer_results.zip` output also holds the raw jsonl and plots.

Median of repeats. Throughput = generated tokens / wall time. TTFT and TPOT are measured in-process for TinyInfer (TTFT includes queueing). Ratios are vs Hugging Face static batching at the same concurrency.

| System | Concurrency | Output tok/s | Req/s | x vs HF static | x vs HF sequential | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms | Peak mem GiB | Preemptions | Repeats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hf-sequential | 1 | 35.9 | 0.20 | 1.00 | 1.00 | – | – | – | – | 2.16 | 0 | 3 |
| hf-static | 1 | 35.9 | 0.20 | 1.00 | 1.00 | – | – | – | – | 2.16 | 0 | 3 |
| tinyinfer | 1 | 40.6 | 0.23 | 1.13 | 1.13 | 34 | 60 | 24.5 | 25.1 | 13.00 | 0 | 3 |
| tinyinfer-int8 | 1 | 15.5 | 0.09 | 0.43 | 0.43 | 97 | 120 | 64.2 | 64.6 | 13.22 | 0 | 3 |
| hf-static | 4 | 110.1 | 0.61 | 1.00 | 3.07 | – | – | – | – | 2.20 | 0 | 3 |
| tinyinfer | 4 | 138.3 | 0.77 | 1.26 | 3.86 | 33 | 274 | 25.7 | 26.1 | 13.10 | 0 | 3 |
| tinyinfer-int8 | 4 | 53.8 | 0.30 | 0.49 | 1.50 | 98 | 328 | 66.5 | 67.7 | 13.23 | 0 | 3 |
| hf-static | 16 | 221.8 | 1.34 | 1.00 | 6.19 | – | – | – | – | 2.61 | 0 | 3 |
| tinyinfer | 16 | 455.1 | 2.75 | 2.05 | 12.69 | 42 | 990 | 29.1 | 35.3 | 13.10 | 0 | 3 |
| tinyinfer-int8 | 16 | 175.5 | 1.06 | 0.79 | 4.89 | 105 | 1,305 | 77.0 | 87.2 | 13.23 | 0 | 3 |
| hf-static | 32 | 234.0 | 1.49 | 1.00 | 6.53 | – | – | – | – | 3.15 | 0 | 3 |
| tinyinfer | 32 | 754.4 | 4.79 | 3.22 | 21.04 | 49 | 1,959 | 35.0 | 46.3 | 13.10 | 0 | 3 |
| tinyinfer-int8 | 32 | 291.3 | 1.85 | 1.24 | 8.12 | 105 | 2,564 | 93.7 | 106.5 | 13.23 | 0 | 3 |

## Headline (concurrency 16, fp16)

- Throughput: **2.05x** Hugging Face static batching (455.1 tok/s)
- TTFT p99: **990 ms**
