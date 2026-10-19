# TinyInfer benchmark results

Environment: torch 2.11.0+cu128, python 3.13.15, gpu Tesla T4, cuda 12.8, transformers 5.16.1

Median of repeats. Throughput = generated tokens / wall time. TTFT and TPOT are measured in-process for TinyInfer (TTFT includes queueing). Ratios are vs Hugging Face static batching at the same concurrency.

| System | Concurrency | Output tok/s | Req/s | x vs HF static | x vs HF sequential | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms | Peak mem GiB | Preemptions | Repeats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hf-sequential | 1 | 35.5 | 0.20 | 1.00 | 1.00 | – | – | – | – | 2.16 | 0 | 3 |
| hf-static | 1 | 35.4 | 0.20 | 1.00 | 1.00 | – | – | – | – | 2.16 | 0 | 3 |
| tinyinfer | 1 | 35.6 | 0.20 | 1.01 | 1.00 | 34 | 63 | 28.0 | 28.7 | 12.96 | 0 | 3 |
| tinyinfer-int8 | 1 | 15.4 | 0.09 | 0.44 | 0.43 | 99 | 127 | 64.6 | 65.6 | 13.23 | 0 | 3 |
| hf-static | 4 | 109.4 | 0.61 | 1.00 | 3.09 | – | – | – | – | 2.20 | 0 | 3 |
| tinyinfer | 4 | 119.8 | 0.66 | 1.09 | 3.38 | 37 | 205 | 30.0 | 31.3 | 12.96 | 0 | 3 |
| tinyinfer-int8 | 4 | 49.6 | 0.28 | 0.45 | 1.40 | 100 | 415 | 73.1 | 75.9 | 13.23 | 0 | 3 |
| hf-static | 16 | 222.8 | 1.35 | 1.00 | 6.28 | – | – | – | – | 2.61 | 0 | 3 |
| tinyinfer | 16 | 282.2 | 1.71 | 1.27 | 7.95 | 53 | 850 | 51.5 | 57.5 | 13.03 | 0 | 3 |
| tinyinfer-int8 | 16 | 133.3 | 0.81 | 0.60 | 3.76 | 108 | 1,699 | 105.9 | 116.0 | 13.23 | 0 | 3 |
| hf-static | 32 | 233.8 | 1.49 | 1.00 | 6.59 | – | – | – | – | 3.15 | 0 | 3 |
| tinyinfer | 32 | 319.9 | 2.03 | 1.37 | 9.02 | 54 | 1,699 | 94.2 | 106.2 | 13.14 | 0 | 3 |
| tinyinfer-int8 | 32 | 187.7 | 1.19 | 0.80 | 5.29 | 107 | 3,396 | 154.4 | 173.3 | 13.23 | 0 | 3 |

## Headline (concurrency 16, fp16)

- Throughput: **1.27x** Hugging Face static batching (282.2 tok/s)
- TTFT p99: **850 ms**
