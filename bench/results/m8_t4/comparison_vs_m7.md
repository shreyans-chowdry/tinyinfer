# M8 vs M7 on the same T4 protocol

| System | Concurrency | tok/s before | tok/s after | Speed-up | x HF static before | x HF static after | TTFT p99 ms before | TTFT p99 ms after | TPOT p50 ms before | TPOT p50 ms after |
|---|---|---|---|---|---|---|---|---|---|---|
| tinyinfer | 1 | 35.6 | 40.6 | 1.14x | 1.01 | 1.13 | 63 | 60 | 28.0 | 24.5 |
| tinyinfer | 4 | 119.8 | 138.3 | 1.15x | 1.09 | 1.26 | 205 | 274 | 30.0 | 25.7 |
| tinyinfer | 16 | 282.2 | 455.1 | 1.61x | 1.27 | 2.05 | 850 | 990 | 51.5 | 29.1 |
| tinyinfer | 32 | 319.9 | 754.4 | 2.36x | 1.37 | 3.22 | 1,699 | 1,959 | 94.2 | 35.0 |
| tinyinfer-int8 | 1 | 15.4 | 15.5 | 1.01x | 0.44 | 0.43 | 127 | 120 | 64.6 | 64.2 |
| tinyinfer-int8 | 4 | 49.6 | 53.8 | 1.09x | 0.45 | 0.49 | 415 | 328 | 73.1 | 66.5 |
| tinyinfer-int8 | 16 | 133.3 | 175.5 | 1.32x | 0.60 | 0.79 | 1,699 | 1,305 | 105.9 | 77.0 |
| tinyinfer-int8 | 32 | 187.7 | 291.3 | 1.55x | 0.80 | 1.24 | 3,396 | 2,564 | 154.4 | 93.7 |
