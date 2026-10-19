# TinyInfer

An LLM inference engine written from scratch in PyTorch. It serves TinyLlama-1.1B and implements the core ideas of vLLM itself: a KV cache, paged attention, continuous batching, int8 weights and a streaming HTTP API. Each optimisation is measured against Hugging Face `transformers`.

## Status

| Milestone | What | Status |
|---|---|---|
| M0 | Setup, CI, tiny test config, Hugging Face reference fixtures | ✅ done |
| M1 | Llama forward pass from scratch (RMSNorm, RoPE, GQA, SwiGLU) | ✅ done |
| M2 | KV cache, sampling, single-request `generate()` | ✅ done |
| M3 | Paged KV cache (block manager + paged attention) | ✅ done |
| M4 | Continuous batching engine with preemption | ⬜ |
| M5 | Streaming server (OpenAI-style SSE) | ⬜ |
| M6 | int8 weight-only quantisation | ⬜ |
| M7 | Benchmarks vs Hugging Face on a T4 | ⬜ |
| M8 | Stretch: Triton decode kernel, CUDA graphs or prefix caching | ⬜ |
| M9 | README results and demo | ⬜ |

## Correctness so far

Our forward pass loads the real TinyLlama-1.1B-Chat safetensors and is compared with `transformers` on 5 fixed prompts of 6 to 361 tokens.

| Check | Result |
|---|---|
| Final-position logits vs Hugging Face, fp32 CPU | max abs difference **0.0** on all 5 prompts (bit-identical) |
| Argmax token vs Hugging Face | identical on all 5 prompts |
| Random tiny model vs Hugging Face (CI) | max abs difference < 1e-4 |
| Sensitivity check | scaling one MLP weight by 1% moves logits by 0.016, so the test detects real differences |
| KV-cache greedy decoding vs Hugging Face `generate()` | identical tokens on all 5 prompts (up to 32 new tokens) |
| Cached vs uncached decoding (tiny model, CI) | identical for 64 greedy tokens and for seeded top-k/top-p sampling |
| Paged-cache greedy decoding vs Hugging Face `generate()` | identical tokens on all 5 prompts; all 64 blocks returned afterwards |
| Paged vs contiguous cache (CI) | identical 40-token outputs for prompts of 1, 15, 16, 17 and 40 tokens (across block boundaries) |
| Two sequences of different lengths in one batch (CI) | same logits as each sequence alone; padding is masked |
| Block accounting (CI) | 1,000 random allocate/append/free sequences, free list returns to full, no leaks |

### Single-stream speed (development baseline, CPU)

From `scripts/single_stream.py`, median of 3 runs after warm-up. The headline GPU numbers come from the T4 benchmark in M7.

| Hardware | dtype | Prefill 128 tok | Prefill 512 tok | Decode |
|---|---|---|---|---|
| Intel Xeon 2.8 GHz, 2 vCPU | fp32 | 2,339 ms | 8,090 ms | 2.81 tok/s |

## How to run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q                                   # fast CPU tests (tiny random model)

python scripts/download.py --out ~/.cache/tinyllama
TINYINFER_MODEL_DIR=~/.cache/tinyllama pytest -q -m real -s   # real weights vs fixture
```

To rebuild the reference fixture (needs about 5 GB RAM in fp32):

```bash
python tests/fixtures/make_reference.py --model-dir ~/.cache/tinyllama --device cpu
```

## Layout

```
tinyinfer/model/config.py    LlamaConfig, tiny_config() for tests
tinyinfer/model/llama.py     RMSNorm, RoPE, GQA attention, SwiGLU MLP, decoder
tinyinfer/model/loader.py    safetensors -> our modules, with name and shape checks
tinyinfer/cache/contiguous.py   preallocated per-layer KV cache (M2)
tinyinfer/cache/block_manager.py  block allocator: free list, block tables, ref counts (M3)
tinyinfer/cache/paged.py        paged KV storage; attention reads through block tables (M3)
tinyinfer/engine/sampling.py    greedy, temperature, top-k, top-p, stop tokens
tinyinfer/engine/engine.py      LLMEngine.generate(): prefill + decode loop
scripts/single_stream.py        prefill latency and decode tokens/s
tests/test_model.py          shapes, causality, RoPE property, HF parity
tests/test_generate.py       cached == uncached decoding, sampling, HF generate() parity
tests/test_paged.py          paged == contiguous, batched padding, no block leaks
tests/fixtures/              reference.pt and the script that builds it
scripts/download.py          fetch TinyLlama from the Hugging Face Hub
```

## Design decisions

1. **Positions are always explicit.** The forward pass takes `positions` instead of inferring them from tensor shape, because a KV cache, padding and preemption all run tokens at positions that do not start at 0.
2. **GQA by `repeat_interleave` on the head axis.** Query head *h* reads KV head *h // groups*. This matches Hugging Face's `repeat_kv`; a wrong mapping still produces fluent text, which is why the parity test compares logits rather than generated text.
3. **RMSNorm in fp32.** Normalisation is computed in fp32 and cast back, matching the reference and avoiding fp16 overflow.
4. **Masks come from positions, not shapes.** Attention builds its causal mask by comparing query and key positions. The same code then handles a full prompt, one decode step against the cache, and (later) sequences at different lengths in one batch.
5. **Prefill and decode are separate calls.** Prefill runs the whole prompt in one compute-bound pass; decode runs one token per step and is limited by reading weights from memory, which is why batching (M4) and int8 weights (M6) help it.
6. **Paged KV cache with 16-token blocks.** Reserving KV memory for every request's maximum length wastes most of it. Blocks are handed out on demand, so a sequence wastes less than one block. In TinyLlama fp16 one block holds 16 tokens across all 22 layers in 360,448 bytes. The first version gathers a sequence's blocks into a contiguous view before attention: simple and correct, at the cost of a copy that a custom kernel can remove later.
7. **The loader is strict.** Every checkpoint tensor must map to exactly one parameter with the same shape, and every parameter must be filled, so a renamed weight fails loudly instead of loading half a model.

## How it works

### Block tables (M3)

```
logical positions of sequence A (37 tokens)     physical KV blocks (16 slots each)
 0..15  -> table[0] = block 7                    block 3: [B0..B15]           (sequence B)
16..31  -> table[1] = block 2                    block 2: [A16..A31]
32..36  -> table[2] = block 9 (5 of 16 used)     block 7: [A0..A15]
                                                 block 9: [A32..A36, free x11]
token at position p  ->  block table[p // 16], offset p % 16
```

Blocks need not be adjacent or in order. When A finishes, blocks 7, 2 and 9 go back to the free list for any other sequence.

*One step of continuous batching is added with M4.*
