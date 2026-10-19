# TinyInfer

An LLM inference engine written from scratch in PyTorch. It serves TinyLlama-1.1B and implements the core ideas of vLLM itself: a KV cache, paged attention, continuous batching, int8 weights and a streaming HTTP API. Each optimisation is measured against Hugging Face `transformers`.

## Status

| Milestone | What | Status |
|---|---|---|
| M0 | Setup, CI, tiny test config, Hugging Face reference fixtures | ✅ done |
| M1 | Llama forward pass from scratch (RMSNorm, RoPE, GQA, SwiGLU) | ✅ done |
| M2 | KV cache, sampling, single-request `generate()` | ⬜ |
| M3 | Paged KV cache (block manager + paged attention) | ⬜ |
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
tests/test_model.py          shapes, causality, RoPE property, HF parity
tests/fixtures/              reference.pt and the script that builds it
scripts/download.py          fetch TinyLlama from the Hugging Face Hub
```

## Design decisions

1. **Positions are always explicit.** The forward pass takes `positions` instead of inferring them from tensor shape, because a KV cache, padding and preemption all run tokens at positions that do not start at 0.
2. **GQA by `repeat_interleave` on the head axis.** Query head *h* reads KV head *h // groups*. This matches Hugging Face's `repeat_kv`; a wrong mapping still produces fluent text, which is why the parity test compares logits rather than generated text.
3. **RMSNorm in fp32.** Normalisation is computed in fp32 and cast back, matching the reference and avoiding fp16 overflow.
4. **The loader is strict.** Every checkpoint tensor must map to exactly one parameter with the same shape, and every parameter must be filled, so a renamed weight fails loudly instead of loading half a model.

## How it works

*Filled in as milestones land: block tables and one step of continuous batching.*
