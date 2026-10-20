#!/usr/bin/env bash
# Full M7 protocol on a GPU box (Kaggle/Colab T4). Writes everything to bench/results/.
#   MODEL_DIR=~/.cache/tinyllama bash bench/run_all.sh
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-$HOME/.cache/tinyllama}"
REPEATS="${REPEATS:-3}"
cd "$(dirname "$0")/.."
mkdir -p bench/results
rm -f bench/results/raw.jsonl

[ -f "$MODEL_DIR/config.json" ] || python scripts/download.py --out "$MODEL_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee bench/results/gpu.txt

echo "== correctness on this GPU (fp16, fixture regenerated with HF on the same GPU)"
python tests/fixtures/make_reference.py --model-dir "$MODEL_DIR" --device cuda \
  --out tests/fixtures/reference_cuda.pt
TINYINFER_MODEL_DIR="$MODEL_DIR" TINYINFER_FIXTURE=tests/fixtures/reference_cuda.pt \
  python -m pytest -q -m "real or gpu" | tee bench/results/gpu_tests.txt || true

echo "== single stream"
python scripts/single_stream.py --model-dir "$MODEL_DIR" --out bench/results/single_stream_gpu.jsonl

echo "== memory"
python bench/memory.py --model-dir "$MODEL_DIR" --out bench/results/memory_gpu.json

echo "== perplexity (full WikiText-2 test split)"
python bench/perplexity.py --model-dir "$MODEL_DIR" --out bench/results/perplexity_gpu.json

echo "== throughput and latency vs Hugging Face"
python bench/throughput.py --model-dir "$MODEL_DIR" --repeats "$REPEATS" \
  --concurrency 1 4 16 32 --out bench/results/raw.jsonl

python bench/report.py --raw bench/results/raw.jsonl --out bench/results
if [ -f bench/results/m7_baseline/summary.json ]; then
  python bench/compare.py bench/results/m7_baseline/summary.json bench/results/summary.json \
    --out bench/results/comparison_vs_m7.md
fi
echo "done: bench/results/summary.md"
