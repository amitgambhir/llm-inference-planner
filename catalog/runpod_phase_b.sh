#!/usr/bin/env bash
# Phase B calibration benchmark — run this on a RunPod 1×H100-SXM-80GB pod.
#
# Prerequisites on the pod:
#   pip install vllm
#   huggingface-cli login  (or set HF_TOKEN)
#   git clone https://github.com/vllm-project/vllm /tmp/vllm  (for benchmark_throughput.py)
#
# Usage:
#   bash runpod_phase_b.sh 2>&1 | tee phase_b_results.txt
#   # then copy phase_b_results.txt back and fill in benchmarks_public.yaml

set -euo pipefail

MODEL="meta-llama/Llama-3.1-8B-Instruct"
GPU_MEM_UTIL=0.95
MAX_MODEL_LEN=25000
NUM_PROMPTS=500

# Record vLLM version first
echo "=== Environment ==="
python -c "import vllm; print('vLLM version:', vllm.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Start vLLM server in background
echo "=== Starting vLLM server ==="
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --quantization fp8 \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size 1 \
    --disable-log-requests &
SERVER_PID=$!

# Wait for server readiness
echo "Waiting for server..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    sleep 5
done

# Warmup run (discarded)
echo "=== Warmup (ISL=128 OSL=128) ==="
python /tmp/vllm/benchmarks/benchmark_throughput.py \
    --backend vllm \
    --model "$MODEL" \
    --input-len 128 \
    --output-len 128 \
    --num-prompts 100 \
    2>&1 | grep -E "Throughput|output_token" || true

echo ""
echo "=== Phase B Calibration Runs ==="
echo "gpu_memory_utilization=$GPU_MEM_UTIL  num_prompts=$NUM_PROMPTS  tp=1"
echo ""

for pair in "128 128" "1000 1000" "2048 128" "5000 500" "20000 2000"; do
    ISL=$(echo "$pair" | cut -d' ' -f1)
    OSL=$(echo "$pair" | cut -d' ' -f2)
    echo "--- ISL=$ISL OSL=$OSL ---"
    python /tmp/vllm/benchmarks/benchmark_throughput.py \
        --backend vllm \
        --model "$MODEL" \
        --input-len "$ISL" \
        --output-len "$OSL" \
        --num-prompts "$NUM_PROMPTS" \
        2>&1 | grep -E "Throughput|output_token|tokens/s"
    echo ""
done

# Clean up
kill $SERVER_PID 2>/dev/null || true

echo "=== Done. Copy this file back and record output_tps in catalog/vllm_calibration_scratch.txt ==="
