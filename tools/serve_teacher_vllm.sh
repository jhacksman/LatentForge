#!/usr/bin/env bash
#
# Serve Qwen3-Next-80B with vLLM OpenAI-compatible API server.
# Optimized for single GB10 with INT4 quantization.
#

set -euo pipefail

# Load env vars from .env if available
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Configuration from environment or defaults
# GB10 optimized: Lower memory util and max len to leave room for student
MODEL="${VLLM_LOCAL_MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
QUANT="${VLLM_QUANTIZATION:-gptq}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.80}"  # Reduced from 0.85 for headroom
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"  # Reduced from 32768 to lower KV pressure

echo "=================================================="
echo "Starting vLLM OpenAI-compatible API server (GB10 optimized)"
echo "=================================================="
echo "Model: $MODEL"
echo "Quantization: $QUANT (INT4 recommended)"
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_MEM_UTIL (headroom for student)"
echo "Max Model Length: $MAX_MODEL_LEN (reduced for KV cache)"
echo "Tensor Parallel: 1 (single GB10)"
echo "=================================================="
echo ""

# Run vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --quantization "$QUANT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --port "$PORT" \
    --served-model-name qwen3-next-80b \
    --trust-remote-code \
    --disable-log-requests

# Note: Server will be available at http://localhost:$PORT/v1
# Compatible with OpenAI API format
echo ""
echo "Server running at http://localhost:$PORT/v1"
echo "Docs available at http://localhost:$PORT/docs"
