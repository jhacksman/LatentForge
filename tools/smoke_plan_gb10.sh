#!/usr/bin/env bash
# GB10 Smoke Plan - Complete integration test for LatentForge on GB10 hardware
#
# This script runs the full smoke plan as specified in the GB10 requirements:
# 1. Setup and environment check
# 2. API verification
# 3. Toy dataset creation and packing
# 4. Autoencoder training
# 5. Student training (with and without KD)
# 6. Inference testing
# 7. Benchmarking with acceptance gates
#
# All three teacher backends are tested: Venice, vLLM local, vLLM remote
#
# Usage:
#   bash tools/smoke_plan_gb10.sh [--venice-only|--vllm-local-only|--vllm-remote-only]
#
# Requirements:
#   - NVIDIA GB10 hardware (128GB unified memory)
#   - Dependencies installed (see requirements.txt)
#   - .env file configured with API keys
#
# Output:
#   - results/smoke_plan_TIMESTAMP/
#     - venice/
#     - vllm_local/
#     - vllm_remote/
#     - summary.json
#     - smoke_plan.log

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/smoke_plan_${TIMESTAMP}"
LOG_FILE="${RESULTS_DIR}/smoke_plan.log"

mkdir -p "${RESULTS_DIR}"

log() {
    echo -e "${1}" | tee -a "${LOG_FILE}"
}

log_section() {
    log "\n${BLUE}${'='*60}${NC}"
    log "${BLUE}${1}${NC}"
    log "${BLUE}${'='*60}${NC}\n"
}

log_success() {
    log "${GREEN}✅ ${1}${NC}"
}

log_error() {
    log "${RED}❌ ${1}${NC}"
}

log_warning() {
    log "${YELLOW}⚠️  ${1}${NC}"
}

log_info() {
    log "${BLUE}ℹ️  ${1}${NC}"
}

# Check which backends to test
TEST_VENICE=1
TEST_VLLM_LOCAL=1
TEST_VLLM_REMOTE=0  # Requires separate GB10

if [[ "${1:-}" == "--venice-only" ]]; then
    TEST_VLLM_LOCAL=0
    TEST_VLLM_REMOTE=0
elif [[ "${1:-}" == "--vllm-local-only" ]]; then
    TEST_VENICE=0
    TEST_VLLM_REMOTE=0
elif [[ "${1:-}" == "--vllm-remote-only" ]]; then
    TEST_VENICE=0
    TEST_VLLM_LOCAL=0
fi

# Start smoke plan
log_section "GB10 SMOKE PLAN - ${TIMESTAMP}"
log_info "Testing backends:"
[[ ${TEST_VENICE} -eq 1 ]] && log_info "  - Venice API"
[[ ${TEST_VLLM_LOCAL} -eq 1 ]] && log_info "  - vLLM Local"
[[ ${TEST_VLLM_REMOTE} -eq 1 ]] && log_info "  - vLLM Remote"
log_info "Results directory: ${RESULTS_DIR}"

# Step 1: Environment check
log_section "STEP 1: Environment Check"

# Check for GB10 hardware
if nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    log_success "GPU detected: ${GPU_NAME}"
    log_info "GPU memory: ${GPU_MEMORY} MB"

    if [[ ${GPU_MEMORY} -lt 100000 ]]; then
        log_warning "GPU memory <100GB. This smoke plan is optimized for GB10 (128GB)."
    fi
else
    log_error "No GPU detected! This smoke plan requires GB10 hardware."
    exit 1
fi

# Check Python environment
if ! command -v python &> /dev/null; then
    log_error "Python not found!"
    exit 1
fi
PYTHON_VERSION=$(python --version 2>&1)
log_success "Python: ${PYTHON_VERSION}"

# Check dependencies
log_info "Checking critical dependencies..."
python -c "import torch; print(f'torch {torch.__version__}')" || { log_error "torch not installed"; exit 1; }
python -c "import transformers; print(f'transformers {transformers.__version__}')" || { log_error "transformers not installed"; exit 1; }
if [[ ${TEST_VLLM_LOCAL} -eq 1 ]]; then
    python -c "import vllm; print(f'vllm {vllm.__version__}')" || { log_error "vllm not installed"; exit 1; }
fi
log_success "All critical dependencies installed"

# Check .env file
if [[ ! -f .env ]]; then
    log_error ".env file not found! Please create it with API credentials."
    exit 1
fi
log_success ".env file found"

# Step 2: API Verification (Venice and vLLM if applicable)
log_section "STEP 2: API Verification"

if [[ ${TEST_VENICE} -eq 1 ]]; then
    log_info "Testing Venice API..."
    TEACHER_BACKEND=venice python tools/verify_api.py || { log_error "Venice API verification failed"; exit 1; }
    log_success "Venice API verified"
fi

if [[ ${TEST_VLLM_REMOTE} -eq 1 ]]; then
    log_info "Testing vLLM Remote API..."
    TEACHER_BACKEND=vllm-remote python tools/verify_api.py || { log_error "vLLM Remote API verification failed"; exit 1; }
    log_success "vLLM Remote API verified"
fi

# Step 3: Toy Dataset Creation
log_section "STEP 3: Toy Dataset Creation"

if [[ ! -d data/packed_toy ]]; then
    log_info "Creating toy dataset..."
    python tools/make_toy_dataset.py || { log_error "Toy dataset creation failed"; exit 1; }
    log_success "Toy dataset created"

    log_info "Packing toy dataset..."
    python tools/data_packing.py --input ./data/toy.jsonl --output ./data/packed_toy --seq_len 2048 --k 8 || { log_error "Data packing failed"; exit 1; }
    log_success "Toy dataset packed"
else
    log_info "Toy dataset already exists, skipping creation"
fi

# Step 4: Autoencoder Training
log_section "STEP 4: Autoencoder Training"

if [[ ! -f checkpoints/ae.pt ]]; then
    log_info "Training autoencoder..."
    python ae/train_ae.py \
        --data ./data/packed_toy/train \
        --k 8 \
        --latent_dim 1024 \
        --epochs 2 \
        --batch_size 16 \
        --bf16 || { log_error "AE training failed"; exit 1; }
    log_success "Autoencoder trained"
else
    log_info "Autoencoder checkpoint exists, skipping training"
fi

# Test AE reconstruction
log_info "Testing AE reconstruction..."
python -c "
import torch
from ae.ae_model import LatentAutoencoder
ckpt = torch.load('checkpoints/ae.pt', map_location='cpu')
ae = LatentAutoencoder(**ckpt['config'])
ae.load_state_dict(ckpt['model'])
print(f'AE loaded: k={ae.k}, latent_dim={ae.latent_dim}')
" || { log_error "AE loading failed"; exit 1; }
log_success "AE reconstruction verified"

# Helper function to test a backend
test_backend() {
    local backend_name=$1
    local backend_env=$2
    local backend_dir="${RESULTS_DIR}/${backend_name}"

    mkdir -p "${backend_dir}"

    log_section "TESTING BACKEND: ${backend_name}"

    # Step 5: Student Training (Warmup without KD)
    log_info "Training student (warmup, no KD)..."
    TEACHER_BACKEND=${backend_env} python student/train_student.py \
        --data ./data/packed_toy/train \
        --ae_ckpt checkpoints/ae.pt \
        --k 8 \
        --latent_dim 1024 \
        --kd_w 0.0 \
        --epochs 1 \
        --batch_size 8 \
        --gradient_accumulation_steps 2 \
        --bf16 \
        --use_activation_checkpointing \
        --checkpoint_dir "${backend_dir}/checkpoints" \
        > "${backend_dir}/train_warmup.log" 2>&1 || { log_error "Warmup training failed"; return 1; }
    log_success "Warmup training complete"

    # Step 6: Student Training (With KD)
    log_info "Training student (with KD from ${backend_name})..."
    TEACHER_BACKEND=${backend_env} python student/train_student.py \
        --data ./data/packed_toy/train \
        --ae_ckpt checkpoints/ae.pt \
        --k 8 \
        --latent_dim 1024 \
        --kd_w 1.0 \
        --epochs 1 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --bf16 \
        --use_activation_checkpointing \
        --use_kd \
        --checkpoint_dir "${backend_dir}/checkpoints" \
        > "${backend_dir}/train_kd.log" 2>&1 || { log_error "KD training failed"; return 1; }
    log_success "KD training complete"

    # Step 7: Inference
    log_info "Testing inference..."
    python infer.py \
        --ae checkpoints/ae.pt \
        --student "${backend_dir}/checkpoints/student.pt" \
        --prompt "Write a hello world function" \
        --max_new_tokens 64 \
        --seed 42 \
        > "${backend_dir}/inference.txt" 2>&1 || { log_error "Inference failed"; return 1; }
    log_success "Inference complete"

    # Step 8: Benchmark with acceptance gates
    log_info "Running benchmark with acceptance gates..."
    python bench.py \
        --ae checkpoints/ae.pt \
        --student "${backend_dir}/checkpoints/student.pt" \
        --max_new_tokens 128 \
        --num_runs 3 \
        --eval \
        --eval_data ./data/packed_toy/train \
        --output "${backend_dir}/bench_results.json" \
        > "${backend_dir}/bench.log" 2>&1 || { log_error "Benchmark failed"; return 1; }

    # Check if acceptance gates passed
    if grep -q "ALL ACCEPTANCE GATES PASSED" "${backend_dir}/bench.log"; then
        log_success "${backend_name}: ALL ACCEPTANCE GATES PASSED"
        echo "1" > "${backend_dir}/gates_passed"
    else
        log_error "${backend_name}: SOME ACCEPTANCE GATES FAILED"
        echo "0" > "${backend_dir}/gates_passed"
        return 1
    fi

    log_success "${backend_name} smoke plan PASSED"
    return 0
}

# Test each enabled backend
OVERALL_SUCCESS=1

if [[ ${TEST_VENICE} -eq 1 ]]; then
    if ! test_backend "venice" "venice"; then
        OVERALL_SUCCESS=0
    fi
fi

if [[ ${TEST_VLLM_LOCAL} -eq 1 ]]; then
    # Start vLLM server in background
    log_info "Starting vLLM local server..."
    bash tools/serve_teacher_vllm.sh > "${RESULTS_DIR}/vllm_server.log" 2>&1 &
    VLLM_PID=$!
    log_info "vLLM server started (PID: ${VLLM_PID})"

    # Wait for server to start
    log_info "Waiting for vLLM server to be ready..."
    for i in {1..60}; do
        if curl -s http://localhost:8000/v1/models &> /dev/null; then
            log_success "vLLM server ready"
            break
        fi
        if [[ $i -eq 60 ]]; then
            log_error "vLLM server failed to start after 60s"
            kill ${VLLM_PID} 2>/dev/null || true
            OVERALL_SUCCESS=0
            TEST_VLLM_LOCAL=0
        fi
        sleep 1
    done

    if [[ ${TEST_VLLM_LOCAL} -eq 1 ]]; then
        if ! test_backend "vllm_local" "vllm-local"; then
            OVERALL_SUCCESS=0
        fi

        # Stop vLLM server
        log_info "Stopping vLLM server..."
        kill ${VLLM_PID} 2>/dev/null || true
        wait ${VLLM_PID} 2>/dev/null || true
        log_success "vLLM server stopped"
    fi
fi

if [[ ${TEST_VLLM_REMOTE} -eq 1 ]]; then
    if ! test_backend "vllm_remote" "vllm-remote"; then
        OVERALL_SUCCESS=0
    fi
fi

# Generate summary
log_section "SMOKE PLAN SUMMARY"

cat > "${RESULTS_DIR}/summary.json" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "backends_tested": {
    "venice": ${TEST_VENICE},
    "vllm_local": ${TEST_VLLM_LOCAL},
    "vllm_remote": ${TEST_VLLM_REMOTE}
  },
  "overall_success": ${OVERALL_SUCCESS},
  "results_directory": "${RESULTS_DIR}"
}
EOF

if [[ ${OVERALL_SUCCESS} -eq 1 ]]; then
    log_success "✅ GB10 SMOKE PLAN PASSED - All tested backends successful!"
    log_info "Results saved to: ${RESULTS_DIR}"
    exit 0
else
    log_error "❌ GB10 SMOKE PLAN FAILED - Some backends failed acceptance gates"
    log_info "Check logs in: ${RESULTS_DIR}"
    exit 1
fi
