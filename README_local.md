# LatentForge

**LatentForge** is a minimal stack to distill a token-level LLM into a CALM-style latent-vector autoregressive model using **Qwen 3 Next 80B** as the frozen teacher via the Venice API.

This implementation achieves ~K× speedup (K=8 by default) by operating in compressed latent space instead of token space, with knowledge distillation from teacher logprobs.

## Overview

### Architecture

- **Autoencoder (AE)**: Compresses K tokens → 1 latent vector (default K=8)
- **Student Transformer**: Predicts next latent autoregressively
- **Knowledge Distillation**: From Qwen 3 Next 80B via Venice API with sparse top_logprobs (top-20 tokens per position)
- **Full Stack**: CLI, REST API, benchmarking, and comprehensive tests

### Key Features

- **K× fewer AR steps**: ~8× speedup by compressing tokens into latents
- **Teacher distillation**: KL divergence loss from Qwen 3 Next 80B logprobs
- **Production-ready**: FastAPI server with CORS, pytest test suite
- **Flexible losses**: Configurable CE, MSE, and KD weights
- **GB10 Optimized**: Single-node deployment on NVIDIA GB10 Grace-Blackwell superchip

## GB10 Single-Node Teacher Options

LatentForge supports three teacher deployment scenarios optimized for **single GB10** (NVIDIA Grace-Blackwell superchip with 128GB unified memory):

### Scenario A: Venice API (Remote Teacher)

Use Venice.ai's hosted Qwen3-Next-80B via API. No local GPU resources required for teacher.

```bash
# .env configuration
TEACHER_BACKEND=venice
VENICE_BASE_URL=https://api.venice.ai/api/v1
VENICE_API_KEY=your_key_here
VENICE_MODEL=qwen3-next-80b

# Train student
make train-student-venice
```

**Pros**: No local teacher compute, always available, easy setup
**Cons**: API rate limits, network latency, costs per token

### Scenario B: Local vLLM Teacher (Same GB10)

Run Qwen3-Next-80B INT4 quantized **on the same GB10** using vLLM. Recommended configuration:

```bash
# .env configuration
TEACHER_BACKEND=vllm-local
VLLM_LOCAL_MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct
VLLM_QUANTIZATION=gptq           # or awq
VLLM_MAX_MODEL_LEN=32768
VLLM_GPU_MEMORY_UTIL=0.85        # Leave room for student

# Option 1: In-process (loads model directly in training)
make train-student-vllm-local

# Option 2: Separate server process (recommended)
make serve-vllm-local &          # Start teacher server
make train-student-vllm-local    # Train student
```

**Memory Breakdown on GB10 (128GB unified)**:
- Qwen3-Next-80B INT4 (GPTQ): ~40GB
- Student + gradients + activations: ~30GB
- KV cache + vLLM overhead: ~20GB
- System + buffers: ~38GB

**Pros**: No API costs, low latency, full control
**Cons**: Requires downloading 80B model (~40GB), shares GPU with student

### Scenario C: Remote vLLM Teacher (Another GB10)

Run Qwen3-Next-80B on a **separate GB10** and connect via OpenAI-compatible API:

```bash
# On teacher GB10:
cd LatentForge
make serve-vllm-local           # Serves on port 8000

# On student GB10 - .env configuration:
TEACHER_BACKEND=vllm-remote
VLLM_REMOTE_URL=http://10.0.0.5:8000/v1
VLLM_REMOTE_API_KEY=              # Optional

# Train student
make train-student-vllm-remote
```

**Pros**: Full 128GB available for each model, no resource contention, fastest inference
**Cons**: Requires two GB10 systems, network setup

### GB10 Optimization Notes

1. **INT4 Quantization Recommended**:
   - GPTQ or AWQ reduces 80B model from ~160GB (FP16) to ~40GB
   - Minimal quality loss for KD purposes
   - Fits comfortably on single GB10

2. **Bandwidth Constraints**:
   - GB10 has 301 GB/s memory bandwidth (vs 3350 GB/s on H100)
   - Use gradient accumulation: `--gradient_accumulation_steps 4`
   - Use activation checkpointing: `--use_activation_checkpointing`
   - Smaller batch sizes: 8-16 instead of 32+

3. **KD Caching**:
   - All backends cache distributions in `./cache/kd_cache.sqlite`
   - Reduces API calls and network traffic
   - View stats: automatically printed after training

4. **Recommended Settings for GB10**:
   ```bash
   # AE training
   make train-ae K=8 D=1024 EPOCHS=10 BATCH_SIZE=16

   # Student training (any backend)
   python student/train_student.py \
     --data ./data/packed_toy \
     --ae_ckpt checkpoints/ae.pt \
     --k 8 \
     --latent_dim 1024 \
     --batch_size 8 \
     --gradient_accumulation_steps 4 \
     --use_activation_checkpointing \
     --bf16
   ```

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- Venice API key for Qwen 3 Next 80B

### Installation

Clone the repository:

```bash
git clone https://github.com/jhacksman/LatentForge
cd LatentForge
```

### Environment Setup

Create `.env` file at the repository root with your Venice API credentials:

```bash
cat > .env <<'EOF'
VENICE_BASE_URL=https://api.venice.ai/api/v1
VENICE_API_KEY=your_api_key_here
VENICE_MODEL=qwen3-next-80b
EOF
```

**Important**: The `.env` file is in `.gitignore` and should never be committed!

### Install Dependencies

Create virtual environment and install dependencies:

```bash
make setup
```

This will:
- Create `.venv/` virtual environment
- Install all dependencies (PyTorch, transformers, FastAPI, pytest, etc.)
- Create necessary directories

Then activate the environment:

```bash
source .venv/bin/activate
```

## Quick Start

### 1. Verify Venice API

Check that your API credentials work and Qwen 3 Next 80B is available:

```bash
make check-api
```

Expected output:
```
Test 1: Listing available models...
✅ Found N models
✅ Target model 'qwen3-next-80b' is available
✅ supportsLogProbs: True

Test 2: Chat completion with logprobs...
✅ Chat completion successful
✅ Logprobs are working correctly

🎉 All checks passed! Venice API is ready for use.
```

### 2. Create Toy Dataset

Generate a small test dataset for development:

```bash
make toy
```

This creates `./data/toy.jsonl` and packs it into `./data/packed_toy/` with sequences of length 2048 (multiple of K=8).

### 3. Train Autoencoder

Train the autoencoder to compress K=8 tokens into 1 latent:

```bash
make train-ae
```

Target: ≥99.5% exact reconstruction rate on validation set.

Default parameters:
- `K=8` (compression factor)
- `D=1024` (latent dimension)
- `EPOCHS=1` (for quick testing)

For better results:

```bash
make train-ae K=8 D=1024 EPOCHS=10
```

Checkpoint saved to: `checkpoints/ae.pt`

### 4. Train Student with KD

Train the student transformer with knowledge distillation:

```bash
make train-student
```

This trains the student to predict next latents with:
- **CE loss**: Cross-entropy on decoded tokens
- **MSE loss**: Mean squared error in latent space
- **KD loss**: KL divergence to teacher distributions from sparse top_logprobs (top-20 tokens per position)

Default weights: `KD_W=1.0`, `MSE_W=1.0`, `CE_W=1.0`

**Note**: The KD loss uses Venice API's `top_logprobs=20` parameter to get sparse teacher distributions, providing efficient knowledge transfer without requiring full vocabulary distributions.

Checkpoint saved to: `checkpoints/student.pt`

### 5. Run Inference

Generate text with the trained models:

```bash
make infer
```

Or with custom parameters:

```bash
make infer PROMPT="Write a function to sort a list" MAX_TOKENS=128 TEMP=0.8
```

### 6. Benchmark Performance

Run throughput benchmarks:

```bash
make bench
```

This runs 5 prompts from `prompts.txt` and reports:
- Latent steps per second
- Decoded tokens per second
- Compression ratio (K×)

Results saved to: `results/bench_YYYYMMDD_HHMMSS.json`

### 7. Start REST API Server

Launch the FastAPI server:

```bash
make serve
```

Server runs on `http://localhost:7860` with:
- **Swagger docs**: `http://localhost:7860/docs`
- **Health check**: `GET /health`
- **Generate endpoint**: `POST /generate`

Example request:

```bash
curl -X POST http://localhost:7860/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function",
    "max_new_tokens": 64,
    "temperature": 0.8,
    "top_p": 0.95
  }'
```

## Testing

### Run All Tests

```bash
make test
```

### Run Fast Unit Tests Only

```bash
make test-fast
```

Tests:
- `test_env.py` - Environment and .env validation
- `test_kd_api.py` - Venice API client (list_models, logprobs)
- `test_pack.py` - Data packing (sequences are multiples of K)
- `test_ae_unit.py` - Autoencoder forward, decode, no NaN/inf
- `test_student_unit.py` - Student forward, optimizer step updates params

### Run End-to-End Tests

```bash
make test-e2e
```

Tests:
- `test_infer_e2e.py` - Inference returns non-empty, deterministic with seed
- `test_server_e2e.py` - Server health, generate, 5 concurrent requests
- `test_failure_modes.py` - Missing .env, bad key, bad URL, K validation

## Project Structure

```
LatentForge/
├── ae/                      # Autoencoder
│   ├── ae_model.py         # Model architecture
│   ├── tokenizer_adapter.py # Tokenizer wrapper
│   └── train_ae.py         # Training script
├── student/                 # Student model
│   ├── student_model.py    # Transformer architecture
│   ├── sampler.py          # Generation sampler
│   └── train_student.py    # Training with KD
├── kd/                      # Knowledge distillation
│   ├── kd_client.py        # Venice API client
│   └── kd_batcher.py       # Batch processing
├── tools/                   # Utilities
│   ├── verify_api.py       # API verification
│   ├── make_toy_dataset.py # Toy data generation
│   └── data_packing.py     # Data preprocessing
├── tests/                   # Pytest test suite
│   ├── test_env.py
│   ├── test_kd_api.py
│   ├── test_pack.py
│   ├── test_ae_unit.py
│   ├── test_student_unit.py
│   ├── test_infer_e2e.py
│   ├── test_server_e2e.py
│   └── test_failure_modes.py
├── infer.py                 # CLI inference
├── server.py                # FastAPI REST server
├── bench.py                 # Benchmarking tool
├── Makefile                 # Automation targets
├── requirements.txt         # Python dependencies
├── prompts.txt              # Benchmark prompts
├── .env                     # API credentials (DO NOT COMMIT)
└── README_local.md          # This file
```

## Configuration

### Autoencoder Parameters

- `K=8` - Compression factor (tokens per latent)
- `D=1024` - Latent dimension
- `EPOCHS=10` - Training epochs
- Target: ≥99.5% exact reconstruction

### Student Parameters

- `KD_W=1.0` - Knowledge distillation weight
- `MSE_W=1.0` - MSE in latent space weight
- `CE_W=1.0` - Cross-entropy on tokens weight
- `STEPS=50` - Training steps for smoke test

### Sampling Parameters

- `temperature` - Sampling temperature (0.0 = greedy, higher = more random)
- `top_p` - Nucleus sampling threshold
- `top_k` - Top-k sampling (0 = disabled)
- `seed` - Random seed for reproducibility

## Makefile Targets

```bash
make help          # Show all targets
make setup         # Setup environment and install deps
make check-api     # Verify Venice API
make toy           # Create toy dataset
make train-ae      # Train autoencoder
make train-student # Train student with KD
make infer         # Run inference
make serve         # Start FastAPI server
make bench         # Run benchmark
make test          # Run all tests
make test-fast     # Run fast unit tests
make test-e2e      # Run end-to-end tests
make clean         # Clean generated files
```

## Smoke Test Sequence

Run these commands in order to verify the complete pipeline:

```bash
# 1. Setup
make setup
source .venv/bin/activate

# 2. Verify API
make check-api

# 3. Create toy data
make toy

# 4. Train AE
make train-ae

# 5. Train student
make train-student

# 6. Run inference
make infer

# 7. Run benchmark
make bench

# 8. Run tests
pytest -q -k "env or kd_api or pack or ae_unit or student_unit"
pytest -q -k "infer_e2e" --timeout=120
pytest -q -k "server_e2e" --timeout=120
```

## Acceptance Criteria

✅ **API Verification**: `check-api` finds `qwen3-next-80b` with logprobs support
✅ **Toy Run**: Train AE and student, infer works end-to-end
✅ **Benchmark**: Writes `results/bench_*.json` and prints table
✅ **Tests Pass**: Unit and e2e tests pass with trained models
✅ **Documentation**: README matches working commands

## References

### Venice AI Documentation

- **Getting Started**: https://docs.venice.ai/overview/getting-started
- **Models List API**: https://docs.venice.ai/api-reference/endpoint/models/list
- **Chat Completions**: https://docs.venice.ai/api-reference/endpoint/chat/completions
- **Available Models**: https://docs.venice.ai/overview/models

### CALM (Confident Adaptive Language Modeling)

- **Paper**: https://arxiv.org/abs/2207.07061
- **Repository**: https://github.com/shaochenze/calm

CALM enables early exiting in language models. LatentForge extends this by:
- Operating in latent space instead of token space
- Using knowledge distillation from a powerful teacher
- Achieving K× speedup through token compression

### Qwen Models

- **Hugging Face**: https://huggingface.co/Qwen
- **Model Cards**: Qwen 3 Next series documentation

## Troubleshooting

### Venice API Errors

**401 Unauthorized**:
- Check `VENICE_API_KEY` in `.env`
- Verify key is valid and not expired

**Connection Errors**:
- Check `VENICE_BASE_URL` is correct
- Verify internet connectivity
- Check for firewall/proxy issues

**429 Rate Limiting**:
- KD client has exponential backoff (max 32s)
- Reduce batch size or add delays

### Training Issues

**CUDA Out of Memory**:
- Reduce `BATCH_SIZE`
- Use `--bf16` flag (already default)
- Reduce `seq_len` or `latent_dim`

**Low AE Reconstruction**:
- Increase `EPOCHS` (try 10-20)
- Increase `num_layers` in ae_model.py
- Check data quality

**Student Not Learning**:
- Verify AE is trained first
- Check loss weights balance
- Increase training steps

### Test Failures

**test_kd_api timeout**:
- Venice API may be slow
- Increase timeout or skip with `--timeout=180`

**test_infer_e2e not found**:
- Need trained checkpoints first
- Run `make train-ae` and `make train-student`

**test_server_e2e fails**:
- Port 7861 may be in use
- Kill existing servers: `pkill -f uvicorn`

## Development

### Code Style

- Type hints everywhere
- Dataclasses for configurations
- Docstrings for public APIs
- F-strings for formatting

### Adding New Features

1. Write tests first (`tests/test_*.py`)
2. Implement feature
3. Update Makefile if needed
4. Update this README
5. Run full test suite

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure all tests pass
5. Submit pull request

## License

MIT

## Citation

```bibtex
@software{latentforge2025,
  title={LatentForge: CALM-Style Latent-Vector AR Model with KD},
  author={Your Name},
  year={2025},
  url={https://github.com/jhacksman/LatentForge}
}
```

## Acknowledgments

- **Venice AI** for API access to Qwen 3 Next 80B
- **CALM paper** for architectural inspiration
- **Qwen team** for the teacher model
