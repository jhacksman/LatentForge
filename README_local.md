# LatentForge

A minimal stack to distill a token-level LLM into a CALM-style latent-vector autoregressive model using **Qwen 3 Next 80B** as the frozen teacher via the Venice API.

## Overview

LatentForge implements:

- **Autoencoder (AE)**: Maps K tokens → 1 latent vector (default K=8)
- **Student Transformer**: Predicts next latent vector autoregressively
- **Knowledge Distillation**: From Qwen 3 Next 80B teacher via Venice API
- **CLI & REST API**: Easy inference and serving

### Key Features

- **K× faster inference**: ~8× fewer autoregressive steps than token-level models
- **Flexible compression**: Configurable K parameter
- **Knowledge distillation**: Learn from teacher logprobs via Venice API
- **Production-ready**: FastAPI server, benchmarking tools, Docker support

## Architecture

```
Input Text → Tokenize → [Token₁, Token₂, ..., Tokenₖ]
                              ↓
                         Autoencoder
                              ↓
                         Latent z₁
                              ↓
                    Student Transformer
                              ↓
                     Predicted Latent z₂
                              ↓
                     AE Decoder
                              ↓
              [Token_{k+1}, ..., Token_{2k}]
```

### Training Process

1. **AE Training**: Learn to compress K tokens into 1 latent (≥99.5% reconstruction)
2. **Student Training**: Predict next latent with KD from teacher
   - CE loss on decoded tokens
   - MSE loss in latent space
   - KL divergence to teacher distributions

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- Venice API key

### Installation

```bash
# Clone repository
git clone https://github.com/jhacksman/LatentForge
cd LatentForge

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create `.env` file with your Venice API credentials:

```bash
VENICE_BASE_URL=https://api.venice.ai/api/v1
VENICE_API_KEY=your_api_key_here
VENICE_MODEL=qwen3-next-80b
```

**Important**: Never commit `.env` to version control!

### Verify API Connection

```bash
make check-api
# or
python tools/verify_api.py
```

## Quick Start

### 1. Prepare Data

Pack your text data into sequences:

```bash
python tools/data_packing.py \
  --input data/text.jsonl \
  --output data/packed \
  --seq_len 4096 \
  --k 8
```

### 2. Train Autoencoder

```bash
make train-ae K=8 D=1024 EPOCHS=10
# or
python ae/train_ae.py \
  --data data/packed/train \
  --k 8 \
  --latent_dim 1024 \
  --epochs 10 \
  --bf16
```

**Target**: ≥99.5% exact reconstruction on validation

### 3. Train Student with KD

```bash
make train-student KD_W=1.0 MSE_W=1.0 CE_W=1.0
# or
python student/train_student.py \
  --data data/packed/train \
  --ae_ckpt checkpoints/ae.pt \
  --k 8 \
  --latent_dim 1024 \
  --kd_w 1.0 --mse_w 1.0 --ce_w 1.0 \
  --use_kd \
  --bf16
```

### 4. Generate Text

```bash
make infer PROMPT="Write a function"
# or
python infer.py \
  --ae checkpoints/ae.pt \
  --student checkpoints/student.pt \
  --prompt "Write a short function" \
  --max_new_tokens 128 \
  --temperature 0.8
```

### 5. Start API Server

```bash
make serve PORT=7860
# or
uvicorn server:app --port 7860
```

API endpoint:

```bash
curl -X POST http://localhost:7860/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a function",
    "max_new_tokens": 128,
    "temperature": 0.8,
    "top_p": 0.95
  }'
```

### 6. Benchmark Performance

```bash
make bench
# or
python bench.py \
  --ae checkpoints/ae.pt \
  --student checkpoints/student.pt \
  --max_new_tokens 128
```

## Project Structure

```
LatentForge/
├── ae/                      # Autoencoder module
│   ├── ae_model.py         # AE architecture
│   ├── train_ae.py         # AE training script
│   └── tokenizer_adapter.py # Tokenizer wrapper
├── student/                 # Student model module
│   ├── student_model.py    # Student architecture
│   ├── train_student.py    # Student training with KD
│   └── sampler.py          # Generation sampler
├── kd/                      # Knowledge distillation
│   ├── kd_client.py        # Venice API client
│   └── kd_batcher.py       # Batch KD requests
├── tools/                   # Utilities
│   ├── data_packing.py     # Data preprocessing
│   └── verify_api.py       # API verification
├── configs/                 # YAML configurations
│   ├── ae_default.yaml
│   ├── student_default.yaml
│   └── inference.yaml
├── infer.py                 # CLI inference
├── server.py                # FastAPI server
├── bench.py                 # Benchmarking tool
├── Makefile                 # Automation targets
├── requirements.txt         # Python dependencies
├── .env                     # API credentials (do not commit)
└── README_local.md          # This file
```

## Configuration

### Autoencoder Parameters

- `k`: Compression factor (default: 8)
- `latent_dim`: Latent vector dimension (default: 1024)
- `embed_dim`: Token embedding dimension (default: 768)
- `num_layers`: Number of transformer layers (default: 4)

### Student Parameters

- `latent_dim`: Must match AE (default: 1024)
- `hidden_dim`: Hidden dimension (default: 2048)
- `num_layers`: Number of layers (default: 12)
- `seq_len`: Sequence length in latents (default: 64)

### Loss Weights

- `kd_w`: Knowledge distillation weight (default: 1.0)
- `mse_w`: MSE in latent space weight (default: 1.0)
- `ce_w`: Cross-entropy on tokens weight (default: 1.0)

## Makefile Targets

```bash
make help          # Show all targets
make setup         # Setup environment
make check-api     # Verify Venice API
make train-ae      # Train autoencoder
make train-student # Train student with KD
make infer         # Run inference
make serve         # Start FastAPI server
make bench         # Run benchmark
make clean         # Clean generated files
```

## Venice API Documentation

- **Getting Started**: https://docs.venice.ai/overview/getting-started
- **Models List**: https://docs.venice.ai/api-reference/endpoint/models/list
- **Chat Completions**: https://docs.venice.ai/api-reference/endpoint/chat/completions
- **Available Models**: https://docs.venice.ai/overview/models

## Related Work

### CALM (Confident Adaptive Language Modeling)

CALM enables early exiting in language models by learning when to stop computation.

- **Paper**: https://arxiv.org/abs/2207.07061
- **Repository**: https://github.com/shaochenze/calm

LatentForge extends CALM's ideas by:
- Operating in latent space instead of token space
- Using knowledge distillation from a teacher model
- Achieving K× speedup through compression

### Qwen Models

Qwen 3 Next 80B is used as the frozen teacher model via Venice API.

- **Model Card**: https://huggingface.co/Qwen
- **Venice Integration**: https://docs.venice.ai/overview/models

## Acceptance Criteria

✅ API verification passes and shows `qwen3-next-80b` available
✅ AE achieves ≥99.5% exact reconstruction on validation for K=8
✅ Student generates end-to-end via latent steps without errors
✅ Benchmark shows ~K× fewer AR steps than token LLM

## Troubleshooting

### Venice API Issues

```bash
# Test API manually
curl -H "Authorization: Bearer $VENICE_API_KEY" \
  "$VENICE_BASE_URL/models"
```

### CUDA Out of Memory

- Reduce `batch_size`
- Use `--bf16` for mixed precision
- Reduce `seq_len` for student training

### Low Reconstruction Accuracy

- Increase `epochs` for AE training
- Increase `num_layers` in AE
- Reduce `l2_weight`

## Performance Tips

1. **Use BF16**: Add `--bf16` flag for 2× faster training
2. **Batch Size**: Increase until OOM for better GPU utilization
3. **Data Quality**: Clean, diverse data improves results
4. **K Parameter**: Start with K=8, experiment with 4, 16, 32

## Development

### Running Tests

```bash
make test-tokenizer  # Test tokenizer
make test-ae         # Test autoencoder
make test-student    # Test student model
make test-kd         # Test KD client
```

### Code Style

- Type hints everywhere
- Dataclasses for configuration
- Docstrings for public APIs

## License

MIT

## Citation

If you use LatentForge in your research, please cite:

```bibtex
@software{latentforge2025,
  title={LatentForge: Latent-Space Autoregressive Language Modeling},
  author={Your Name},
  year={2025},
  url={https://github.com/jhacksman/LatentForge}
}
```

## Acknowledgments

- Venice AI for API access to Qwen 3 Next 80B
- CALM paper for inspiration
- Qwen team for the teacher model
