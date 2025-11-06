# LatentForge

**Minimal CALM-style Latent Vector Autoregressive Language Model**

LatentForge distills a token-level LLM into a latent-vector AR model, achieving ~K× faster autoregressive generation by predicting compressed latent vectors instead of individual tokens.

## Overview

LatentForge implements a two-stage architecture:

1. **Autoencoder (AE)**: Compresses K tokens into a single latent vector with high-fidelity reconstruction (target ≥99.5% exact match)
2. **Student Model**: Predicts next latent vectors autoregressively, trained with knowledge distillation from a frozen teacher LLM

### Key Features

- **Configurable compression**: Default K=8 tokens → 1 latent (8× fewer AR steps)
- **High-fidelity reconstruction**: VAE-based AE with KL clamping for stable training
- **Knowledge distillation**: Student trained with latent MSE + token CE + KL divergence to teacher
- **Production-ready**: CLI tools, FastAPI server, and benchmarking utilities
- **Efficient**: BF16 mixed precision, gradient accumulation, KV caching

## Architecture

### Autoencoder

```
Tokens [batch, K*N]
  → Embeddings
  → Patch Grouping [batch, N, K*hidden]
  → Encoder Layers + Squeeze [batch, N, hidden]
  → Encoder Layers
  → VAE (mean, logvar) [batch, N, D]
  ⟲ Reparameterize → Latent z [batch, N, D]
  → Decoder Layers
  → Expand [batch, N, K*hidden]
  → Decoder Layers
  → Token Logits [batch, K*N, vocab_size]
```

**Loss**: `recon_loss = CE(logits, tokens) * K` + `kl_loss = max(KL(z || N(0,1)), kl_clamp) * kl_weight`

### Student Model

```
Latent Input [batch, seq_len, D]
  → Linear Projection [batch, seq_len, hidden]
  → Transformer Layers (RoPE attention)
  → RMSNorm
  → Output Projection [batch, seq_len, D]
```

**Training Losses**:
- `mse_loss = MSE(pred_latent, target_latent)`
- `ce_loss = CE(AE.decode(pred_latent), target_tokens)`
- `kd_loss = KL(teacher_logits || student_decoded_logits)`
- `total = MSE_W * mse + CE_W * ce + KD_W * kd`

## Installation

```bash
# Clone repository
git clone <repo_url>
cd LatentForge

# Install dependencies
make setup
# or manually:
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers==4.43.0
- accelerate, datasets, fastapi, uvicorn

## Quick Start

### 1. Prepare Data

You need text data for training. Format options:
- Single text file: `train.txt`
- HuggingFace dataset: `wikitext-2-raw-v1`
- JSONL with `"text"` field

Example: Download a small dataset
```bash
# Use wikitext for testing
# Or prepare your own data.txt file with plain text
```

### 2. Train Autoencoder

```bash
make train-ae DATA=train.txt K=8 D=1024 EPOCHS=2
```

**Parameters**:
- `DATA`: Path to training data (required)
- `K`: Patch size / compression factor (default: 8)
- `D`: Latent dimension (default: 1024)
- `EPOCHS`: Training epochs (default: 2)

**Detailed command**:
```bash
python ae/train_ae.py \
  --data_path train.txt \
  --output_dir checkpoints/ae \
  --K 8 \
  --D 1024 \
  --hidden_size 512 \
  --intermediate_size 1280 \
  --num_encoder_layers 2 \
  --num_decoder_layers 2 \
  --dropout 0.15 \
  --kl_weight 1e-3 \
  --kl_clamp 0.5 \
  --epochs 2 \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-4 \
  --block_size 2048 \
  --bf16
```

**Monitoring**: Training logs show:
- `loss`: Total loss
- `recon`: Reconstruction loss (CE)
- `kl`: KL divergence loss

**Target**: Validation exact match ≥ 99.5% (evaluated every `--eval_steps`)

### 3. Train Student Model

```bash
make train-student DATA=train.txt TEACHER=meta-llama/Llama-3.2-1B KD_W=1.0 MSE_W=1.0 CE_W=1.0
```

**Parameters**:
- `DATA`: Training data path (required)
- `TEACHER`: Teacher model for distillation (HF model ID)
- `KD_W`: KL divergence weight (default: 1.0)
- `MSE_W`: Latent MSE weight (default: 1.0)
- `CE_W`: Token CE weight (default: 1.0)

**Detailed command**:
```bash
python student/train_student.py \
  --data_path train.txt \
  --ae_path checkpoints/ae \
  --teacher_model meta-llama/Llama-3.2-1B \
  --output_dir checkpoints/student \
  --K 8 \
  --hidden_size 768 \
  --num_layers 12 \
  --num_heads 12 \
  --intermediate_size 2048 \
  --KD_W 1.0 \
  --MSE_W 1.0 \
  --CE_W 1.0 \
  --temperature 2.0 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-4 \
  --block_size 2048 \
  --bf16
```

**Note**: The autoencoder remains frozen during student training. Only the student model is trained.

### 4. Generate Text

```bash
make infer PROMPT="Once upon a time" MAX_NEW_TOKENS=128
```

**Detailed command**:
```bash
python infer.py \
  --ae checkpoints/ae \
  --student checkpoints/student \
  --prompt "Once upon a time" \
  --max_new_tokens 128 \
  --temperature 0.8 \
  --top_p 0.9 \
  --top_k 0 \
  --seed 42 \
  --bf16 \
  --use_kv_cache
```

**Sampling parameters**:
- `temperature`: 0.0 = greedy, higher = more random
- `top_p`: Nucleus sampling threshold
- `top_k`: Top-k filtering (0 = disabled)
- `seed`: Random seed for reproducibility

### 5. Start REST API Server

```bash
make serve PORT=7860
```

**API Endpoints**:

- `GET /`: Health check
- `GET /info`: Model information
- `POST /generate`: Generate text

**Example request**:
```bash
curl -X POST http://localhost:7860/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_new_tokens": 128,
    "temperature": 0.8,
    "top_p": 0.9,
    "seed": 42
  }'
```

**Response**:
```json
{
  "generated_text": "Once upon a time, there was...",
  "prompt": "Once upon a time",
  "num_tokens_generated": 128,
  "compression_factor": 8
}
```

### 6. Run Benchmarks

```bash
make bench
```

**Detailed command**:
```bash
python bench.py \
  --ae checkpoints/ae \
  --student checkpoints/student \
  --teacher meta-llama/Llama-3.2-1B \
  --output benchmark_results.json \
  --num_samples 20 \
  --max_new_tokens 128 \
  --bf16
```

**Metrics**:
- Tokens per second
- Latency per token (ms)
- AR steps reduction (should be ~K×)
- Throughput ratio vs baseline
- Generated samples for quality evaluation

Results saved to `benchmark_results.json`.

## Project Structure

```
LatentForge/
├── ae/
│   ├── ae_model.py          # Autoencoder architecture
│   ├── train_ae.py          # AE training script
│   └── tokenizer_adapter.py # Data preparation utilities
├── student/
│   ├── student_model.py     # Student transformer model
│   ├── train_student.py     # Student training with KD
│   └── sampler.py           # Generation utilities
├── infer.py                 # CLI inference script
├── server.py                # FastAPI REST server
├── bench.py                 # Benchmark script
├── Makefile                 # Convenient commands
├── requirements.txt         # Python dependencies
└── README_local.md          # This file
```

## Configuration

### Autoencoder Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K` | 8 | Tokens per latent (compression factor) |
| `D` | 1024 | Latent dimension |
| `hidden_size` | 512 | Hidden dimension |
| `intermediate_size` | 1280 | MLP dimension |
| `num_encoder_layers` | 2 | Encoder layers |
| `num_decoder_layers` | 2 | Decoder layers |
| `dropout` | 0.15 | Dropout rate |
| `kl_weight` | 1e-3 | KL divergence weight |
| `kl_clamp` | 0.5 | Minimum KL (prevents collapse) |

### Student Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 768 | Transformer hidden size |
| `num_layers` | 12 | Number of layers |
| `num_heads` | 12 | Attention heads |
| `intermediate_size` | 2048 | MLP dimension |
| `dropout` | 0.0 | Dropout rate |

### Training Hyperparameters

| Parameter | Default | AE | Student |
|-----------|---------|----|---------|
| `learning_rate` | 3e-4 | ✓ | ✓ |
| `batch_size` | - | 8 | 4 |
| `gradient_accumulation` | - | 4 | 8 |
| `warmup_steps` | - | 1000 | 2000 |
| `weight_decay` | 0.1 | ✓ | ✓ |
| `block_size` | 2048 | ✓ | ✓ |
| `bf16` | True | ✓ | ✓ |

## Advanced Usage

### Custom Tokenizer

By default, LatentForge uses `meta-llama/Llama-3.2-1B` tokenizer. To use a different tokenizer:

```bash
python ae/train_ae.py \
  --tokenizer gpt2 \
  --data_path train.txt \
  ...
```

### Adjusting Loss Weights

Experiment with different loss weight combinations:

```bash
# More emphasis on latent reconstruction
make train-student DATA=train.txt MSE_W=2.0 CE_W=0.5 KD_W=0.5

# More emphasis on knowledge distillation
make train-student DATA=train.txt MSE_W=0.5 CE_W=0.5 KD_W=2.0
```

### Distributed Training

For multi-GPU training, use `torchrun`:

```bash
# AE training on 4 GPUs
torchrun --nproc_per_node=4 ae/train_ae.py \
  --data_path train.txt \
  --output_dir checkpoints/ae \
  --batch_size 8 \
  ...

# Student training on 4 GPUs
torchrun --nproc_per_node=4 student/train_student.py \
  --data_path train.txt \
  --ae_path checkpoints/ae \
  --teacher_model meta-llama/Llama-3.2-1B \
  --batch_size 4 \
  ...
```

### Resume Training

To resume from a checkpoint:

```bash
# Checkpoints are saved at intervals specified by --save_steps
# To resume, just point to the checkpoint directory
python ae/train_ae.py \
  --data_path train.txt \
  --output_dir checkpoints/ae/checkpoint-10000 \
  ...
```

### Custom Evaluation Prompts

For benchmarking with custom prompts:

```bash
# Create prompts.txt with one prompt per line
echo "The future of AI is" > prompts.txt
echo "Once upon a time" >> prompts.txt
echo "In the year 2050," >> prompts.txt

python bench.py \
  --ae checkpoints/ae \
  --student checkpoints/student \
  --teacher meta-llama/Llama-3.2-1B \
  --prompts_file prompts.txt \
  --num_samples 10
```

## Troubleshooting

### Low Reconstruction Accuracy

If AE reconstruction < 99.5%:
- Train longer (`--epochs` or `--max_steps`)
- Increase latent dimension `D`
- Increase model capacity (`--hidden_size`, `--num_encoder_layers`)
- Check data quality and diversity

### Student Training Instability

If student losses diverge:
- Ensure AE is well-trained (≥99.5% reconstruction)
- Reduce learning rate
- Adjust loss weights (try lower `KD_W`)
- Increase warmup steps
- Check for NaN gradients (reduce `temperature` for KD)

### OOM (Out of Memory)

If you encounter CUDA OOM:
- Reduce `--batch_size`
- Increase `--gradient_accumulation_steps` (keeps effective batch size)
- Reduce `--block_size` (e.g., 1024 instead of 2048)
- Use gradient checkpointing (add to model code)
- Use smaller models (`--hidden_size`, `--num_layers`)

### Generation Quality Issues

If generated text is poor:
- Train student model longer
- Increase student model capacity
- Adjust loss weights (try higher `CE_W` or `KD_W`)
- Use lower temperature during generation
- Check that AE reconstruction is accurate

## Performance Tips

1. **Use BF16**: Always enable `--bf16` if your GPU supports it (A100, H100, etc.)
2. **Gradient accumulation**: Increase for larger effective batch sizes without OOM
3. **Block size**: Larger `--block_size` (2048-8192) improves context learning
4. **KV caching**: Use `--use_kv_cache` during inference for faster generation
5. **Flash Attention**: Install `flash-attn` for faster attention (optional)

## Acceptance Criteria

✓ **AE reconstruction**: ≥99.5% exact token match on validation set for K=8
✓ **Student generation**: Successfully generates text via latent-space AR
✓ **Efficiency**: ~K× fewer AR steps compared to token-level baseline
✓ **Coherence**: Generated text is coherent and relevant to prompts (qualitative)

## Implementation Details

### Key Design Decisions

1. **VAE with KL Clamping**: Prevents posterior collapse while maintaining latent structure
2. **Two-Stage Training**: AE frozen during student training provides stable targets
3. **Knowledge Distillation**: Three complementary losses (MSE, CE, KL) for robust learning
4. **Patch-Based Compression**: K tokens → 1 latent for K× speedup
5. **RoPE**: Rotary position embeddings for better long-range modeling

### Differences from CALM

LatentForge adapts CALM's core idea with some simplifications:

- **Simpler student training**: Direct latent prediction vs energy-based/diffusion methods
- **Knowledge distillation**: Added teacher KL loss for better quality
- **Configurable K and D**: K=8, D=1024 vs CALM's K=4, D=128
- **Minimal dependencies**: No custom CUDA kernels or specialized libraries

## Citation

If you use LatentForge, please cite the original CALM paper:

```bibtex
@article{shao2025calm,
  title={Continuous Autoregressive Language Models},
  author={Shao, Chenze and others},
  journal={arXiv preprint arXiv:2510.27688},
  year={2025}
}
```

## License

MIT License - See repository for details

## Contributing

Contributions welcome! Areas for improvement:
- Energy-based or diffusion student heads
- Flash Attention integration
- Quantization (INT8/INT4)
- Larger-scale training recipes
- Streaming inference
- Multi-modal extensions

## Support

For issues, questions, or feature requests, please open a GitHub issue.

---

**Built with ❤️ by your senior engineer**
