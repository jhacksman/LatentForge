.PHONY: help setup check-api toy train-ae train-student infer serve bench test test-fast test-e2e clean

# Default target
help:
	@echo "LatentForge - Makefile targets:"
	@echo ""
	@echo "  setup           - Create venv and install dependencies"
	@echo "  check-api       - Verify Venice API connection"
	@echo "  toy             - Create toy dataset and pack it"
	@echo "  train-ae        - Train autoencoder"
	@echo "  train-student   - Train student with KD"
	@echo "  infer           - Run inference"
	@echo "  serve           - Start FastAPI server"
	@echo "  bench           - Run benchmark"
	@echo "  test            - Run all tests"
	@echo "  test-fast       - Run fast unit tests only"
	@echo "  test-e2e        - Run end-to-end tests"
	@echo "  clean           - Clean generated files"
	@echo ""
	@echo "Example usage:"
	@echo "  make setup"
	@echo "  make check-api"
	@echo "  make toy"
	@echo "  make train-ae"
	@echo "  make train-student"
	@echo "  make infer"
	@echo ""

# Setup environment
setup:
	@echo "Setting up LatentForge..."
	python3 -m venv .venv || true
	@echo ""
	@echo "✅ Virtual environment created"
	@echo ""
	@echo "Activating and installing dependencies..."
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install torch transformers==4.43.0 accelerate fastapi uvicorn pydantic requests python-dotenv pytest pytest-timeout httpx tabulate rich
	@echo ""
	@echo "✅ Dependencies installed"
	@echo ""
	@if [ ! -f .env ]; then \
		echo "⚠️  .env file not found. Please create it with your Venice API credentials."; \
	else \
		echo "✅ .env file exists"; \
	fi
	@mkdir -p checkpoints results data configs
	@echo "✅ Directories created"
	@echo ""
	@echo "Setup complete! Activate with:"
	@echo "  source .venv/bin/activate"

# Verify API
check-api:
	@echo "Verifying Venice API..."
	python tools/verify_api.py

# Create toy dataset
toy:
	@echo "Creating toy dataset..."
	python tools/make_toy_dataset.py
	@echo "Packing toy dataset..."
	python tools/data_packing.py --input ./data/toy.jsonl --output ./data/packed_toy --seq_len 2048 --k 8

# Train autoencoder
# Usage: make train-ae K=8 D=1024 EPOCHS=1
K ?= 8
D ?= 1024
EPOCHS ?= 1
DATA ?= ./data/packed_toy/train
BATCH_SIZE ?= 32

train-ae:
	@echo "Training autoencoder..."
	@echo "  K=$(K)"
	@echo "  Latent dim=$(D)"
	@echo "  Epochs=$(EPOCHS)"
	@echo "  Data=$(DATA)"
	python ae/train_ae.py \
		--data $(DATA) \
		--k $(K) \
		--latent_dim $(D) \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--bf16

# Train student
# Usage: make train-student KD_W=1.0 MSE_W=1.0 CE_W=1.0
KD_W ?= 1.0
MSE_W ?= 1.0
CE_W ?= 1.0
STEPS ?= 50
AE_CKPT ?= checkpoints/ae.pt

train-student:
	@echo "Training student with KD..."
	@echo "  AE checkpoint=$(AE_CKPT)"
	@echo "  KD weight=$(KD_W)"
	@echo "  MSE weight=$(MSE_W)"
	@echo "  CE weight=$(CE_W)"
	@echo "  Steps=$(STEPS)"
	python student/train_student.py \
		--data $(DATA) \
		--ae_ckpt $(AE_CKPT) \
		--k $(K) \
		--latent_dim $(D) \
		--kd_w $(KD_W) \
		--mse_w $(MSE_W) \
		--ce_w $(CE_W) \
		--epochs 1 \
		--bf16

# Inference
PROMPT ?= "Smoke test"
MAX_TOKENS ?= 32
TEMP ?= 0.8
TOP_P ?= 0.95
SEED ?= 0
STUDENT_CKPT ?= checkpoints/student.pt

infer:
	@echo "Running inference..."
	python infer.py \
		--ae $(AE_CKPT) \
		--student $(STUDENT_CKPT) \
		--prompt "$(PROMPT)" \
		--max_new_tokens $(MAX_TOKENS) \
		--temperature $(TEMP) \
		--top_p $(TOP_P) \
		--seed $(SEED)

# Serve
PORT ?= 7860
serve:
	@echo "Starting FastAPI server on port $(PORT)..."
	uvicorn server:app --port $(PORT)

# Benchmark
PROMPTS_FILE ?= prompts.txt
bench:
	@echo "Running benchmark..."
	python bench.py \
		--ae $(AE_CKPT) \
		--student $(STUDENT_CKPT) \
		--prompts $(PROMPTS_FILE) \
		--k $(K)

# Tests
test:
	@echo "Running all tests..."
	pytest -q

test-fast:
	@echo "Running fast unit tests..."
	pytest -q -k "unit or env or kd_api or pack" --maxfail=1

test-e2e:
	@echo "Running end-to-end tests..."
	pytest -q -k "e2e" --timeout=120

# Clean
clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned"

# Development helpers
.PHONY: test-tokenizer test-ae test-student test-kd

test-tokenizer:
	python ae/tokenizer_adapter.py

test-ae:
	python ae/ae_model.py

test-student:
	python student/student_model.py

test-kd:
	python kd/kd_client.py
