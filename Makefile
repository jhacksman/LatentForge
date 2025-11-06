.PHONY: help setup check-api train-ae train-student infer serve bench clean

# Default target
help:
	@echo "LatentForge - Makefile targets:"
	@echo ""
	@echo "  setup           - Create venv, install dependencies, create basic config"
	@echo "  check-api       - Verify Venice API connection"
	@echo "  train-ae        - Train autoencoder (use K=8 D=1024 EPOCHS=10)"
	@echo "  train-student   - Train student with KD"
	@echo "  infer           - Run inference"
	@echo "  serve           - Start FastAPI server"
	@echo "  bench           - Run benchmark"
	@echo "  clean           - Clean generated files"
	@echo ""
	@echo "Example usage:"
	@echo "  make setup"
	@echo "  make check-api"
	@echo "  make train-ae K=8 D=1024 EPOCHS=2"
	@echo "  make train-student"
	@echo ""

# Setup environment
setup:
	@echo "Setting up LatentForge..."
	python3 -m venv .venv || true
	@echo ""
	@echo "✅ Virtual environment created"
	@echo ""
	@echo "To activate:"
	@echo "  source .venv/bin/activate"
	@echo ""
	@echo "Then install dependencies:"
	@echo "  pip install -r requirements.txt"
	@echo ""
	@if [ ! -f .env ]; then \
		echo "⚠️  .env file not found. Please create it with your Venice API credentials."; \
	else \
		echo "✅ .env file exists"; \
	fi
	@mkdir -p checkpoints results data configs
	@echo "✅ Directories created"

# Verify API
check-api:
	@echo "Verifying Venice API..."
	python tools/verify_api.py

# Train autoencoder
# Usage: make train-ae K=8 D=1024 EPOCHS=10 DATA=/path/to/data
K ?= 8
D ?= 1024
EPOCHS ?= 10
DATA ?= data/packed/train
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
STUDENT_EPOCHS ?= 5
AE_CKPT ?= checkpoints/ae.pt

train-student:
	@echo "Training student with KD..."
	@echo "  AE checkpoint=$(AE_CKPT)"
	@echo "  KD weight=$(KD_W)"
	@echo "  MSE weight=$(MSE_W)"
	@echo "  CE weight=$(CE_W)"
	python student/train_student.py \
		--data $(DATA) \
		--ae_ckpt $(AE_CKPT) \
		--k $(K) \
		--latent_dim $(D) \
		--kd_w $(KD_W) \
		--mse_w $(MSE_W) \
		--ce_w $(CE_W) \
		--epochs $(STUDENT_EPOCHS) \
		--use_kd \
		--bf16

# Inference
PROMPT ?= "Write a short function"
MAX_TOKENS ?= 128
TEMP ?= 0.8
TOP_P ?= 0.95
STUDENT_CKPT ?= checkpoints/student.pt

infer:
	@echo "Running inference..."
	python infer.py \
		--ae $(AE_CKPT) \
		--student $(STUDENT_CKPT) \
		--prompt "$(PROMPT)" \
		--max_new_tokens $(MAX_TOKENS) \
		--temperature $(TEMP) \
		--top_p $(TOP_P)

# Serve
PORT ?= 7860
serve:
	@echo "Starting FastAPI server on port $(PORT)..."
	python server.py \
		--port $(PORT) \
		--ae $(AE_CKPT) \
		--student $(STUDENT_CKPT)

# Benchmark
bench:
	@echo "Running benchmark..."
	python bench.py \
		--ae $(AE_CKPT) \
		--student $(STUDENT_CKPT) \
		--max_new_tokens 128

# Clean
clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
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
