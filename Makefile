.PHONY: help setup train-ae train-student infer serve bench clean

# Default values
DATA ?= data.txt
TEACHER ?= meta-llama/Llama-3.2-1B
K ?= 8
D ?= 1024
EPOCHS ?= 2
KD_W ?= 1.0
MSE_W ?= 1.0
CE_W ?= 1.0
AE_PATH ?= checkpoints/ae
STUDENT_PATH ?= checkpoints/student
PROMPT ?= "Once upon a time"
MAX_NEW_TOKENS ?= 128
PORT ?= 7860

help:
	@echo "LatentForge - Makefile commands:"
	@echo ""
	@echo "  make setup              - Install dependencies"
	@echo "  make train-ae           - Train autoencoder"
	@echo "  make train-student      - Train student model with KD"
	@echo "  make infer              - Run inference"
	@echo "  make serve              - Start FastAPI server"
	@echo "  make bench              - Run benchmarks"
	@echo "  make clean              - Clean checkpoints"
	@echo ""
	@echo "Variables (use with make VARIABLE=value):"
	@echo "  DATA=<path>             - Training data path"
	@echo "  TEACHER=<model>         - Teacher model (default: meta-llama/Llama-3.2-1B)"
	@echo "  K=<int>                 - Patch size (default: 8)"
	@echo "  D=<int>                 - Latent dimension (default: 1024)"
	@echo "  EPOCHS=<int>            - Training epochs (default: 2)"
	@echo "  KD_W=<float>            - KD loss weight (default: 1.0)"
	@echo "  MSE_W=<float>           - MSE loss weight (default: 1.0)"
	@echo "  CE_W=<float>            - CE loss weight (default: 1.0)"
	@echo "  PROMPT=<text>           - Inference prompt"
	@echo "  MAX_NEW_TOKENS=<int>    - Max tokens to generate (default: 128)"
	@echo "  PORT=<int>              - Server port (default: 7860)"
	@echo ""
	@echo "Examples:"
	@echo "  make train-ae DATA=mydata.txt K=8 D=1024 EPOCHS=2"
	@echo "  make train-student DATA=mydata.txt TEACHER=meta-llama/Llama-3.2-1B"
	@echo "  make infer PROMPT=\"Hello world\" MAX_NEW_TOKENS=256"
	@echo "  make serve PORT=8000"

setup:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Setup complete!"

train-ae:
	@echo "Training autoencoder with K=$(K), D=$(D), EPOCHS=$(EPOCHS)"
	@if [ -z "$(DATA)" ] || [ "$(DATA)" = "data.txt" ]; then \
		echo "ERROR: DATA parameter is required. Usage: make train-ae DATA=<path>"; \
		exit 1; \
	fi
	python ae/train_ae.py \
		--data_path $(DATA) \
		--output_dir $(AE_PATH) \
		--K $(K) \
		--D $(D) \
		--epochs $(EPOCHS) \
		--batch_size 8 \
		--gradient_accumulation_steps 4 \
		--learning_rate 3e-4 \
		--block_size 2048 \
		--bf16

train-student:
	@echo "Training student model with KD_W=$(KD_W), MSE_W=$(MSE_W), CE_W=$(CE_W)"
	@if [ -z "$(DATA)" ] || [ "$(DATA)" = "data.txt" ]; then \
		echo "ERROR: DATA parameter is required. Usage: make train-student DATA=<path>"; \
		exit 1; \
	fi
	@if [ ! -d "$(AE_PATH)" ]; then \
		echo "ERROR: Autoencoder not found at $(AE_PATH). Train AE first with 'make train-ae'"; \
		exit 1; \
	fi
	python student/train_student.py \
		--data_path $(DATA) \
		--ae_path $(AE_PATH) \
		--teacher_model $(TEACHER) \
		--output_dir $(STUDENT_PATH) \
		--K $(K) \
		--KD_W $(KD_W) \
		--MSE_W $(MSE_W) \
		--CE_W $(CE_W) \
		--batch_size 4 \
		--gradient_accumulation_steps 8 \
		--learning_rate 3e-4 \
		--block_size 2048 \
		--bf16

infer:
	@echo "Running inference with prompt: $(PROMPT)"
	@if [ ! -d "$(AE_PATH)" ] || [ ! -d "$(STUDENT_PATH)" ]; then \
		echo "ERROR: Models not found. Train models first."; \
		exit 1; \
	fi
	python infer.py \
		--ae $(AE_PATH) \
		--student $(STUDENT_PATH) \
		--prompt $(PROMPT) \
		--max_new_tokens $(MAX_NEW_TOKENS) \
		--temperature 0.8 \
		--top_p 0.9 \
		--bf16

serve:
	@echo "Starting FastAPI server on port $(PORT)"
	@if [ ! -d "$(AE_PATH)" ] || [ ! -d "$(STUDENT_PATH)" ]; then \
		echo "ERROR: Models not found. Train models first."; \
		exit 1; \
	fi
	AE_PATH=$(AE_PATH) STUDENT_PATH=$(STUDENT_PATH) \
	uvicorn server:app --host 0.0.0.0 --port $(PORT)

bench:
	@echo "Running benchmarks..."
	@if [ ! -d "$(AE_PATH)" ] || [ ! -d "$(STUDENT_PATH)" ]; then \
		echo "ERROR: Models not found. Train models first."; \
		exit 1; \
	fi
	python bench.py \
		--ae $(AE_PATH) \
		--student $(STUDENT_PATH) \
		--teacher $(TEACHER) \
		--output benchmark_results.json \
		--num_samples 20 \
		--max_new_tokens $(MAX_NEW_TOKENS) \
		--bf16

clean:
	@echo "Cleaning checkpoints..."
	rm -rf checkpoints/
	rm -f benchmark_results.json
	@echo "Clean complete!"
