#!/usr/bin/env python3
"""
FastAPI server for LatentForge text generation.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 7860

    or with custom models:

    AE_PATH=checkpoints/ae STUDENT_PATH=checkpoints/student uvicorn server:app --port 7860
"""
import os
import sys
import json
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from transformers import AutoTokenizer

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ae'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'student'))

from ae_model import AutoEncoder
from student_model import StudentModel
from sampler import generate_latent_ar, generate_with_kv_cache


# Configuration from environment variables
AE_PATH = os.getenv("AE_PATH", "checkpoints/ae")
STUDENT_PATH = os.getenv("STUDENT_PATH", "checkpoints/student")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = os.getenv("USE_BF16", "true").lower() == "true"


# Request/response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_new_tokens: int = Field(128, ge=1, le=2048, description="Maximum new tokens to generate")
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature (0=greedy)")
    top_k: int = Field(0, ge=0, description="Top-k filtering (0=disabled)")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling (1.0=disabled)")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    use_kv_cache: bool = Field(True, description="Use KV caching for efficiency")


class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    num_tokens_generated: int
    compression_factor: int


# Global model instances
ae_model = None
student_model = None
tokenizer = None


def load_models():
    """Load models on startup."""
    global ae_model, student_model, tokenizer

    print(f"Loading models from:")
    print(f"  AE: {AE_PATH}")
    print(f"  Student: {STUDENT_PATH}")
    print(f"  Device: {DEVICE}")
    print(f"  BF16: {USE_BF16}")

    # Load AE
    with open(os.path.join(AE_PATH, "config.json"), 'r') as f:
        ae_config = json.load(f)

    ae_model = AutoEncoder(**ae_config)
    ae_model.load_state_dict(torch.load(
        os.path.join(AE_PATH, "model.pt"),
        map_location=DEVICE
    ))

    if USE_BF16 and DEVICE == 'cuda':
        ae_model = ae_model.to(torch.bfloat16)
    ae_model = ae_model.to(DEVICE)
    ae_model.eval()

    # Load student
    with open(os.path.join(STUDENT_PATH, "config.json"), 'r') as f:
        student_config = json.load(f)

    student_model = StudentModel(**student_config)
    student_model.load_state_dict(torch.load(
        os.path.join(STUDENT_PATH, "model.pt"),
        map_location=DEVICE
    ))

    if USE_BF16 and DEVICE == 'cuda':
        student_model = student_model.to(torch.bfloat16)
    student_model = student_model.to(DEVICE)
    student_model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(AE_PATH)

    print("Models loaded successfully!")

    return ae_model, student_model, tokenizer


# Create FastAPI app
app = FastAPI(
    title="LatentForge API",
    description="Text generation using latent-space autoregressive modeling (CALM-style)",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "LatentForge",
        "status": "running",
        "models_loaded": ae_model is not None and student_model is not None,
        "device": DEVICE,
        "compression_factor": ae_model.K if ae_model else None,
    }


@app.get("/info")
async def info():
    """Get model information."""
    if ae_model is None or student_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return {
        "autoencoder": {
            "K": ae_model.K,
            "D": ae_model.D,
            "hidden_size": ae_model.hidden_size,
            "vocab_size": ae_model.vocab_size,
        },
        "student": {
            "latent_dim": student_model.latent_dim,
            "hidden_size": student_model.hidden_size,
            "num_layers": student_model.num_layers,
        },
        "device": DEVICE,
        "bf16": USE_BF16,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text from a prompt.

    This endpoint uses latent-space autoregression to generate text
    with ~K× fewer autoregressive steps than token-level models.
    """
    if ae_model is None or student_model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Set seed if provided
    if request.seed is not None:
        torch.manual_seed(request.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.seed)

    try:
        # Tokenize
        input_ids = tokenizer.encode(request.prompt, return_tensors='pt').to(DEVICE)

        # Generate
        with torch.no_grad():
            if request.use_kv_cache:
                output_ids = generate_with_kv_cache(
                    student_model,
                    ae_model,
                    input_ids,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    eos_token_id=tokenizer.eos_token_id or 2,
                )
            else:
                output_ids = generate_latent_ar(
                    student_model,
                    ae_model,
                    input_ids,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    eos_token_id=tokenizer.eos_token_id or 2,
                )

        # Decode
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        num_generated = output_ids.shape[1] - input_ids.shape[1]

        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            num_tokens_generated=num_generated,
            compression_factor=ae_model.K,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
