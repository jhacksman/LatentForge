#!/usr/bin/env python3
"""
FastAPI server for LatentForge inference.
"""
import os
import argparse
from typing import Optional
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from student.sampler import LatentSampler, load_models


class GenerateRequest(BaseModel):
    """Request model for generation."""

    prompt: str = Field(..., description="Input prompt")
    max_new_tokens: int = Field(128, description="Maximum new tokens", ge=1, le=2048)
    temperature: float = Field(0.8, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(0.95, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    top_k: int = Field(0, description="Top-k sampling parameter", ge=0)
    seed: Optional[int] = Field(None, description="Random seed")


class GenerateResponse(BaseModel):
    """Response model for generation."""

    generated_text: str
    prompt: str
    num_tokens: Optional[int] = None
    model_info: dict


# Global state
app = FastAPI(
    title="LatentForge API",
    description="Text generation with latent autoregressive models",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sampler: Optional[LatentSampler] = None
model_config: dict = {}


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global sampler, model_config

    # Get model paths from environment or default
    ae_path = os.getenv("AE_CHECKPOINT", "checkpoints/ae.pt")
    student_path = os.getenv("STUDENT_CHECKPOINT", "checkpoints/student.pt")
    device_name = os.getenv("DEVICE", "cuda")

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    print(f"Loading models...")
    print(f"  AE: {ae_path}")
    print(f"  Student: {student_path}")
    print(f"  Device: {device}")

    try:
        student, autoencoder, tokenizer = load_models(
            ae_checkpoint=ae_path,
            student_checkpoint=student_path,
            device=device,
        )

        sampler = LatentSampler(
            student=student,
            autoencoder=autoencoder,
            tokenizer=tokenizer,
            device=device,
        )

        model_config = {
            "k": autoencoder.k,
            "latent_dim": autoencoder.latent_dim,
            "student_layers": student.num_layers,
            "device": str(device),
        }

        print(f"✅ Models loaded successfully!")
        print(f"  K={autoencoder.k}")
        print(f"  Latent dim={autoencoder.latent_dim}")

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Server will start but generation will fail until models are loaded.")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "LatentForge API",
        "version": "1.0.0",
        "status": "running" if sampler is not None else "models_not_loaded",
        "model_config": model_config,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if sampler is not None else "unhealthy",
        "models_loaded": sampler is not None,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text from prompt.

    Args:
        request: Generation request

    Returns:
        Generated text response
    """
    if sampler is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check server startup logs.",
        )

    try:
        generated_text = sampler.sample(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            seed=request.seed,
        )

        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            model_info=model_config,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}",
        )


def main():
    """Run the server."""
    parser = argparse.ArgumentParser(description="LatentForge API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=7860, help="Port")
    parser.add_argument("--ae", type=str, help="AE checkpoint path")
    parser.add_argument("--student", type=str, help="Student checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    # Set environment variables for startup event
    if args.ae:
        os.environ["AE_CHECKPOINT"] = args.ae
    if args.student:
        os.environ["STUDENT_CHECKPOINT"] = args.student
    if args.device:
        os.environ["DEVICE"] = args.device

    print(f"Starting LatentForge API server...")
    print(f"Server will be available at http://{args.host}:{args.port}")
    print(f"API docs at http://{args.host}:{args.port}/docs")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
