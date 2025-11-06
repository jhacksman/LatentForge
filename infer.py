#!/usr/bin/env python3
"""
CLI for text generation with LatentForge.
"""
import argparse
import torch
from pathlib import Path

from student.sampler import LatentSampler, load_models


def main():
    parser = argparse.ArgumentParser(description="Generate text with LatentForge")
    parser.add_argument("--ae", type=str, required=True, help="AE checkpoint path")
    parser.add_argument("--student", type=str, required=True, help="Student checkpoint path")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling parameter")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print(f"Loading models...")
    print(f"  AE: {args.ae}")
    print(f"  Student: {args.student}")

    student, autoencoder, tokenizer = load_models(
        ae_checkpoint=args.ae,
        student_checkpoint=args.student,
        device=device,
    )

    print(f"Models loaded successfully!")
    print(f"  K={autoencoder.k}")
    print(f"  Latent dim={autoencoder.latent_dim}")

    # Create sampler
    sampler = LatentSampler(
        student=student,
        autoencoder=autoencoder,
        tokenizer=tokenizer,
        device=device,
    )

    # Generate
    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'='*60}\n")

    generated_text = sampler.sample(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

    print(generated_text)
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")


if __name__ == "__main__":
    main()
