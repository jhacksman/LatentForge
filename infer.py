#!/usr/bin/env python3
"""
End-to-end inference via latent autoregression.

Usage:
    python infer.py \\
        --ae checkpoints/ae \\
        --student checkpoints/student \\
        --prompt "Once upon a time" \\
        --max_new_tokens 128 \\
        --temperature 0.8 \\
        --top_p 0.9 \\
        --seed 42
"""
import argparse
import json
import os
import sys
import torch
from transformers import AutoTokenizer

# Add ae and student to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ae'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'student'))

from ae_model import AutoEncoder
from student_model import StudentModel
from sampler import generate_latent_ar, generate_with_kv_cache


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using LatentForge")
    parser.add_argument("--ae", type=str, required=True, help="Path to autoencoder checkpoint")
    parser.add_argument("--student", type=str, required=True, help="Path to student checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum new tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k filtering (0=disabled)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling (1.0=disabled)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # System
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--use_kv_cache", action="store_true", help="Use KV caching")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")

    return parser.parse_args()


def load_models(ae_path, student_path, device, bf16):
    """Load autoencoder and student models."""
    print(f"Loading models from:")
    print(f"  AE: {ae_path}")
    print(f"  Student: {student_path}")

    # Load AE config and model
    with open(os.path.join(ae_path, "config.json"), 'r') as f:
        ae_config = json.load(f)

    ae_model = AutoEncoder(**ae_config)
    ae_model.load_state_dict(torch.load(
        os.path.join(ae_path, "model.pt"),
        map_location=device
    ))

    if bf16 and device == 'cuda':
        ae_model = ae_model.to(torch.bfloat16)
    ae_model = ae_model.to(device)
    ae_model.eval()

    # Load student config and model
    with open(os.path.join(student_path, "config.json"), 'r') as f:
        student_config = json.load(f)

    student_model = StudentModel(**student_config)
    student_model.load_state_dict(torch.load(
        os.path.join(student_path, "model.pt"),
        map_location=device
    ))

    if bf16 and device == 'cuda':
        student_model = student_model.to(torch.bfloat16)
    student_model = student_model.to(device)
    student_model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ae_path)

    print(f"Models loaded successfully!")
    print(f"  AE: K={ae_config['K']}, D={ae_config['D']}")
    print(f"  Student: {student_config['num_layers']} layers, {student_config['hidden_size']} hidden")

    return ae_model, student_model, tokenizer


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
        args.bf16 = False

    # Load models
    ae_model, student_model, tokenizer = load_models(
        args.ae,
        args.student,
        args.device,
        args.bf16
    )

    # Tokenize prompt
    print(f"\nPrompt: {args.prompt}")
    input_ids = tokenizer.encode(args.prompt, return_tensors='pt').to(args.device)
    print(f"Input tokens: {input_ids.shape[1]}")

    # Generate
    print(f"\nGenerating {args.max_new_tokens} tokens...")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}, Top-k: {args.top_k}")

    with torch.no_grad():
        if args.use_kv_cache:
            output_ids = generate_with_kv_cache(
                student_model,
                ae_model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id or 2,
            )
        else:
            output_ids = generate_latent_ar(
                student_model,
                ae_model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id or 2,
            )

    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Print results
    print("\n" + "=" * 80)
    print("GENERATED TEXT:")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)

    # Stats
    num_generated = output_ids.shape[1] - input_ids.shape[1]
    print(f"\nGenerated {num_generated} tokens")
    print(f"Compression factor: ~{ae_model.K}× fewer AR steps")


if __name__ == "__main__":
    main()
