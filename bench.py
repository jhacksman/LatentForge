#!/usr/bin/env python3
"""
Benchmark LatentForge throughput and compare with baseline token-level LLM.

Measures:
- Tokens per second
- Latency per token
- AR steps reduction (should be ~K×)
- Coherence on eval prompts

Saves results to JSON.

Usage:
    python bench.py \\
        --ae checkpoints/ae \\
        --student checkpoints/student \\
        --teacher meta-llama/Llama-3.2-1B \\
        --output benchmark_results.json \\
        --num_samples 20 \\
        --max_new_tokens 128
"""
import argparse
import json
import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ae'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'student'))

from ae_model import AutoEncoder
from student_model import StudentModel
from sampler import generate_latent_ar


# Default evaluation prompts
DEFAULT_PROMPTS = [
    "Once upon a time",
    "The quick brown fox",
    "In the year 2050,",
    "Artificial intelligence is",
    "The most important thing in life is",
    "Scientists have discovered",
    "The future of humanity",
    "In a galaxy far away",
    "The secret to happiness",
    "Climate change is",
    "Technology has changed",
    "The best way to learn",
    "In the beginning,",
    "Life is like",
    "The meaning of existence",
    "In the distant future,",
    "Once there was",
    "The power of",
    "Throughout history,",
    "At the end of the day,",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LatentForge")
    parser.add_argument("--ae", type=str, required=True, help="Path to autoencoder")
    parser.add_argument("--student", type=str, required=True, help="Path to student model")
    parser.add_argument("--teacher", type=str, required=True, help="Teacher/baseline model for comparison")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON file")

    # Benchmark parameters
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Tokens per sample")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling")
    parser.add_argument("--prompts_file", type=str, default=None, help="File with custom prompts (one per line)")

    # System
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def load_prompts(prompts_file: str, num_samples: int) -> List[str]:
    """Load evaluation prompts."""
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts[:num_samples]
    else:
        return DEFAULT_PROMPTS[:num_samples]


def load_latentforge(ae_path, student_path, device, bf16):
    """Load LatentForge models."""
    print("Loading LatentForge models...")

    # Load AE
    with open(os.path.join(ae_path, "config.json"), 'r') as f:
        ae_config = json.load(f)
    ae_model = AutoEncoder(**ae_config)
    ae_model.load_state_dict(torch.load(os.path.join(ae_path, "model.pt"), map_location=device))
    if bf16 and device == 'cuda':
        ae_model = ae_model.to(torch.bfloat16)
    ae_model = ae_model.to(device)
    ae_model.eval()

    # Load student
    with open(os.path.join(student_path, "config.json"), 'r') as f:
        student_config = json.load(f)
    student_model = StudentModel(**student_config)
    student_model.load_state_dict(torch.load(os.path.join(student_path, "model.pt"), map_location=device))
    if bf16 and device == 'cuda':
        student_model = student_model.to(torch.bfloat16)
    student_model = student_model.to(device)
    student_model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ae_path)

    print(f"LatentForge loaded: K={ae_model.K}, D={ae_model.D}")
    return ae_model, student_model, tokenizer


def load_baseline(teacher_model, device, bf16):
    """Load baseline token-level model."""
    print(f"Loading baseline model: {teacher_model}")
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map=device,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(teacher_model)
    return model, tokenizer


def benchmark_latentforge(ae_model, student_model, tokenizer, prompts, args):
    """Benchmark LatentForge generation."""
    print("\nBenchmarking LatentForge...")

    total_tokens = 0
    total_time = 0
    total_ar_steps = 0
    generations = []

    K = ae_model.K

    for i, prompt in enumerate(prompts):
        print(f"  {i+1}/{len(prompts)}: {prompt[:50]}...")

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)

        torch.cuda.synchronize() if args.device == 'cuda' else None
        start_time = time.time()

        with torch.no_grad():
            output_ids = generate_latent_ar(
                student_model,
                ae_model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id or 2,
            )

        torch.cuda.synchronize() if args.device == 'cuda' else None
        elapsed = time.time() - start_time

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        num_tokens = output_ids.shape[1] - input_ids.shape[1]
        num_ar_steps = (num_tokens + K - 1) // K  # Latent steps

        total_tokens += num_tokens
        total_time += elapsed
        total_ar_steps += num_ar_steps

        generations.append({
            'prompt': prompt,
            'generated': generated_text,
            'num_tokens': num_tokens,
            'time': elapsed,
            'ar_steps': num_ar_steps,
        })

    avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    avg_latency_per_token = (total_time / total_tokens * 1000) if total_tokens > 0 else 0
    avg_ar_steps = total_ar_steps / len(prompts)

    return {
        'total_tokens': total_tokens,
        'total_time': total_time,
        'tokens_per_second': avg_tokens_per_sec,
        'latency_per_token_ms': avg_latency_per_token,
        'avg_ar_steps': avg_ar_steps,
        'compression_factor': K,
        'generations': generations,
    }


def benchmark_baseline(model, tokenizer, prompts, args):
    """Benchmark baseline token-level generation."""
    print("\nBenchmarking baseline model...")

    total_tokens = 0
    total_time = 0
    generations = []

    for i, prompt in enumerate(prompts):
        print(f"  {i+1}/{len(prompts)}: {prompt[:50]}...")

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)

        torch.cuda.synchronize() if args.device == 'cuda' else None
        start_time = time.time()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True if args.temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize() if args.device == 'cuda' else None
        elapsed = time.time() - start_time

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        num_tokens = output_ids.shape[1] - input_ids.shape[1]

        total_tokens += num_tokens
        total_time += elapsed

        generations.append({
            'prompt': prompt,
            'generated': generated_text,
            'num_tokens': num_tokens,
            'time': elapsed,
        })

    avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    avg_latency_per_token = (total_time / total_tokens * 1000) if total_tokens > 0 else 0

    return {
        'total_tokens': total_tokens,
        'total_time': total_time,
        'tokens_per_second': avg_tokens_per_sec,
        'latency_per_token_ms': avg_latency_per_token,
        'avg_ar_steps': total_tokens,  # Token-level: 1 step per token
        'generations': generations,
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
        args.bf16 = False

    # Load prompts
    prompts = load_prompts(args.prompts_file, args.num_samples)
    print(f"Loaded {len(prompts)} prompts")

    # Load models
    ae_model, student_model, lf_tokenizer = load_latentforge(args.ae, args.student, args.device, args.bf16)
    baseline_model, baseline_tokenizer = load_baseline(args.teacher, args.device, args.bf16)

    # Benchmark LatentForge
    lf_results = benchmark_latentforge(ae_model, student_model, lf_tokenizer, prompts, args)

    # Benchmark baseline
    baseline_results = benchmark_baseline(baseline_model, baseline_tokenizer, prompts, args)

    # Compute speedup
    speedup = (baseline_results['avg_ar_steps'] / lf_results['avg_ar_steps']) if lf_results['avg_ar_steps'] > 0 else 0
    throughput_ratio = lf_results['tokens_per_second'] / baseline_results['tokens_per_second'] if baseline_results['tokens_per_second'] > 0 else 0

    # Compile results
    results = {
        'config': {
            'num_samples': len(prompts),
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'device': args.device,
            'bf16': args.bf16,
            'seed': args.seed,
        },
        'latentforge': lf_results,
        'baseline': baseline_results,
        'comparison': {
            'ar_steps_reduction': speedup,
            'theoretical_speedup': lf_results['compression_factor'],
            'throughput_ratio': throughput_ratio,
        }
    }

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\nLatentForge:")
    print(f"  Tokens/sec: {lf_results['tokens_per_second']:.2f}")
    print(f"  Latency/token: {lf_results['latency_per_token_ms']:.2f} ms")
    print(f"  Avg AR steps: {lf_results['avg_ar_steps']:.1f}")
    print(f"  Compression: {lf_results['compression_factor']}×")

    print(f"\nBaseline:")
    print(f"  Tokens/sec: {baseline_results['tokens_per_second']:.2f}")
    print(f"  Latency/token: {baseline_results['latency_per_token_ms']:.2f} ms")
    print(f"  Avg AR steps: {baseline_results['avg_ar_steps']:.1f}")

    print(f"\nComparison:")
    print(f"  AR steps reduction: {speedup:.2f}×")
    print(f"  Theoretical speedup: {lf_results['compression_factor']}×")
    print(f"  Throughput ratio: {throughput_ratio:.2f}×")

    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
