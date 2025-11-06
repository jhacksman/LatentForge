#!/usr/bin/env python3
"""
Benchmark script for LatentForge.
Measures latent steps per second and decoded tokens per second.
"""
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict
import torch

from student.sampler import LatentSampler, load_models

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# Default benchmark prompts
DEFAULT_PROMPTS = [
    "Write a function to calculate fibonacci numbers",
    "Explain the concept of neural networks in simple terms",
    "Create a recipe for chocolate chip cookies",
    "Describe the process of photosynthesis",
    "Write a short story about a robot learning to paint",
]


def benchmark_generation(
    sampler: LatentSampler,
    prompts: List[str],
    max_new_tokens: int = 128,
    num_runs: int = 3,
) -> Dict:
    """
    Benchmark generation performance.

    Args:
        sampler: LatentSampler instance
        prompts: List of prompts to benchmark
        max_new_tokens: Maximum new tokens per prompt
        num_runs: Number of runs per prompt

    Returns:
        Benchmark results dictionary
    """
    results = {
        "prompts": [],
        "summary": {},
    }

    total_latent_steps = 0
    total_tokens = 0
    total_time = 0.0

    print(f"\n{'='*60}")
    print(f"Running benchmark...")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Runs per prompt: {num_runs}")
    print(f"  K (compression): {sampler.k}")
    print(f"{'='*60}\n")

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nPrompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")

        prompt_results = []

        for run in range(num_runs):
            # Warmup on first run
            if run == 0 and prompt_idx == 0:
                print("  [Warmup run]")
                _ = sampler.sample(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens // 2,
                    temperature=0.8,
                    seed=42,
                )

            # Benchmark run
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()

            generated = sampler.sample(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                seed=42 + run,
            )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()

            elapsed = end_time - start_time

            # Calculate metrics
            num_latent_steps = max_new_tokens // sampler.k
            num_tokens = max_new_tokens

            latent_steps_per_sec = num_latent_steps / elapsed
            tokens_per_sec = num_tokens / elapsed

            run_result = {
                "run": run + 1,
                "elapsed_time": elapsed,
                "latent_steps": num_latent_steps,
                "tokens": num_tokens,
                "latent_steps_per_sec": latent_steps_per_sec,
                "tokens_per_sec": tokens_per_sec,
                "generated_length": len(generated),
            }

            prompt_results.append(run_result)

            # Accumulate for summary
            total_latent_steps += num_latent_steps
            total_tokens += num_tokens
            total_time += elapsed

            print(
                f"  Run {run + 1}: {elapsed:.3f}s | "
                f"Latent steps/s: {latent_steps_per_sec:.2f} | "
                f"Tokens/s: {tokens_per_sec:.2f}"
            )

        # Average for this prompt
        avg_latent_steps_per_sec = sum(r["latent_steps_per_sec"] for r in prompt_results) / num_runs
        avg_tokens_per_sec = sum(r["tokens_per_sec"] for r in prompt_results) / num_runs

        results["prompts"].append(
            {
                "prompt": prompt,
                "runs": prompt_results,
                "avg_latent_steps_per_sec": avg_latent_steps_per_sec,
                "avg_tokens_per_sec": avg_tokens_per_sec,
            }
        )

    # Overall summary
    avg_latent_steps_per_sec = total_latent_steps / total_time
    avg_tokens_per_sec = total_tokens / total_time
    compression_ratio = sampler.k

    results["summary"] = {
        "total_prompts": len(prompts),
        "total_runs": len(prompts) * num_runs,
        "total_latent_steps": total_latent_steps,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "avg_latent_steps_per_sec": avg_latent_steps_per_sec,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "compression_ratio": compression_ratio,
        "speedup_factor": f"{compression_ratio}x fewer AR steps vs token LLM",
    }

    return results


def print_summary(results: Dict):
    """Print benchmark summary."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*60}")

    summary = results["summary"]
    print(f"\nOverall Performance:")
    print(f"  Total prompts: {summary['total_prompts']}")
    print(f"  Total runs: {summary['total_runs']}")
    print(f"  Total time: {summary['total_time']:.2f}s")
    print(f"\nThroughput:")
    print(f"  Latent steps/sec: {summary['avg_latent_steps_per_sec']:.2f}")
    print(f"  Tokens/sec: {summary['avg_tokens_per_sec']:.2f}")
    print(f"\nCompression:")
    print(f"  K (compression ratio): {summary['compression_ratio']}")
    print(f"  AR steps reduction: {summary['speedup_factor']}")

    print(f"\n{'='*60}\n")

    # Print table of results
    if HAS_TABULATE:
        print("Per-Prompt Results:")
        table_data = []
        for prompt_result in results["prompts"]:
            prompt_short = prompt_result["prompt"][:40] + "..." if len(prompt_result["prompt"]) > 40 else prompt_result["prompt"]
            table_data.append([
                prompt_short,
                f"{prompt_result['avg_latent_steps_per_sec']:.2f}",
                f"{prompt_result['avg_tokens_per_sec']:.2f}",
            ])

        headers = ["Prompt", "Latent Steps/s", "Tokens/s"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()
    else:
        print("(Install 'tabulate' for formatted tables)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark LatentForge")
    parser.add_argument("--ae", type=str, required=True, help="AE checkpoint")
    parser.add_argument("--student", type=str, required=True, help="Student checkpoint")
    parser.add_argument("--prompts", type=str, nargs="+", help="Custom prompts")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--num_runs", type=int, default=3, help="Runs per prompt")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print(f"Loading models...")
    student, autoencoder, tokenizer = load_models(
        ae_checkpoint=args.ae,
        student_checkpoint=args.student,
        device=device,
    )
    print(f"✅ Models loaded")

    # Create sampler
    sampler = LatentSampler(
        student=student,
        autoencoder=autoencoder,
        tokenizer=tokenizer,
        device=device,
    )

    # Prompts - check if it's a file path
    if args.prompts:
        # Check if first argument is a file
        if len(args.prompts) == 1 and Path(args.prompts[0]).exists():
            prompts_file = Path(args.prompts[0])
            print(f"Loading prompts from: {prompts_file}")
            with open(prompts_file, "r") as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            prompts = args.prompts
    else:
        prompts = DEFAULT_PROMPTS

    print(f"Benchmarking {len(prompts)} prompts")

    # Run benchmark
    results = benchmark_generation(
        sampler=sampler,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        num_runs=args.num_runs,
    )

    # Print summary
    print_summary(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")
    else:
        # Default save location
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"bench_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
