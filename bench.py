#!/usr/bin/env python3
"""
Benchmark script for LatentForge.
Measures latent steps per second and decoded tokens per second.
Includes evaluation on toy dataset for reconstruction rate and KL to teacher.
"""
import argparse
import time
import json
import math
from pathlib import Path
from typing import List, Dict, Optional
import torch
import torch.nn.functional as F

from student.sampler import LatentSampler, load_models
from kd.kd_client import KDClient
from ae.tokenizer_adapter import TokenizerAdapter

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# GB10 Acceptance Gates
ACCEPTANCE_GATES = {
    "min_tokens_per_sec": 10.0,  # Minimum throughput
    "min_token_accuracy": 0.80,  # Minimum AE reconstruction accuracy (80%)
    "max_avg_kl_divergence": 5.0,  # Maximum KL to teacher
    "min_kd_cache_hit_rate": 0.50,  # Minimum cache hit rate after warmup (50%)
}


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
    Benchmark generation performance with percentile metrics.

    Args:
        sampler: LatentSampler instance
        prompts: List of prompts to benchmark
        max_new_tokens: Maximum new tokens per prompt
        num_runs: Number of runs per prompt

    Returns:
        Benchmark results dictionary with p50/p95 latencies
    """
    results = {
        "prompts": [],
        "summary": {},
    }

    total_latent_steps = 0
    total_tokens = 0
    total_time = 0.0
    all_latencies = []  # For percentile calculation
    all_tokens_per_sec = []  # For percentile calculation

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
            all_latencies.append(elapsed)
            all_tokens_per_sec.append(tokens_per_sec)

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

    # Overall summary with percentiles
    avg_latent_steps_per_sec = total_latent_steps / total_time
    avg_tokens_per_sec = total_tokens / total_time
    compression_ratio = sampler.k

    # Calculate percentiles
    all_latencies.sort()
    all_tokens_per_sec.sort()

    p50_latency = all_latencies[len(all_latencies) // 2] if all_latencies else 0
    p95_latency = all_latencies[int(len(all_latencies) * 0.95)] if all_latencies else 0
    p50_tokens_per_sec = all_tokens_per_sec[len(all_tokens_per_sec) // 2] if all_tokens_per_sec else 0
    p95_tokens_per_sec = all_tokens_per_sec[int(len(all_tokens_per_sec) * 0.95)] if all_tokens_per_sec else 0

    # GPU memory (if available)
    gpu_memory_stats = {}
    if torch.cuda.is_available():
        gpu_memory_stats = {
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "current_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        }

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
        "latency_p50": p50_latency,
        "latency_p95": p95_latency,
        "tokens_per_sec_p50": p50_tokens_per_sec,
        "tokens_per_sec_p95": p95_tokens_per_sec,
        "gpu_memory": gpu_memory_stats,
    }

    return results


def evaluate_on_toy_set(
    sampler: LatentSampler,
    data_path: str,
    kd_client: Optional[KDClient] = None,
    num_samples: int = 100,
    num_kd_samples: int = 10,
) -> Dict:
    """
    Evaluate on toy dataset.

    Args:
        sampler: LatentSampler instance
        data_path: Path to packed data directory
        kd_client: Optional KD client for teacher KL computation
        num_samples: Number of samples to evaluate reconstruction
        num_kd_samples: Number of samples to evaluate KL to teacher

    Returns:
        Evaluation results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on toy dataset...")
    print(f"{'='*60}\n")

    # Load toy data
    data_file = Path(data_path) / "sequences.pt"
    if not data_file.exists():
        print(f"⚠️  Data file not found: {data_file}")
        return {"error": "Data file not found"}

    sequences = torch.load(data_file)
    sequences = sequences[:num_samples]  # Limit samples

    # Metrics
    total_tokens = 0
    correct_tokens = 0
    total_chunks = 0
    exact_match_chunks = 0

    print(f"Evaluating reconstruction on {len(sequences)} sequences...")

    for seq in sequences:
        # Split into chunks of K
        k = sampler.k
        num_chunks = len(seq) // k

        for i in range(num_chunks):
            chunk = seq[i * k : (i + 1) * k]
            if len(chunk) < k:
                continue

            chunk_tensor = chunk.unsqueeze(0).to(sampler.device)  # (1, k)

            # Encode and decode
            with torch.no_grad():
                latent = sampler.autoencoder.encode(chunk_tensor)
                decoded_logits = sampler.autoencoder.decode(latent)  # (1, k, vocab_size)
                decoded_ids = decoded_logits.argmax(dim=-1).squeeze(0)  # (k,)

            # Count matches
            matches = (decoded_ids == chunk).sum().item()
            correct_tokens += matches
            total_tokens += k
            total_chunks += 1

            if matches == k:
                exact_match_chunks += 1

    # Reconstruction metrics
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    exact_match_rate = exact_match_chunks / total_chunks if total_chunks > 0 else 0.0

    results = {
        "reconstruction": {
            "total_tokens": total_tokens,
            "correct_tokens": correct_tokens,
            "token_accuracy": token_accuracy,
            "total_chunks": total_chunks,
            "exact_match_chunks": exact_match_chunks,
            "exact_match_rate": exact_match_rate,
        }
    }

    print(f"\nReconstruction Results:")
    print(f"  Token accuracy: {token_accuracy*100:.2f}%")
    print(f"  Exact match rate: {exact_match_rate*100:.2f}%")
    print(f"  Total chunks: {total_chunks}")

    # KL to teacher (optional, more expensive)
    if kd_client and num_kd_samples > 0:
        print(f"\nEvaluating KL to teacher on {num_kd_samples} samples...")

        kl_divergences = []
        tokenizer = TokenizerAdapter()

        for idx in range(min(num_kd_samples, len(sequences))):
            seq = sequences[idx]
            k = sampler.k

            # Take first chunk
            if len(seq) < k:
                continue

            chunk = seq[:k]
            chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)

            try:
                # Get teacher distribution
                teacher_output = kd_client.get_teacher_distribution(
                    prompt=chunk_text,
                    max_tokens=k,
                    temperature=1.0,
                    top_logprobs=20,
                )

                # Get student distribution via AE decode
                chunk_tensor = chunk.unsqueeze(0).to(sampler.device)
                with torch.no_grad():
                    latent = sampler.autoencoder.encode(chunk_tensor)
                    student_logits = sampler.autoencoder.decode(latent)  # (1, k, vocab_size)

                # Compute KL for each position where we have teacher probs
                for pos in range(min(k, len(teacher_output.normalized_probs))):
                    teacher_probs_dict = teacher_output.normalized_probs[pos]

                    if not teacher_probs_dict:
                        continue

                    # Get student probabilities for this position
                    student_probs = F.softmax(student_logits[0, pos, :], dim=-1)  # (vocab_size,)

                    # Compute KL divergence (sparse teacher, dense student)
                    kl = 0.0
                    for token_str, teacher_prob in teacher_probs_dict.items():
                        # Get token ID from teacher token string
                        # This is approximate - in production would need proper alignment
                        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
                        if len(token_ids) > 0:
                            token_id = token_ids[0]
                            student_prob = student_probs[token_id].item()

                            # KL(teacher || student) = sum(teacher * log(teacher / student))
                            if teacher_prob > 0 and student_prob > 0:
                                kl += teacher_prob * math.log(teacher_prob / student_prob)

                    kl_divergences.append(kl)

            except Exception as e:
                print(f"  ⚠️  KL evaluation failed for sample {idx}: {e}")
                continue

        if kl_divergences:
            avg_kl = sum(kl_divergences) / len(kl_divergences)
            results["kl_to_teacher"] = {
                "num_samples": len(kl_divergences),
                "avg_kl_divergence": avg_kl,
                "kl_divergences": kl_divergences[:10],  # Store first 10
            }
            print(f"\nKL to Teacher:")
            print(f"  Average KL divergence: {avg_kl:.4f}")
            print(f"  Evaluated on {len(kl_divergences)} positions")
        else:
            print(f"  ⚠️  No KL divergences computed")

    return results


def check_acceptance_gates(results: Dict) -> Dict:
    """
    Check if benchmark results pass acceptance gates.

    Args:
        results: Benchmark results dictionary

    Returns:
        Dict with gate check results and pass/fail status
    """
    gate_results = {
        "passed": True,
        "gates": {},
    }

    summary = results.get("summary", {})
    evaluation = results.get("evaluation", {})
    reconstruction = evaluation.get("reconstruction", {})
    kl_to_teacher = evaluation.get("kl_to_teacher", {})

    # Gate 1: Minimum throughput
    tokens_per_sec = summary.get("avg_tokens_per_sec", 0)
    gate_results["gates"]["tokens_per_sec"] = {
        "value": tokens_per_sec,
        "threshold": ACCEPTANCE_GATES["min_tokens_per_sec"],
        "passed": tokens_per_sec >= ACCEPTANCE_GATES["min_tokens_per_sec"],
        "name": "Minimum Throughput",
    }
    if not gate_results["gates"]["tokens_per_sec"]["passed"]:
        gate_results["passed"] = False

    # Gate 2: Minimum token accuracy (if eval was run)
    if reconstruction:
        token_accuracy = reconstruction.get("token_accuracy", 0)
        gate_results["gates"]["token_accuracy"] = {
            "value": token_accuracy,
            "threshold": ACCEPTANCE_GATES["min_token_accuracy"],
            "passed": token_accuracy >= ACCEPTANCE_GATES["min_token_accuracy"],
            "name": "AE Reconstruction Accuracy",
        }
        if not gate_results["gates"]["token_accuracy"]["passed"]:
            gate_results["passed"] = False

    # Gate 3: Maximum KL divergence (if eval was run)
    if kl_to_teacher:
        avg_kl = kl_to_teacher.get("avg_kl_divergence", float("inf"))
        gate_results["gates"]["kl_divergence"] = {
            "value": avg_kl,
            "threshold": ACCEPTANCE_GATES["max_avg_kl_divergence"],
            "passed": avg_kl <= ACCEPTANCE_GATES["max_avg_kl_divergence"],
            "name": "KL Divergence to Teacher",
        }
        if not gate_results["gates"]["kl_divergence"]["passed"]:
            gate_results["passed"] = False

    # Gate 4: KD cache hit rate (if available)
    kd_cache_stats = evaluation.get("kd_cache_stats", {})
    if kd_cache_stats:
        hit_rate = kd_cache_stats.get("hit_rate", 0)
        gate_results["gates"]["kd_cache_hit_rate"] = {
            "value": hit_rate,
            "threshold": ACCEPTANCE_GATES["min_kd_cache_hit_rate"],
            "passed": hit_rate >= ACCEPTANCE_GATES["min_kd_cache_hit_rate"],
            "name": "KD Cache Hit Rate",
        }
        if not gate_results["gates"]["kd_cache_hit_rate"]["passed"]:
            gate_results["passed"] = False

    return gate_results


def print_summary(results: Dict):
    """Print benchmark summary with acceptance gates."""
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
    print(f"  Tokens/sec (avg): {summary['avg_tokens_per_sec']:.2f}")
    print(f"  Tokens/sec (p50): {summary['tokens_per_sec_p50']:.2f}")
    print(f"  Tokens/sec (p95): {summary['tokens_per_sec_p95']:.2f}")
    print(f"\nLatency:")
    print(f"  Latency (p50): {summary['latency_p50']:.3f}s")
    print(f"  Latency (p95): {summary['latency_p95']:.3f}s")
    print(f"\nCompression:")
    print(f"  K (compression ratio): {summary['compression_ratio']}")
    print(f"  AR steps reduction: {summary['speedup_factor']}")

    # GPU memory
    if summary.get("gpu_memory"):
        gpu_mem = summary["gpu_memory"]
        print(f"\nGPU Memory (GB10):")
        print(f"  Peak allocated: {gpu_mem['peak_allocated_gb']:.2f} GB")
        print(f"  Peak reserved: {gpu_mem['peak_reserved_gb']:.2f} GB")
        print(f"  Current allocated: {gpu_mem['current_allocated_gb']:.2f} GB")

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
    parser.add_argument("--eval", action="store_true", help="Run evaluation on toy dataset")
    parser.add_argument("--eval_data", type=str, default="./data", help="Path to eval data directory")
    parser.add_argument("--eval_num_samples", type=int, default=100, help="Number of samples for reconstruction eval")
    parser.add_argument("--eval_kd_samples", type=int, default=10, help="Number of samples for KL to teacher eval")
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

    # Run evaluation if requested
    if args.eval:
        # Initialize KD client if available
        kd_client = None
        try:
            kd_client = KDClient()
            print("✅ KD client initialized for teacher KL evaluation")
        except Exception as e:
            print(f"⚠️  KD client initialization failed: {e}")
            print("   Skipping KL to teacher evaluation")

        eval_results = evaluate_on_toy_set(
            sampler=sampler,
            data_path=args.eval_data,
            kd_client=kd_client,
            num_samples=args.eval_num_samples,
            num_kd_samples=args.eval_kd_samples,
        )

        # Add eval results to main results
        results["evaluation"] = eval_results

        # Add KD cache stats if available
        if kd_client:
            results["evaluation"]["kd_cache_stats"] = kd_client.get_cache_stats()

    # Check acceptance gates
    gate_results = check_acceptance_gates(results)
    results["acceptance_gates"] = gate_results

    # Print gate results
    print(f"\n{'='*60}")
    print(f"ACCEPTANCE GATES")
    print(f"{'='*60}")
    for gate_name, gate_data in gate_results["gates"].items():
        status_emoji = "✅" if gate_data["passed"] else "❌"
        print(f"\n{status_emoji} {gate_data['name']}:")
        print(f"  Value: {gate_data['value']:.4f}")
        print(f"  Threshold: {gate_data['threshold']:.4f}")
        print(f"  Status: {'PASS' if gate_data['passed'] else 'FAIL'}")

    print(f"\n{'='*60}")
    if gate_results["passed"]:
        print(f"✅ ALL ACCEPTANCE GATES PASSED")
    else:
        print(f"❌ SOME ACCEPTANCE GATES FAILED")
    print(f"{'='*60}\n")

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
