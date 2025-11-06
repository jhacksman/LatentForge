#!/usr/bin/env python3
"""
Create a toy dataset for testing.
Combines public domain text with synthetic examples.
"""
import json
import os
from pathlib import Path


# Public domain text samples
PUBLIC_DOMAIN_SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "All human beings are born free and equal in dignity and rights.",
    "To be or not to be, that is the question.",
    "In the beginning was the Word, and the Word was with God.",
    "Four score and seven years ago our fathers brought forth on this continent a new nation.",
    "We hold these truths to be self-evident, that all men are created equal.",
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael. Some years ago, never mind how long precisely.",
    "Happy families are all alike; every unhappy family is unhappy in its own way.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
]

# Synthetic code samples
CODE_SAMPLES = [
    """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)""",

    """def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)""",

    """def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True""",

    """class Calculator:
    def add(self, a, b):
        return a + b
    def subtract(self, a, b):
        return a - b""",

    """import numpy as np
def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector""",
]

# Synthetic instruction-response pairs
SYNTHETIC_PAIRS = [
    ("Explain photosynthesis.", "Photosynthesis is the process by which green plants convert light energy into chemical energy stored in glucose."),
    ("What is machine learning?", "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."),
    ("Define recursion.", "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem."),
    ("What are prime numbers?", "Prime numbers are natural numbers greater than 1 that have no positive divisors other than 1 and themselves."),
    ("Describe neural networks.", "Neural networks are computing systems inspired by biological neural networks that learn to perform tasks by considering examples."),
]

# Longer form content
LONGER_SAMPLES = [
    """The Industrial Revolution was a period of major industrialization and innovation during the late 1700s and early 1800s. The Industrial Revolution began in Great Britain and quickly spread throughout the world. The American Industrial Revolution commonly referred to as the Second Industrial Revolution, started sometime between 1820 and 1870. This period saw the mechanization of agriculture and textile manufacturing and a revolution in power, including steam ships and railroads, that effected social, cultural and economic conditions.""",

    """Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, such as through variations in the solar cycle. But since the 1800s, human activities have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil and gas. Burning fossil fuels generates greenhouse gas emissions that act like a blanket wrapped around the Earth, trapping the sun's heat and raising temperatures.""",

    """Artificial intelligence is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term artificial intelligence had previously been used to describe machines that mimic and display human cognitive skills that are associated with the human mind, such as learning and problem-solving.""",
]


def create_toy_dataset(output_file: str = "data/toy.jsonl", num_repeats: int = 3):
    """
    Create toy dataset with mixed content.

    Args:
        output_file: Output JSONL file path
        num_repeats: Number of times to repeat the dataset
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []

    # Add public domain samples
    for text in PUBLIC_DOMAIN_SAMPLES:
        samples.append({"text": text, "source": "public_domain"})

    # Add code samples
    for code in CODE_SAMPLES:
        samples.append({"text": code, "source": "code"})

    # Add instruction-response pairs
    for instruction, response in SYNTHETIC_PAIRS:
        samples.append({"text": f"{instruction} {response}", "source": "synthetic"})

    # Add longer samples
    for text in LONGER_SAMPLES:
        samples.append({"text": text, "source": "long_form"})

    # Repeat to get more data
    all_samples = samples * num_repeats

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Created toy dataset: {output_path}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Unique samples: {len(samples)}")
    print(f"  File size: {output_path.stat().st_size} bytes")

    # Print sample
    print(f"\nFirst sample:")
    print(f"  {all_samples[0]}")


if __name__ == "__main__":
    create_toy_dataset()
