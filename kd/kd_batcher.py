"""
Batching utilities for knowledge distillation requests.
"""
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from tqdm import tqdm

from .kd_client import VeniceKDClient, TeacherOutput


@dataclass
class BatchRequest:
    """Single batch request."""
    idx: int
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float


@dataclass
class BatchResult:
    """Batch result with index."""
    idx: int
    output: Optional[TeacherOutput]
    error: Optional[str]


class KDBatcher:
    """Batch processor for KD requests."""

    def __init__(
        self,
        client: Optional[VeniceKDClient] = None,
        max_workers: int = 4,
        rate_limit_delay: float = 0.5,
    ):
        """
        Initialize KD batcher.

        Args:
            client: Venice KD client (creates new if None)
            max_workers: Maximum concurrent requests
            rate_limit_delay: Delay between requests in seconds
        """
        self.client = client or VeniceKDClient()
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay

    def process_batch(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_logprobs: int = 20,
        show_progress: bool = True,
    ) -> List[TeacherOutput]:
        """
        Process a batch of prompts.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_logprobs: Number of top logprobs
            show_progress: Show progress bar

        Returns:
            List of TeacherOutput (None for failed requests)
        """
        requests = [
            BatchRequest(
                idx=i,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for i, prompt in enumerate(prompts)
        ]

        results = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_req = {}
            for req in requests:
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                future = executor.submit(
                    self._process_single,
                    req,
                    top_logprobs,
                )
                future_to_req[future] = req

            # Collect results with progress bar
            iterator = as_completed(future_to_req)
            if show_progress:
                iterator = tqdm(iterator, total=len(requests), desc="KD batch")

            for future in iterator:
                req = future_to_req[future]
                try:
                    result = future.result()
                    if result.error is None:
                        results[result.idx] = result.output
                    else:
                        print(f"\nWarning: Request {result.idx} failed: {result.error}")
                except Exception as e:
                    print(f"\nError processing request {req.idx}: {e}")

        return results

    def _process_single(
        self,
        req: BatchRequest,
        top_logprobs: int,
    ) -> BatchResult:
        """Process a single request."""
        try:
            output = self.client.get_teacher_distribution(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_logprobs=top_logprobs,
            )
            return BatchResult(idx=req.idx, output=output, error=None)
        except Exception as e:
            return BatchResult(idx=req.idx, output=None, error=str(e))

    def process_sequences(
        self,
        sequences: List[List[str]],
        context_window: int = 512,
        **kwargs,
    ) -> List[List[TeacherOutput]]:
        """
        Process sequences of tokens, creating prompts from context windows.

        Args:
            sequences: List of token sequences
            context_window: Number of tokens for context
            **kwargs: Additional arguments for process_batch

        Returns:
            List of lists of TeacherOutput
        """
        all_outputs = []

        for seq in tqdm(sequences, desc="Processing sequences"):
            prompts = []
            # Create sliding window prompts
            for i in range(0, len(seq) - context_window, context_window // 2):
                context = " ".join(seq[i : i + context_window])
                prompts.append(context)

            if prompts:
                outputs = self.process_batch(prompts, **kwargs)
                all_outputs.append(outputs)
            else:
                all_outputs.append([])

        return all_outputs


def test_batcher():
    """Test the batcher."""
    batcher = KDBatcher(max_workers=2, rate_limit_delay=1.0)

    prompts = [
        "Count to 3:",
        "Say hello:",
        "Name a color:",
    ]

    print("Testing KD Batcher...")
    results = batcher.process_batch(
        prompts,
        max_tokens=10,
        temperature=0.8,
        show_progress=True,
    )

    print("\nResults:")
    for i, (prompt, output) in enumerate(zip(prompts, results)):
        if output:
            print(f"\nPrompt {i}: {prompt}")
            print(f"Output: {output.full_text}")
            print(f"Tokens: {output.tokens[:5]}")
        else:
            print(f"\nPrompt {i}: FAILED")


if __name__ == "__main__":
    test_batcher()
