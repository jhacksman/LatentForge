"""
Knowledge distillation client for Venice API chat completions.
"""
import os
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import requests
import numpy as np
from dotenv import load_dotenv


@dataclass
class TeacherOutput:
    """Teacher model output with token distributions."""
    tokens: List[str]  # Generated tokens
    token_ids: List[int]  # Token IDs (if available)
    logprobs: List[Dict[str, float]]  # Per-position logprobs {token: logprob}
    normalized_probs: List[Dict[str, float]]  # Normalized probabilities {token: prob}
    full_text: str  # Complete generated text


class VeniceKDClient:
    """Client for knowledge distillation from Venice API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """
        Initialize Venice KD client.

        Args:
            base_url: Venice API base URL (default from env)
            api_key: API key (default from env)
            model: Model name (default from env)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        load_dotenv()

        self.base_url = base_url or os.getenv("VENICE_BASE_URL")
        self.api_key = api_key or os.getenv("VENICE_API_KEY")
        self.model = model or os.getenv("VENICE_MODEL")
        self.timeout = timeout
        self.max_retries = max_retries

        if not all([self.base_url, self.api_key, self.model]):
            raise ValueError("Missing Venice API credentials. Check .env file.")

        self.endpoint = f"{self.base_url}/chat/completions"

    def get_teacher_distribution(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_logprobs: int = 20,
    ) -> TeacherOutput:
        """
        Get teacher model output with token distributions.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_logprobs: Number of top logprobs to return

        Returns:
            TeacherOutput with tokens and distributions
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Venice API request failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

        data = response.json()
        return self._parse_response(data)

    def _parse_response(self, data: Dict) -> TeacherOutput:
        """
        Parse Venice API response into TeacherOutput.

        Args:
            data: API response JSON

        Returns:
            TeacherOutput with parsed distributions
        """
        choice = data["choices"][0]
        full_text = choice["message"]["content"]

        # Extract logprobs from response
        logprobs_data = choice.get("logprobs", {})
        content_logprobs = logprobs_data.get("content", [])

        tokens = []
        token_ids = []
        logprobs = []
        normalized_probs = []

        for item in content_logprobs:
            # Current token
            token = item.get("token", "")
            token_id = item.get("token_id", -1)
            tokens.append(token)
            token_ids.append(token_id)

            # Top logprobs for this position
            top_logprobs = item.get("top_logprobs", [])

            # Build logprob dict
            logprob_dict = {}
            for entry in top_logprobs:
                tok = entry.get("token", "")
                logp = entry.get("logprob", -float("inf"))
                logprob_dict[tok] = logp

            logprobs.append(logprob_dict)

            # Normalize to probabilities
            # Use log-sum-exp trick for numerical stability
            if logprob_dict:
                max_logp = max(logprob_dict.values())
                log_sum_exp = max_logp + math.log(
                    sum(math.exp(lp - max_logp) for lp in logprob_dict.values())
                )

                prob_dict = {}
                for tok, logp in logprob_dict.items():
                    prob = math.exp(logp - log_sum_exp)
                    prob_dict[tok] = prob

                normalized_probs.append(prob_dict)
            else:
                normalized_probs.append({})

        return TeacherOutput(
            tokens=tokens,
            token_ids=token_ids,
            logprobs=logprobs,
            normalized_probs=normalized_probs,
            full_text=full_text,
        )

    def get_distribution_for_tokens(
        self,
        prompt: str,
        target_tokens: List[str],
        **kwargs,
    ) -> List[Dict[str, float]]:
        """
        Get teacher distributions for specific target tokens.

        Args:
            prompt: Input prompt
            target_tokens: Target tokens to get distributions for
            **kwargs: Additional arguments for get_teacher_distribution

        Returns:
            List of probability distributions
        """
        output = self.get_teacher_distribution(prompt, **kwargs)

        # Match output tokens to target tokens
        # This is a simplified version; in practice you may need alignment
        distributions = []
        for i, target_tok in enumerate(target_tokens):
            if i < len(output.normalized_probs):
                distributions.append(output.normalized_probs[i])
            else:
                distributions.append({})

        return distributions


def test_client():
    """Test the KD client."""
    client = VeniceKDClient()

    print("Testing Venice KD Client...")
    output = client.get_teacher_distribution(
        prompt="Say hello in one word",
        max_tokens=5,
        temperature=1.0,
        top_logprobs=5,
    )

    print(f"\nGenerated text: {output.full_text}")
    print(f"\nTokens: {output.tokens}")
    print("\nTop probabilities per position:")
    for i, (token, probs) in enumerate(zip(output.tokens, output.normalized_probs)):
        print(f"\nPosition {i} (token: '{token}'):")
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        for tok, prob in sorted_probs:
            print(f"  '{tok}': {prob:.4f}")


if __name__ == "__main__":
    test_client()
