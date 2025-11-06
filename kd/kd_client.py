"""
Knowledge distillation client for Venice API chat completions.
"""
import os
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import requests
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
        max_backoff: float = 32.0,
    ):
        """
        Initialize Venice KD client.

        Args:
            base_url: Venice API base URL (default from env)
            api_key: API key (default from env)
            model: Model name (default from env)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            max_backoff: Maximum backoff time in seconds
        """
        load_dotenv()

        self.base_url = base_url or os.getenv("VENICE_BASE_URL")
        self.api_key = api_key or os.getenv("VENICE_API_KEY")
        self.model = model or os.getenv("VENICE_MODEL")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_backoff = max_backoff

        if not all([self.base_url, self.api_key, self.model]):
            raise ValueError("Missing Venice API credentials. Check .env file.")

        self.completions_endpoint = f"{self.base_url}/chat/completions"
        self.models_endpoint = f"{self.base_url}/models"

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authorization."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _handle_request_with_backoff(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> requests.Response:
        """
        Make HTTP request with exponential backoff for retryable errors.

        Args:
            method: HTTP method (GET, POST)
            url: Request URL
            **kwargs: Additional request arguments

        Returns:
            Response object

        Raises:
            RuntimeError: On auth errors or max retries exceeded
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.request(method, url, **kwargs)

                # Check for auth errors (don't retry)
                if response.status_code == 401:
                    raise RuntimeError(
                        "Authentication failed. Check your VENICE_API_KEY in .env file."
                    )

                # Check for rate limiting or server errors (retry with backoff)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.max_retries - 1:
                        # Capped exponential backoff
                        backoff = min(2 ** attempt, self.max_backoff)
                        time.sleep(backoff)
                        continue
                    else:
                        raise RuntimeError(
                            f"API request failed after {self.max_retries} retries. "
                            f"Status: {response.status_code}, Response: {response.text}"
                        )

                # Raise for other HTTP errors
                response.raise_for_status()
                return response

            except requests.exceptions.Timeout as e:
                if attempt < self.max_retries - 1:
                    backoff = min(2 ** attempt, self.max_backoff)
                    time.sleep(backoff)
                    continue
                raise RuntimeError(f"Request timed out after {self.max_retries} attempts: {e}")

            except requests.exceptions.ConnectionError as e:
                if attempt < self.max_retries - 1:
                    backoff = min(2 ** attempt, self.max_backoff)
                    time.sleep(backoff)
                    continue
                raise RuntimeError(f"Connection error after {self.max_retries} attempts: {e}")

            except requests.exceptions.RequestException as e:
                # Don't retry for other request exceptions
                raise RuntimeError(f"Request failed: {e}")

        raise RuntimeError(f"Request failed after {self.max_retries} attempts")

    def list_models(self) -> List[Dict]:
        """
        List available models from Venice API.

        Returns:
            List of model dictionaries with id, name, etc.

        Raises:
            RuntimeError: On API errors
        """
        response = self._handle_request_with_backoff(
            "GET",
            self.models_endpoint,
            headers=self._get_headers(),
            timeout=self.timeout,
        )

        data = response.json()
        return data.get("data", [])

    def get_logprobs(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        top_logprobs: int = 20,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> TeacherOutput:
        """
        Get logprobs from chat completion.

        Args:
            messages: List of message dicts with role and content
            max_tokens: Maximum tokens to generate
            top_logprobs: Number of top logprobs to return (1-20)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            TeacherOutput with sparse per-position distributions

        Raises:
            RuntimeError: On API errors
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }

        response = self._handle_request_with_backoff(
            "POST",
            self.completions_endpoint,
            headers=self._get_headers(),
            json=payload,
            timeout=self.timeout,
        )

        data = response.json()
        return self._parse_response(data)

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
        messages = [{"role": "user", "content": prompt}]
        return self.get_logprobs(
            messages=messages,
            max_tokens=max_tokens,
            top_logprobs=top_logprobs,
            temperature=temperature,
            top_p=top_p,
        )

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
