#!/usr/bin/env python3
"""
vLLM backend for knowledge distillation.
Supports both local in-process and remote API modes.
"""
import os
import math
import time
import random
from typing import Dict, List, Optional
from dotenv import load_dotenv

from kd.kd_backend import TeacherBackend, TeacherOutput


class VLLMBackend(TeacherBackend):
    """
    vLLM backend supporting local and remote modes.

    Local mode: Loads model in-process with vLLM.LLM
    Remote mode: Connects to remote vLLM OpenAI-compatible API
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        quantization: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 32768,
        timeout: int = 120,
        max_retries: int = 3,
        max_backoff: float = 32.0,
    ):
        """
        Initialize vLLM backend.

        Args:
            model_path: Path or HF model ID for local mode (e.g., Qwen/Qwen3-Next-80B-A3B-Instruct)
            base_url: Base URL for remote mode (e.g., http://other-gb10:8000/v1)
            api_key: Optional API key for remote mode
            quantization: Quantization method (gptq, awq) for local mode
            tensor_parallel_size: Tensor parallel size for local mode (1 for single GB10)
            gpu_memory_utilization: GPU memory utilization fraction
            max_model_len: Maximum model length
            timeout: Request timeout
            max_retries: Maximum retries
            max_backoff: Maximum backoff time
        """
        load_dotenv()

        # Determine mode
        self.model_path = model_path or os.getenv("VLLM_LOCAL_MODEL")
        self.base_url = base_url or os.getenv("VLLM_REMOTE_URL")

        if self.model_path and not self.base_url:
            self.mode = "local"
        elif self.base_url and not self.model_path:
            self.mode = "remote"
        elif self.model_path and self.base_url:
            raise ValueError("Cannot specify both model_path (local) and base_url (remote). Choose one.")
        else:
            raise ValueError("Must specify either model_path (local) or base_url (remote).")

        self.timeout = timeout
        self.max_retries = max_retries
        self.max_backoff = max_backoff

        # Initialize based on mode
        if self.mode == "local":
            self._init_local(
                quantization=quantization,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
        else:
            self._init_remote(api_key=api_key)

    def _init_local(
        self,
        quantization: Optional[str],
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
    ):
        """Initialize local vLLM model."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm"
            )

        # Get params from env if not provided
        quantization = quantization or os.getenv("VLLM_QUANTIZATION", "gptq")
        gpu_memory_utilization = float(os.getenv("VLLM_GPU_MEMORY_UTIL", str(gpu_memory_utilization)))
        max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", str(max_model_len)))

        print(f"Loading vLLM model locally: {self.model_path}")
        print(f"  Quantization: {quantization}")
        print(f"  Tensor parallel size: {tensor_parallel_size}")
        print(f"  GPU memory utilization: {gpu_memory_utilization}")
        print(f"  Max model length: {max_model_len}")

        self.llm = LLM(
            model=self.model_path,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        self.SamplingParams = SamplingParams

        print(f"✅ vLLM model loaded successfully")

    def _init_remote(self, api_key: Optional[str]):
        """Initialize remote vLLM client."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI client not installed. Install with: pip install openai"
            )

        self.api_key = api_key or os.getenv("VLLM_REMOTE_API_KEY", "")

        print(f"Connecting to remote vLLM: {self.base_url}")

        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key if self.api_key else "EMPTY",
            timeout=self.timeout,
        )

        print(f"✅ Remote vLLM client initialized")

    def get_logprobs(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        top_logprobs: int = 20,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> TeacherOutput:
        """
        Get logprobs from vLLM (local or remote).

        Args:
            messages: OpenAI-style message list
            max_tokens: Maximum tokens to generate
            top_logprobs: Number of top logprobs per position
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            TeacherOutput with sparse distributions
        """
        if self.mode == "local":
            return self._get_logprobs_local(
                messages=messages,
                max_tokens=max_tokens,
                top_logprobs=top_logprobs,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            return self._get_logprobs_remote(
                messages=messages,
                max_tokens=max_tokens,
                top_logprobs=top_logprobs,
                temperature=temperature,
                top_p=top_p,
            )

    def _get_logprobs_local(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        top_logprobs: int,
        temperature: float,
        top_p: float,
    ) -> TeacherOutput:
        """Get logprobs from local vLLM model."""
        # Convert messages to prompt
        # Simple concatenation; in production use proper chat template
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        # Configure sampling
        sampling_params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=top_logprobs,  # Request top-k logprobs
        )

        # Generate
        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]

        # Parse output
        tokens = []
        logprobs = []
        normalized_probs = []

        for token_output in output.outputs[0].logprobs:
            # token_output is a dict: {token_id: Logprob}
            # Logprob has .logprob and .decoded_token
            logprob_dict = {}
            for token_id, logprob_obj in token_output.items():
                token_str = logprob_obj.decoded_token
                logp = logprob_obj.logprob
                logprob_dict[token_str] = logp

            # Get the actual generated token (highest probability)
            if logprob_dict:
                top_token = max(logprob_dict.items(), key=lambda x: x[1])[0]
                tokens.append(top_token)
            else:
                tokens.append("")

            logprobs.append(logprob_dict)

            # Normalize to probabilities
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
            logprobs=logprobs,
            normalized_probs=normalized_probs,
        )

    def _get_logprobs_remote(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        top_logprobs: int,
        temperature: float,
        top_p: float,
    ) -> TeacherOutput:
        """Get logprobs from remote vLLM API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="qwen3-next-80b",  # Model name from vLLM server
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    logprobs=True,
                    top_logprobs=top_logprobs,
                )

                return self._parse_remote_response(response)

            except Exception as e:
                error_str = str(e)

                # Check for rate limit or server errors
                if "429" in error_str or "500" in error_str or "502" in error_str or "503" in error_str:
                    if attempt < self.max_retries - 1:
                        base_backoff = min(2 ** attempt, self.max_backoff)
                        jitter = random.uniform(0, 0.1 * base_backoff)
                        backoff = base_backoff + jitter
                        print(f"  Retrying after {backoff:.2f}s (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(backoff)
                        continue
                    else:
                        raise RuntimeError(
                            f"Remote vLLM request failed after {self.max_retries} retries: {e}"
                        )
                else:
                    # Non-retryable error
                    raise RuntimeError(f"Remote vLLM request failed: {e}")

        raise RuntimeError(f"Request failed after {self.max_retries} attempts")

    def _parse_remote_response(self, response) -> TeacherOutput:
        """Parse OpenAI-compatible response."""
        choice = response.choices[0]

        tokens = []
        logprobs = []
        normalized_probs = []

        # Parse logprobs from response
        if hasattr(choice, "logprobs") and choice.logprobs:
            content_logprobs = choice.logprobs.content or []

            for item in content_logprobs:
                # Current token
                token = item.token
                tokens.append(token)

                # Top logprobs for this position
                logprob_dict = {}
                if hasattr(item, "top_logprobs") and item.top_logprobs:
                    for entry in item.top_logprobs:
                        tok = entry.token
                        logp = entry.logprob
                        logprob_dict[tok] = logp

                logprobs.append(logprob_dict)

                # Normalize to probabilities
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
            logprobs=logprobs,
            normalized_probs=normalized_probs,
        )
