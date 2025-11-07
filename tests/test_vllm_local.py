"""Test vLLM local backend for KD."""
import sys
import os
from pathlib import Path
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.skipif(
    not os.getenv("TEST_VLLM_LOCAL"),
    reason="TEST_VLLM_LOCAL not set (requires 80B model and GPU)"
)
@pytest.mark.timeout(300)
def test_vllm_local_backend():
    """Test vLLM local backend with full 80B model."""
    from kd.vllm_backend import VLLMBackend

    # Use model from env
    model_path = os.getenv("VLLM_LOCAL_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct")

    backend = VLLMBackend(
        model_path=model_path,
        quantization="gptq",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=8192,  # Smaller for testing
    )

    assert backend.mode == "local"

    output = backend.get_logprobs(
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=5,
        top_logprobs=5,
        temperature=1.0,
    )

    assert len(output.tokens) > 0
    assert len(output.logprobs) > 0
    assert len(output.normalized_probs) > 0

    # Check sparse distributions
    first_logprobs = output.logprobs[0]
    assert isinstance(first_logprobs, dict)
    assert len(first_logprobs) > 0


@pytest.mark.skipif(
    not os.getenv("TEST_VLLM_LOCAL"),
    reason="TEST_VLLM_LOCAL not set"
)
@pytest.mark.timeout(300)
def test_kd_client_with_vllm_local():
    """Test KDClient with local vLLM backend."""
    os.environ["TEACHER_BACKEND"] = "vllm-local"

    from kd.kd_client import KDClient

    client = KDClient()
    assert client.backend_type == "vllm-local"

    output = client.get_teacher_distribution(
        prompt="Say hello",
        max_tokens=5,
        top_logprobs=5,
    )

    assert len(output.tokens) > 0
    assert len(output.logprobs) > 0
