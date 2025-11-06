"""Test vLLM remote backend for KD."""
import sys
import os
from pathlib import Path
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.skipif(
    not os.getenv("VLLM_REMOTE_URL"),
    reason="VLLM_REMOTE_URL not set (requires remote vLLM server)"
)
@pytest.mark.timeout(180)
def test_vllm_remote_backend():
    """Test vLLM remote backend."""
    from kd.vllm_backend import VLLMBackend

    base_url = os.getenv("VLLM_REMOTE_URL")
    api_key = os.getenv("VLLM_REMOTE_API_KEY", "")

    backend = VLLMBackend(
        base_url=base_url,
        api_key=api_key,
    )

    assert backend.mode == "remote"

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

    # Check normalized probs
    first_probs = output.normalized_probs[0]
    prob_sum = sum(first_probs.values())
    assert 0.9 <= prob_sum <= 1.1, f"Probs should sum to ~1, got {prob_sum}"


@pytest.mark.skipif(
    not os.getenv("VLLM_REMOTE_URL"),
    reason="VLLM_REMOTE_URL not set"
)
@pytest.mark.timeout(180)
def test_kd_client_with_vllm_remote():
    """Test KDClient with remote vLLM backend."""
    os.environ["TEACHER_BACKEND"] = "vllm-remote"

    from kd.kd_client import KDClient

    client = KDClient()
    assert client.backend_type == "vllm-remote"

    output = client.get_teacher_distribution(
        prompt="Say hello",
        max_tokens=5,
        top_logprobs=5,
    )

    assert len(output.tokens) > 0
    assert len(output.logprobs) > 0

    # Test caching
    stats = client.get_cache_stats()
    assert stats["total_requests"] > 0
