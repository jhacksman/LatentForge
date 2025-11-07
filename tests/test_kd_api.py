"""Test Venice backend for KD."""
import sys
import os
from pathlib import Path
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kd.venice_backend import VeniceBackend


def test_venice_backend_init():
    """Test that Venice backend initializes from .env."""
    backend = VeniceBackend()

    assert backend.base_url is not None
    assert backend.api_key is not None
    assert backend.model is not None


@pytest.mark.timeout(60)
def test_venice_list_models_includes_qwen():
    """Test that GET /models includes qwen3-next-80b."""
    backend = VeniceBackend()

    models = backend.list_models()

    assert isinstance(models, list)
    assert len(models) > 0

    model_ids = [m.get("id") for m in models]
    assert "qwen3-next-80b" in model_ids, f"qwen3-next-80b not found in {model_ids[:5]}"


@pytest.mark.timeout(120)
def test_venice_returns_logprobs():
    """Test that Venice backend returns top_logprobs."""
    backend = VeniceBackend()

    output = backend.get_logprobs(
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=5,
        top_logprobs=5,
        temperature=1.0,
    )

    # Check structure
    assert len(output.tokens) > 0
    assert len(output.logprobs) > 0
    assert len(output.normalized_probs) > 0

    # Check first position has logprobs
    first_logprobs = output.logprobs[0]
    assert isinstance(first_logprobs, dict)
    assert len(first_logprobs) > 0, "Should have logprobs for first token"

    # Check normalized probs sum to ~1
    first_probs = output.normalized_probs[0]
    prob_sum = sum(first_probs.values())
    assert 0.9 <= prob_sum <= 1.1, f"Probs should sum to ~1, got {prob_sum}"


@pytest.mark.timeout(120)
def test_kd_client_with_venice():
    """Test KDClient with Venice backend."""
    os.environ["TEACHER_BACKEND"] = "venice"

    from kd.kd_client import KDClient

    client = KDClient()
    assert client.backend_type == "venice"

    output = client.get_teacher_distribution(
        prompt="Say hello",
        max_tokens=5,
        top_logprobs=5,
    )

    assert len(output.tokens) > 0
    assert len(output.logprobs) > 0

    # Test cache
    stats = client.get_cache_stats()
    assert stats["total_requests"] > 0
