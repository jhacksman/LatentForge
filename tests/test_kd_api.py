"""Test KD API client."""
import sys
from pathlib import Path
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kd.kd_client import VeniceKDClient


def test_kd_client_init():
    """Test that KD client initializes from .env."""
    client = VeniceKDClient()

    assert client.base_url is not None
    assert client.api_key is not None
    assert client.model is not None


@pytest.mark.timeout(60)
def test_list_models_includes_qwen():
    """Test that GET /models includes qwen3-next-80b."""
    client = VeniceKDClient()

    models = client.list_models()

    assert isinstance(models, list)
    assert len(models) > 0

    model_ids = [m.get("id") for m in models]
    assert "qwen3-next-80b" in model_ids, f"qwen3-next-80b not found in {model_ids[:5]}"


@pytest.mark.timeout(120)
def test_chat_completions_returns_logprobs():
    """Test that chat completions returns top_logprobs."""
    client = VeniceKDClient()

    output = client.get_logprobs(
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=5,
        top_logprobs=5,
        temperature=1.0,
    )

    # Check structure
    assert output.full_text is not None
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
