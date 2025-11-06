"""Test failure modes and error handling."""
import sys
from pathlib import Path
import pytest
import os
import tempfile
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from kd.kd_client import VeniceKDClient
from tools.data_packing import pack_sequences
from ae.tokenizer_adapter import TokenizerAdapter


def test_missing_env_raises_error():
    """Test that missing .env raises clear error."""
    # Save original env vars
    original_url = os.getenv("VENICE_BASE_URL")
    original_key = os.getenv("VENICE_API_KEY")
    original_model = os.getenv("VENICE_MODEL")

    # Clear env vars
    if "VENICE_BASE_URL" in os.environ:
        del os.environ["VENICE_BASE_URL"]
    if "VENICE_API_KEY" in os.environ:
        del os.environ["VENICE_API_KEY"]
    if "VENICE_MODEL" in os.environ:
        del os.environ["VENICE_MODEL"]

    try:
        with pytest.raises(ValueError, match="Missing Venice API credentials"):
            VeniceKDClient()
    finally:
        # Restore env vars
        if original_url:
            os.environ["VENICE_BASE_URL"] = original_url
        if original_key:
            os.environ["VENICE_API_KEY"] = original_key
        if original_model:
            os.environ["VENICE_MODEL"] = original_model


def test_bad_api_key_raises_clear_error():
    """Test that bad API key raises clear auth error."""
    # This test might actually hit the API, so we'll skip it in CI
    # unless explicitly enabled
    if not os.getenv("TEST_BAD_KEY"):
        pytest.skip("Skipping bad key test (set TEST_BAD_KEY=1 to enable)")

    client = VeniceKDClient(api_key="bad_key_12345")

    with pytest.raises(RuntimeError, match="Authentication failed"):
        client.list_models()


def test_bad_url_raises_clear_error():
    """Test that bad URL raises clear connection error."""
    # Load real credentials first
    load_dotenv()

    client = VeniceKDClient(base_url="https://invalid-url-that-does-not-exist.com/api/v1")

    with pytest.raises(RuntimeError, match="(Connection error|Request failed)"):
        client.list_models()


def test_k_not_dividing_seq_len_raises_error():
    """Test that K not dividing seq_len raises clear error."""
    K = 8
    seq_len_bad = 65  # Not multiple of 8

    tokenizer = TokenizerAdapter()
    texts = ["Test text"]

    with pytest.raises(AssertionError, match="must be multiple of k"):
        pack_sequences(
            iter(texts),
            tokenizer=tokenizer,
            seq_len=seq_len_bad,
            k=K,
        )


def test_nonexistent_checkpoint_raises_error():
    """Test that loading nonexistent checkpoint raises error."""
    import torch

    with pytest.raises(FileNotFoundError):
        torch.load("/nonexistent/path/to/checkpoint.pt")
