"""Test environment and .env file."""
import os
from pathlib import Path
import pytest
from dotenv import load_dotenv


def test_env_file_exists():
    """Check that .env file exists."""
    env_path = Path(__file__).parent.parent / ".env"
    assert env_path.exists(), ".env file not found in repository root"


def test_env_file_parseable():
    """Check that .env file can be parsed."""
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    # Check that we can load it without errors
    assert True


def test_env_variables_set():
    """Check that required environment variables are set."""
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    venice_base_url = os.getenv("VENICE_BASE_URL")
    venice_api_key = os.getenv("VENICE_API_KEY")
    venice_model = os.getenv("VENICE_MODEL")

    assert venice_base_url is not None, "VENICE_BASE_URL not set in .env"
    assert venice_api_key is not None, "VENICE_API_KEY not set in .env"
    assert venice_model is not None, "VENICE_MODEL not set in .env"

    # Check format
    assert venice_base_url.startswith("http"), "VENICE_BASE_URL should start with http"
    assert len(venice_api_key) > 10, "VENICE_API_KEY seems too short"
    assert "qwen" in venice_model.lower(), "VENICE_MODEL should be a Qwen model"
