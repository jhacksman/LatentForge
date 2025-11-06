#!/usr/bin/env python3
"""
Verify Venice API connectivity and model availability.
"""
import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv


def verify_api():
    """Verify Venice API credentials and model availability."""
    # Load environment variables
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    base_url = os.getenv("VENICE_BASE_URL")
    api_key = os.getenv("VENICE_API_KEY")
    model_name = os.getenv("VENICE_MODEL")

    if not all([base_url, api_key, model_name]):
        print("❌ ERROR: Missing environment variables in .env")
        print(f"   VENICE_BASE_URL: {'✓' if base_url else '✗'}")
        print(f"   VENICE_API_KEY: {'✓' if api_key else '✗'}")
        print(f"   VENICE_MODEL: {'✓' if model_name else '✗'}")
        return False

    print(f"🔍 Verifying Venice API connection...")
    print(f"   Base URL: {base_url}")
    print(f"   Target Model: {model_name}")
    print()

    # Call GET /models endpoint
    url = f"{base_url}/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)

        # Assert 200 status
        if response.status_code != 200:
            print(f"❌ ERROR: API returned status {response.status_code}")
            print(f"   Response: {response.text}")
            return False

        print(f"✅ API connection successful (HTTP {response.status_code})")

        # Parse response
        data = response.json()
        models = data.get("data", [])

        print(f"📋 Found {len(models)} models")
        print()

        # Check if target model exists
        target_model = None
        for model in models:
            if model.get("id") == model_name:
                target_model = model
                break

        if not target_model:
            print(f"❌ ERROR: Model '{model_name}' not found in available models")
            print("\nAvailable models:")
            for model in models:
                print(f"   - {model.get('id')}")
            return False

        print(f"✅ Target model '{model_name}' is available")

        # Print logprobs support if present
        if "supportsLogProbs" in target_model:
            supports_logprobs = target_model["supportsLogProbs"]
            status = "✅" if supports_logprobs else "⚠️"
            print(f"{status} supportsLogProbs: {supports_logprobs}")

        # Print additional model info
        print("\n📊 Model details:")
        for key, value in target_model.items():
            if key != "id":
                print(f"   {key}: {value}")

        print("\n🎉 All checks passed! Venice API is ready for use.")
        return True

    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timed out")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Request failed: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON response: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = verify_api()
    sys.exit(0 if success else 1)
