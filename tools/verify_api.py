#!/usr/bin/env python3
"""
Verify Venice API connectivity and model availability.
"""
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kd.kd_client import VeniceKDClient


def verify_api():
    """Verify Venice API credentials and model availability."""
    try:
        print("🔍 Verifying Venice API connection...")

        # Create client (will load .env automatically)
        try:
            client = VeniceKDClient()
        except ValueError as e:
            print(f"❌ ERROR: {e}")
            return False

        print(f"   Base URL: {client.base_url}")
        print(f"   Target Model: {client.model}")
        print()

        # Test 1: GET /models
        print("Test 1: Listing available models...")
        try:
            models = client.list_models()
            print(f"✅ Found {len(models)} models")

            # Check if target model exists
            target_model = None
            for model in models:
                if model.get("id") == client.model:
                    target_model = model
                    break

            if not target_model:
                print(f"❌ ERROR: Model '{client.model}' not found in available models")
                print("\nAvailable models:")
                for model in models[:10]:  # Show first 10
                    print(f"   - {model.get('id')}")
                return False

            print(f"✅ Target model '{client.model}' is available")

            # Print logprobs support if present
            if "supportsLogProbs" in target_model:
                supports_logprobs = target_model["supportsLogProbs"]
                status = "✅" if supports_logprobs else "⚠️"
                print(f"{status} supportsLogProbs: {supports_logprobs}")

        except Exception as e:
            print(f"❌ Model listing failed: {e}")
            return False

        # Test 2: Small chat completion with logprobs
        print("\nTest 2: Chat completion with logprobs...")
        try:
            output = client.get_logprobs(
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5,
                top_logprobs=5,
                temperature=1.0,
            )

            print(f"✅ Chat completion successful")
            print(f"   Generated: {output.full_text}")
            print(f"   Tokens: {len(output.tokens)}")
            print(f"   Logprobs positions: {len(output.logprobs)}")

            # Check that we got logprobs
            if output.logprobs and len(output.logprobs) > 0:
                first_pos_probs = output.logprobs[0]
                print(f"   Top logprobs at position 0: {len(first_pos_probs)} tokens")
                print(f"✅ Logprobs are working correctly")
            else:
                print(f"⚠️  Warning: No logprobs returned")

        except Exception as e:
            print(f"❌ Chat completion failed: {e}")
            return False

        print("\n🎉 All checks passed! Venice API is ready for use.")
        return True

    except Exception as e:
        print(f"❌ ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_api()
    sys.exit(0 if success else 1)
