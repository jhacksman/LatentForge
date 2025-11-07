"""Test tokenization alignment between teacher and student.

CRITICAL: This test ensures that the tokenizer used in the AE and student training
matches the teacher's tokenizer EXACTLY. Any mismatch will cause KD to fail.
"""
import sys
import os
from pathlib import Path
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ae.tokenizer_adapter import TokenizerAdapter
from transformers import AutoTokenizer


def test_tokenizer_uses_teacher_model():
    """Test that TokenizerAdapter uses the correct teacher tokenizer."""
    os.environ["TEACHER_BACKEND"] = "venice"

    adapter = TokenizerAdapter()

    # Should use Qwen3-Next-80B tokenizer
    assert "Qwen" in adapter.model_name
    assert adapter.vocab_size > 0
    print(f"TokenizerAdapter model: {adapter.model_name}, vocab_size: {adapter.vocab_size}")


def test_tokenization_roundtrip():
    """Test that encode/decode roundtrip works correctly."""
    adapter = TokenizerAdapter()

    test_text = "Hello, world! This is a test of the tokenizer."
    token_ids = adapter.encode(test_text, return_tensors="pt")
    decoded_text = adapter.decode(token_ids[0])

    # Should be identical or very close (may differ in whitespace)
    assert decoded_text.strip() == test_text.strip()
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded_text}")


def test_tokenization_matches_teacher():
    """Test that tokenization matches the actual teacher tokenizer."""
    os.environ["TEACHER_BACKEND"] = "venice"

    adapter = TokenizerAdapter()

    # Load teacher tokenizer directly
    teacher_tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        trust_remote_code=True,
    )

    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Knowledge distillation from large language models.",
        "GB10 Grace-Blackwell superchip with 128GB unified memory.",
        "对于中文文本的支持测试",  # Chinese text support
    ]

    for text in test_texts:
        # Encode with adapter
        adapter_ids = adapter.encode(text, return_tensors="pt", add_special_tokens=False)

        # Encode with teacher directly
        teacher_ids = teacher_tok.encode(text, return_tensors="pt", add_special_tokens=False)

        # Should be IDENTICAL
        assert adapter_ids.shape == teacher_ids.shape, f"Shape mismatch for '{text}'"
        assert (adapter_ids == teacher_ids).all(), f"Token ID mismatch for '{text}'"

        print(f"✓ Matched: {text[:50]}...")


def test_teacher_token_encoding():
    """Test that teacher token strings encode correctly."""
    adapter = TokenizerAdapter()

    # Simulate teacher returning these tokens
    teacher_tokens = ["Hello", ",", " world", "!", " This", " is", " a", " test", "."]

    for token_str in teacher_tokens:
        # Encode the token string
        token_ids = adapter.encode(token_str, return_tensors="pt", add_special_tokens=False)

        # Should get at least one token ID
        assert token_ids.shape[1] > 0, f"Failed to encode teacher token: '{token_str}'"

        # Decode back
        decoded = adapter.decode(token_ids[0])

        # Should be identical or very close
        assert decoded.strip() == token_str.strip() or token_str.strip() in decoded

        print(f"Token: '{token_str}' -> IDs: {token_ids[0].tolist()} -> '{decoded}'")


def test_vocab_size_consistency():
    """Test that vocab size is consistent across different backends."""
    backends = ["venice", "vllm-local", "vllm-remote"]
    vocab_sizes = []

    for backend in backends:
        os.environ["TEACHER_BACKEND"] = backend
        adapter = TokenizerAdapter()
        vocab_sizes.append(adapter.vocab_size)
        print(f"{backend}: vocab_size={adapter.vocab_size}")

    # All should be the same (all use Qwen3-Next-80B)
    assert len(set(vocab_sizes)) == 1, f"Vocab sizes differ across backends: {vocab_sizes}"


def test_kd_alignment_simulation():
    """Simulate the KD alignment process to ensure it works correctly."""
    adapter = TokenizerAdapter()

    # Simulate teacher returning tokens and probs
    teacher_tokens = [" Hello", ",", " world"]
    teacher_probs = [
        {" Hello": 0.9, " Hi": 0.05, " Hey": 0.05},
        {",": 0.95, ".": 0.03, "!": 0.02},
        {" world": 0.8, " there": 0.15, " everyone": 0.05},
    ]

    for i, (token_str, probs_dict) in enumerate(zip(teacher_tokens, teacher_probs)):
        # Encode teacher token
        token_ids = adapter.encode(token_str, return_tensors="pt", add_special_tokens=False)
        assert token_ids.shape[1] > 0, f"Failed to encode '{token_str}'"

        # Encode all tokens in the probability distribution
        for prob_token_str, prob in probs_dict.items():
            prob_token_ids = adapter.encode(prob_token_str, return_tensors="pt", add_special_tokens=False)
            assert prob_token_ids.shape[1] > 0, f"Failed to encode '{prob_token_str}'"

        print(f"Position {i}: Teacher token '{token_str}' encoded successfully")


if __name__ == "__main__":
    print("Running tokenization alignment tests...")
    test_tokenizer_uses_teacher_model()
    test_tokenization_roundtrip()
    test_tokenization_matches_teacher()
    test_teacher_token_encoding()
    test_vocab_size_consistency()
    test_kd_alignment_simulation()
    print("\n✅ All tokenization alignment tests passed!")
