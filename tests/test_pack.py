"""Test data packing."""
import sys
from pathlib import Path
import pytest
import torch
import tempfile
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.data_packing import pack_sequences
from ae.tokenizer_adapter import TokenizerAdapter


def test_packed_sequences_multiple_of_k():
    """Verify that packed sequences are multiples of K."""
    K = 8

    # Create sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a test.",
        "Machine learning is fascinating.",
    ]

    # Create tokenizer
    tokenizer = TokenizerAdapter()

    # Pack sequences
    sequences = pack_sequences(
        iter(texts),
        tokenizer=tokenizer,
        seq_len=64,  # Must be multiple of K
        k=K,
        max_sequences=10,
    )

    assert len(sequences) > 0, "Should have created at least one sequence"

    for i, seq in enumerate(sequences):
        assert isinstance(seq, torch.Tensor)
        assert seq.shape[0] == 64, f"Sequence {i} should be length 64, got {seq.shape[0]}"
        assert seq.shape[0] % K == 0, f"Sequence {i} length not multiple of K={K}"


def test_pack_raises_on_bad_seq_len():
    """Test that packing raises error when seq_len is not multiple of K."""
    K = 8
    seq_len_bad = 65  # Not multiple of 8

    texts = ["Test text"]
    tokenizer = TokenizerAdapter()

    with pytest.raises(AssertionError, match="must be multiple of k"):
        pack_sequences(
            iter(texts),
            tokenizer=tokenizer,
            seq_len=seq_len_bad,
            k=K,
        )
