"""
Tokenizer adapter for the autoencoder.
Reuses the teacher model's tokenizer (Qwen).
"""
from typing import List, Union, Optional
import torch
from transformers import AutoTokenizer


class TokenizerAdapter:
    """Adapter for teacher tokenizer."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B",
        trust_remote_code: bool = True,
    ):
        """
        Initialize tokenizer adapter.

        Args:
            model_name: HuggingFace model name for tokenizer
            trust_remote_code: Whether to trust remote code
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vocab_size = len(self.tokenizer)

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = "pt",
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs.

        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add special tokens
            return_tensors: Return format ('pt' for PyTorch)
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length

        Returns:
            Token IDs as list or tensor
        """
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

        if return_tensors == "pt":
            return encoded["input_ids"]
        else:
            return encoded["input_ids"]

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs (single sequence or batch)
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() == 1:
                # Single sequence
                return self.tokenizer.decode(
                    token_ids.tolist(),
                    skip_special_tokens=skip_special_tokens,
                )
            else:
                # Batch
                return self.tokenizer.batch_decode(
                    token_ids.tolist(),
                    skip_special_tokens=skip_special_tokens,
                )
        else:
            # List of ints
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
            )

    def get_token_embeddings(self) -> torch.Tensor:
        """
        Get token embedding matrix from tokenizer.

        Returns:
            Token embedding matrix (vocab_size, embed_dim)
        """
        # This is a placeholder; in practice, you'd load from the teacher model
        # For now, return None to indicate we should use learned embeddings
        return None

    def token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        return self.tokenizer.convert_tokens_to_ids(token)

    def id_to_token(self, token_id: int) -> str:
        """Convert ID to token."""
        return self.tokenizer.convert_ids_to_tokens(token_id)

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
    ) -> torch.Tensor:
        """
        Batch encode texts.

        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            padding: Whether to pad

        Returns:
            Token IDs tensor (batch_size, seq_len)
        """
        return self.encode(
            texts,
            return_tensors="pt",
            padding=padding,
            truncation=True if max_length else False,
            max_length=max_length,
        )

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Batch decode token IDs.

        Args:
            token_ids: Token IDs tensor (batch_size, seq_len)
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded texts
        """
        return self.tokenizer.batch_decode(
            token_ids.tolist(),
            skip_special_tokens=skip_special_tokens,
        )


def test_tokenizer():
    """Test the tokenizer adapter."""
    adapter = TokenizerAdapter()

    print(f"Vocab size: {adapter.vocab_size}")

    # Test encode/decode
    text = "Hello, world! This is a test."
    token_ids = adapter.encode(text)
    print(f"\nOriginal: {text}")
    print(f"Token IDs: {token_ids}")
    print(f"Shape: {token_ids.shape}")

    decoded = adapter.decode(token_ids[0])
    print(f"Decoded: {decoded}")

    # Test batch
    texts = ["First sentence.", "Second sentence.", "Third one."]
    batch_ids = adapter.batch_encode(texts, padding=True)
    print(f"\nBatch shape: {batch_ids.shape}")

    batch_decoded = adapter.batch_decode(batch_ids)
    print(f"Batch decoded: {batch_decoded}")


if __name__ == "__main__":
    test_tokenizer()
