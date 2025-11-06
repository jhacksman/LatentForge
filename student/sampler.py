"""
Sampler for generating text with student model + autoencoder.
"""
import torch
import torch.nn.functional as F
from typing import Optional, List
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from student.student_model import StudentTransformer
from ae.ae_model import LatentAutoencoder
from ae.tokenizer_adapter import TokenizerAdapter


class LatentSampler:
    """Sampler that generates via latent space."""

    def __init__(
        self,
        student: StudentTransformer,
        autoencoder: LatentAutoencoder,
        tokenizer: TokenizerAdapter,
        device: torch.device,
    ):
        """
        Initialize sampler.

        Args:
            student: Student transformer model
            autoencoder: Latent autoencoder
            tokenizer: Tokenizer adapter
            device: Device to run on
        """
        self.student = student
        self.autoencoder = autoencoder
        self.tokenizer = tokenizer
        self.device = device
        self.k = autoencoder.k

        # Set to eval mode
        self.student.eval()
        self.autoencoder.eval()

    @torch.no_grad()
    def sample(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            seed: Random seed

        Returns:
            Generated text
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Encode prompt to tokens
        prompt_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.device)

        # Pad to multiple of k
        prompt_len = prompt_tokens.shape[1]
        if prompt_len % self.k != 0:
            pad_len = self.k - (prompt_len % self.k)
            pad_tokens = torch.full(
                (1, pad_len),
                self.tokenizer.tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            prompt_tokens = torch.cat([prompt_tokens, pad_tokens], dim=1)

        # Encode prompt to latents
        initial_latents = []
        for i in range(0, prompt_tokens.shape[1], self.k):
            chunk = prompt_tokens[:, i : i + self.k]
            latent = self.autoencoder.encode(chunk)
            initial_latents.append(latent)

        initial_latents = torch.stack(initial_latents, dim=1)  # (1, num_chunks, latent_dim)

        # Calculate how many latent steps to generate
        num_latent_steps = (max_new_tokens + self.k - 1) // self.k

        # Generate latents autoregressively
        current_seq = initial_latents

        for step in range(num_latent_steps):
            # Predict next latent
            next_latent = self.student.predict_next(current_seq)

            # Add to sequence
            current_seq = torch.cat(
                [current_seq, next_latent.unsqueeze(1)],
                dim=1,
            )

            # Truncate if exceeds max_seq_len
            if current_seq.shape[1] > self.student.max_seq_len:
                current_seq = current_seq[:, -self.student.max_seq_len :, :]

        # Decode all latents to tokens
        all_tokens = []
        for i in range(current_seq.shape[1]):
            latent = current_seq[:, i, :]
            logits = self.autoencoder.decode(latent)  # (1, k, vocab_size)

            # Sample tokens
            if temperature > 0:
                # Apply temperature
                scaled_logits = logits / temperature

                # Apply top-k
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)
                    scaled_logits = torch.full_like(scaled_logits, float("-inf"))
                    scaled_logits.scatter_(-1, top_k_indices, top_k_logits)

                # Apply top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 0] = False  # Keep at least one token

                    for batch_idx in range(scaled_logits.shape[0]):
                        for pos_idx in range(scaled_logits.shape[1]):
                            indices_to_remove = sorted_indices[batch_idx, pos_idx][
                                sorted_indices_to_remove[batch_idx, pos_idx]
                            ]
                            scaled_logits[batch_idx, pos_idx, indices_to_remove] = float("-inf")

                # Sample
                probs = F.softmax(scaled_logits, dim=-1)
                tokens = torch.multinomial(
                    probs.view(-1, probs.shape[-1]),
                    num_samples=1,
                ).view(1, self.k)
            else:
                # Greedy
                tokens = torch.argmax(logits, dim=-1)

            all_tokens.append(tokens)

        # Concatenate all tokens
        all_tokens = torch.cat(all_tokens, dim=1)

        # Truncate to max_new_tokens after prompt
        total_tokens = all_tokens[:, : prompt_len + max_new_tokens]

        # Decode to text
        generated_text = self.tokenizer.decode(total_tokens[0], skip_special_tokens=True)

        return generated_text

    @torch.no_grad()
    def sample_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        **kwargs,
    ) -> List[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum new tokens per prompt
            **kwargs: Additional sampling parameters

        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            result = self.sample(prompt, max_new_tokens=max_new_tokens, **kwargs)
            results.append(result)
        return results


def load_models(
    ae_checkpoint: str,
    student_checkpoint: str,
    device: torch.device,
):
    """
    Load models from checkpoints.

    Args:
        ae_checkpoint: Path to AE checkpoint
        student_checkpoint: Path to student checkpoint
        device: Device to load on

    Returns:
        Tuple of (student, autoencoder, tokenizer)
    """
    # Load tokenizer
    tokenizer = TokenizerAdapter()

    # Load AE
    ae_ckpt = torch.load(ae_checkpoint, map_location=device)
    ae_config = ae_ckpt["config"]
    autoencoder = LatentAutoencoder(**ae_config)
    autoencoder.load_state_dict(ae_ckpt["model"])
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    # Load student
    student_ckpt = torch.load(student_checkpoint, map_location=device)
    student_config = student_ckpt["config"]
    student = StudentTransformer(**student_config)
    student.load_state_dict(student_ckpt["model"])
    student = student.to(device)
    student.eval()

    return student, autoencoder, tokenizer


def test_sampler():
    """Test the sampler."""
    # This is a placeholder test
    print("Sampler test requires trained models.")
    print("Use infer.py to test with actual checkpoints.")


if __name__ == "__main__":
    test_sampler()
