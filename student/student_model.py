"""
Student transformer: predicts next latent vector autoregressively.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class StudentTransformer(nn.Module):
    """
    Student model that predicts next latent vector.

    Architecture: Standard autoregressive transformer operating in latent space.
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dim: int = 2048,
        num_layers: int = 12,
        num_heads: int = 16,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Initialize student transformer.

        Args:
            latent_dim: Dimension of latent vectors
            hidden_dim: Hidden dimension for feedforward
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Input projection (latent -> hidden)
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Positional embeddings
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # Output projection (hidden -> latent)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, latent_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.position_embedding.weight, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        latents: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            latents: Input latent sequence (batch_size, seq_len, latent_dim)
            padding_mask: Padding mask (batch_size, seq_len)

        Returns:
            Predicted next latents (batch_size, seq_len, latent_dim)
        """
        batch_size, seq_len, _ = latents.shape

        # Project to hidden dimension
        x = self.input_proj(latents)  # (B, L, hidden_dim)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=latents.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)  # (1, L, hidden_dim)
        x = x + pos_embeds

        # Create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len,
            device=latents.device,
        )

        # Transform (using x as both target and memory for autoregressive)
        # Note: TransformerDecoder expects (tgt, memory)
        # For autoregressive, we use same sequence as both
        transformed = self.transformer(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )

        # Project to latent space
        output = self.output_proj(transformed)  # (B, L, latent_dim)

        return output

    def predict_next(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next latent given sequence of latents.

        Args:
            latents: Input latent sequence (batch_size, seq_len, latent_dim)

        Returns:
            Next latent (batch_size, latent_dim)
        """
        output = self.forward(latents)
        # Return last position prediction
        next_latent = output[:, -1, :]
        return next_latent

    def generate(
        self,
        initial_latent: torch.Tensor,
        num_steps: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate sequence of latents autoregressively.

        Args:
            initial_latent: Initial latent (batch_size, latent_dim)
            num_steps: Number of steps to generate
            temperature: Sampling temperature (not used in deterministic mode)

        Returns:
            Generated latent sequence (batch_size, num_steps, latent_dim)
        """
        batch_size = initial_latent.shape[0]
        device = initial_latent.device

        # Start with initial latent
        current_seq = initial_latent.unsqueeze(1)  # (B, 1, latent_dim)

        generated = [initial_latent]

        for step in range(num_steps - 1):
            # Predict next
            next_latent = self.predict_next(current_seq)

            # Add to sequence
            generated.append(next_latent)
            current_seq = torch.cat(
                [current_seq, next_latent.unsqueeze(1)],
                dim=1,
            )

            # Truncate if exceeds max_seq_len
            if current_seq.shape[1] > self.max_seq_len:
                current_seq = current_seq[:, -self.max_seq_len :, :]

        # Stack generated latents
        output = torch.stack(generated, dim=1)  # (B, num_steps, latent_dim)

        return output


def test_student():
    """Test the student model."""
    latent_dim = 1024
    batch_size = 4
    seq_len = 16

    # Create model
    model = StudentTransformer(
        latent_dim=latent_dim,
        hidden_dim=2048,
        num_layers=6,
        num_heads=16,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Random latents
    latents = torch.randn(batch_size, seq_len, latent_dim)

    # Forward
    output = model(latents)
    print(f"\nInput shape: {latents.shape}")
    print(f"Output shape: {output.shape}")

    # Predict next
    next_latent = model.predict_next(latents)
    print(f"Next latent shape: {next_latent.shape}")

    # Generate
    initial = torch.randn(batch_size, latent_dim)
    generated = model.generate(initial, num_steps=10)
    print(f"Generated shape: {generated.shape}")


if __name__ == "__main__":
    test_student()
