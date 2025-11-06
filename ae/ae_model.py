"""
Latent autoencoder: maps K tokens to 1 latent vector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class LatentAutoencoder(nn.Module):
    """
    Autoencoder that compresses K tokens into a single latent vector.

    Architecture:
    - Encoder: K token embeddings -> 1 latent vector
    - Decoder: 1 latent vector -> K token logits
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        latent_dim: int = 1024,
        k: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize autoencoder.

        Args:
            vocab_size: Size of vocabulary
            embed_dim: Token embedding dimension
            latent_dim: Latent vector dimension
            k: Number of tokens per latent (compression factor)
            num_encoder_layers: Number of encoder transformer layers
            num_decoder_layers: Number of decoder transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.k = k
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        # Token embeddings (shared between encoder and decoder)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(k, embed_dim)

        # Encoder: K tokens -> 1 latent
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # Projection to latent space
        self.to_latent = nn.Sequential(
            nn.Linear(embed_dim * k, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # Decoder: 1 latent -> K tokens
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, embed_dim * k),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # Output projection to vocabulary
        self.to_vocab = nn.Linear(embed_dim, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Xavier initialization for embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode K tokens to a latent vector.

        Args:
            token_ids: Token IDs (batch_size, k)

        Returns:
            Latent vectors (batch_size, latent_dim)
        """
        batch_size, seq_len = token_ids.shape
        assert seq_len == self.k, f"Expected {self.k} tokens, got {seq_len}"

        # Token embeddings
        token_embeds = self.token_embedding(token_ids)  # (B, k, embed_dim)

        # Add positional embeddings
        positions = torch.arange(self.k, device=token_ids.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)  # (1, k, embed_dim)
        x = token_embeds + pos_embeds

        # Encode
        encoded = self.encoder(x)  # (B, k, embed_dim)

        # Flatten and project to latent
        flat = encoded.reshape(batch_size, -1)  # (B, k * embed_dim)
        latent = self.to_latent(flat)  # (B, latent_dim)

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to K token logits.

        Args:
            latent: Latent vectors (batch_size, latent_dim)

        Returns:
            Token logits (batch_size, k, vocab_size)
        """
        batch_size = latent.shape[0]

        # Project from latent to decoder input
        decoder_input = self.from_latent(latent)  # (B, k * embed_dim)
        decoder_input = decoder_input.reshape(
            batch_size, self.k, self.embed_dim
        )  # (B, k, embed_dim)

        # Add positional embeddings
        positions = torch.arange(self.k, device=latent.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        decoder_input = decoder_input + pos_embeds

        # Decode (using decoder_input as both memory and target)
        decoded = self.decoder(
            tgt=decoder_input,
            memory=decoder_input,
        )  # (B, k, embed_dim)

        # Project to vocabulary
        logits = self.to_vocab(decoded)  # (B, k, vocab_size)

        return logits

    def forward(
        self,
        token_ids: torch.Tensor,
        return_latent: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass: encode then decode.

        Args:
            token_ids: Token IDs (batch_size, k)
            return_latent: Whether to return latent vectors

        Returns:
            logits: Token logits (batch_size, k, vocab_size)
            latent: Latent vectors if return_latent=True
        """
        latent = self.encode(token_ids)
        logits = self.decode(latent)

        if return_latent:
            return logits, latent
        else:
            return logits, None

    def reconstruct(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct tokens (greedy decoding).

        Args:
            token_ids: Input token IDs (batch_size, k)

        Returns:
            Reconstructed token IDs (batch_size, k)
        """
        logits, _ = self.forward(token_ids)
        reconstructed = torch.argmax(logits, dim=-1)
        return reconstructed

    def compute_loss(
        self,
        token_ids: torch.Tensor,
        l2_weight: float = 0.001,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute reconstruction loss.

        Args:
            token_ids: Token IDs (batch_size, k)
            l2_weight: Weight for L2 regularization on latent

        Returns:
            loss: Total loss
            metrics: Dictionary of metrics
        """
        logits, latent = self.forward(token_ids, return_latent=True)

        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            token_ids.reshape(-1),
        )

        # L2 regularization on latent
        l2_loss = torch.mean(latent**2)

        # Total loss
        total_loss = ce_loss + l2_weight * l2_loss

        # Compute accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == token_ids).float()
            token_accuracy = correct.mean()
            exact_match = (correct.sum(dim=-1) == self.k).float().mean()

        metrics = {
            "loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "l2_loss": l2_loss.item(),
            "token_accuracy": token_accuracy.item(),
            "exact_match": exact_match.item(),
        }

        return total_loss, metrics


def test_ae():
    """Test the autoencoder."""
    vocab_size = 32000
    k = 8
    batch_size = 4

    # Create model
    ae = LatentAutoencoder(
        vocab_size=vocab_size,
        embed_dim=512,
        latent_dim=1024,
        k=k,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )

    print(f"Model parameters: {sum(p.numel() for p in ae.parameters()):,}")

    # Random tokens
    token_ids = torch.randint(0, vocab_size, (batch_size, k))

    # Encode
    latent = ae.encode(token_ids)
    print(f"\nLatent shape: {latent.shape}")

    # Decode
    logits = ae.decode(latent)
    print(f"Logits shape: {logits.shape}")

    # Reconstruct
    reconstructed = ae.reconstruct(token_ids)
    print(f"Reconstructed shape: {reconstructed.shape}")

    # Loss
    loss, metrics = ae.compute_loss(token_ids)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    test_ae()
