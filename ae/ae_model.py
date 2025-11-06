"""
Autoencoder for compressing K tokens into a single latent vector.
Based on CALM architecture with VAE formulation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class MLP(nn.Module):
    """MLP block with SiLU activation."""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        # Gated MLP: SiLU(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        if self.dropout:
            hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class AELayer(nn.Module):
    """Transformer-style layer with LayerNorm + MLP + residual."""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size, dropout)

    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.mlp(x)
        return residual + x


class AutoEncoder(nn.Module):
    """
    Variational Autoencoder that compresses K tokens into 1 latent vector.

    Architecture:
        Encoder: Tokens -> Embeddings -> Patches -> Layers -> Squeeze -> Latent (mean, logvar)
        Decoder: Latent -> Expand -> Layers -> Token Logits

    Args:
        vocab_size: Size of token vocabulary
        K: Number of tokens to compress (patch_size)
        D: Latent dimension
        hidden_size: Hidden dimension for encoder/decoder
        intermediate_size: MLP intermediate dimension
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dropout: Dropout rate
        kl_weight: Weight for KL divergence loss
        kl_clamp: Minimum KL divergence (prevents collapse)
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        K: int = 8,
        D: int = 1024,
        hidden_size: int = 512,
        intermediate_size: int = 1280,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.15,
        kl_weight: float = 1e-3,
        kl_clamp: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.K = K
        self.D = D
        self.hidden_size = hidden_size
        self.kl_weight = kl_weight
        self.kl_clamp = kl_clamp

        # Token embeddings (shared between encoder and decoder via weight tying)
        self.embeddings = nn.Embedding(vocab_size, hidden_size)

        # Encoder
        self.encoder_layers_1 = nn.ModuleList([
            AELayer(hidden_size, intermediate_size, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Squeeze: Compress K patches into 1
        self.squeeze = nn.Linear(K * hidden_size, hidden_size, bias=False)
        self.squeeze_norm = RMSNorm(hidden_size)

        self.encoder_layers_2 = nn.ModuleList([
            AELayer(hidden_size, intermediate_size, dropout)
            for _ in range(num_encoder_layers)
        ])

        # VAE: Project to mean and logvar
        self.to_mean = nn.Linear(hidden_size, D, bias=False)
        self.to_logvar = nn.Linear(hidden_size, D, bias=False)

        # Decoder
        self.from_latent = nn.Linear(D, hidden_size, bias=False)

        self.decoder_layers_1 = nn.ModuleList([
            AELayer(hidden_size, intermediate_size, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Expand: Decompress 1 into K patches
        self.expand = nn.Linear(hidden_size, K * hidden_size, bias=False)
        self.expand_norm = RMSNorm(K * hidden_size)

        self.decoder_layers_2 = nn.ModuleList([
            AELayer(hidden_size, intermediate_size, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection (weight-tied with embeddings)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.weight  # Weight tying

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode tokens to latent distribution parameters.

        Args:
            input_ids: [batch, seq_len] where seq_len % K == 0

        Returns:
            mean: [batch, seq_len // K, D]
            logvar: [batch, seq_len // K, D]
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len % self.K == 0, f"seq_len {seq_len} must be divisible by K={self.K}"

        # Embed tokens: [batch, seq_len, hidden]
        x = self.embeddings(input_ids)

        # Encoder layers stage 1
        for layer in self.encoder_layers_1:
            x = layer(x)

        # Reshape into patches: [batch, seq_len // K, K * hidden]
        num_patches = seq_len // self.K
        x = x.view(batch_size, num_patches, self.K * self.hidden_size)

        # Squeeze: [batch, num_patches, hidden]
        x = self.squeeze(x)
        x = self.squeeze_norm(x)

        # Encoder layers stage 2
        for layer in self.encoder_layers_2:
            x = layer(x)

        # Project to latent distribution
        mean = self.to_mean(x)  # [batch, num_patches, D]
        logvar = self.to_logvar(x)  # [batch, num_patches, D]

        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to token logits.

        Args:
            z: [batch, num_patches, D]

        Returns:
            logits: [batch, num_patches * K, vocab_size]
        """
        batch_size, num_patches, _ = z.shape

        # Project from latent: [batch, num_patches, hidden]
        x = self.from_latent(z)

        # Decoder layers stage 1
        for layer in self.decoder_layers_1:
            x = layer(x)

        # Expand: [batch, num_patches, K * hidden]
        x = self.expand(x)
        x = self.expand_norm(x)

        # Reshape to tokens: [batch, num_patches * K, hidden]
        x = x.view(batch_size, num_patches * self.K, self.hidden_size)

        # Decoder layers stage 2
        for layer in self.decoder_layers_2:
            x = layer(x)

        # Project to vocabulary: [batch, seq_len, vocab_size]
        logits = self.lm_head(x)

        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass with reconstruction and KL loss.

        Args:
            input_ids: [batch, seq_len]
            labels: [batch, seq_len] (if None, uses input_ids)

        Returns:
            Dictionary with:
                - logits: [batch, seq_len, vocab_size]
                - loss: total loss
                - recon_loss: reconstruction loss
                - kl_loss: KL divergence loss
                - mean: latent means
                - logvar: latent logvars
        """
        if labels is None:
            labels = input_ids

        # Encode
        mean, logvar = self.encode(input_ids)

        # Sample latent
        z = self.reparameterize(mean, logvar)

        # Decode
        logits = self.decode(z)

        # Reconstruction loss (token cross-entropy)
        recon_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            reduction='mean'
        ) * self.K  # Scale by K as in CALM

        # KL divergence loss (regularize towards standard normal)
        # KL(N(mean, var) || N(0, 1)) = -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
        kl_loss = kl_loss.mean()

        # KL clamping (prevent collapse)
        kl_loss = torch.clamp(kl_loss, min=self.kl_clamp)

        # Total loss
        loss = recon_loss + self.kl_weight * kl_loss

        return {
            'logits': logits,
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'mean': mean,
            'logvar': logvar,
            'z': z,
        }

    @torch.no_grad()
    def reconstruct(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct tokens (deterministic, using mean).

        Args:
            input_ids: [batch, seq_len]

        Returns:
            reconstructed_ids: [batch, seq_len]
        """
        mean, _ = self.encode(input_ids)
        logits = self.decode(mean)  # Use mean, not sample
        return logits.argmax(dim=-1)

    def compute_exact_match(self, input_ids: torch.Tensor) -> float:
        """
        Compute exact reconstruction accuracy.

        Args:
            input_ids: [batch, seq_len]

        Returns:
            exact_match: fraction of tokens perfectly reconstructed
        """
        reconstructed = self.reconstruct(input_ids)
        return (reconstructed == input_ids).float().mean().item()
