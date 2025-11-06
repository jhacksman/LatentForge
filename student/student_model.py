"""
Student autoregressive model that predicts next latent vectors.
Uses a simplified Transformer architecture with knowledge distillation.
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


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_emb(q, k, cos, sin):
    """Apply rotary embeddings to queries and keys."""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention with RoPE."""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, attention_mask=None, use_cache=False, past_kv=None):
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            offset = past_k.shape[2]
        else:
            offset = 0

        cos, sin = self.rope(k.shape[2], device=x.device)
        q_pos = q if offset == 0 else torch.cat([torch.zeros_like(q), q], dim=2)[:, :, -seq_len:]
        cos_q, sin_q = cos[-seq_len:], sin[-seq_len:]
        q, k = apply_rotary_emb(q, k, cos_q.unsqueeze(0).unsqueeze(0), sin_q.unsqueeze(0).unsqueeze(0))

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout:
            attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)

        if use_cache:
            return output, (k, v)
        return output


class MLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        if self.dropout:
            hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, dropout)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size, dropout)

    def forward(self, x, attention_mask=None, use_cache=False, past_kv=None):
        # Self attention with residual
        residual = x
        x = self.input_layernorm(x)
        if use_cache:
            attn_output, kv_cache = self.self_attn(x, attention_mask, use_cache, past_kv)
        else:
            attn_output = self.self_attn(x, attention_mask, use_cache, past_kv)
            kv_cache = None
        x = residual + attn_output

        # MLP with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)

        if use_cache:
            return x, kv_cache
        return x


class StudentModel(nn.Module):
    """
    Student autoregressive model that predicts next latent vectors.

    Architecture:
        Input: Latent vectors from AE encoder
        Transformer layers with RoPE
        Output: Predicted next latent vector

    Args:
        latent_dim: Dimension of latent vectors (D)
        hidden_size: Hidden dimension of transformer
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        intermediate_size: MLP intermediate dimension
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
    """
    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 2048,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Project latent vectors to hidden dimension
        self.latent_proj = nn.Linear(latent_dim, hidden_size, bias=False)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(hidden_size)

        # Output projection to latent dimension
        self.output_proj = nn.Linear(hidden_size, latent_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        latent_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple] = None,
    ):
        """
        Forward pass.

        Args:
            latent_input: [batch, seq_len, latent_dim]
            attention_mask: [batch, 1, seq_len, seq_len]
            use_cache: Whether to return KV cache
            past_key_values: Past KV cache for incremental generation

        Returns:
            predicted_latent: [batch, seq_len, latent_dim]
            past_key_values: KV cache (if use_cache=True)
        """
        batch_size, seq_len, _ = latent_input.shape

        # Project to hidden dimension
        hidden_states = self.latent_proj(latent_input)

        # Create causal mask if not provided
        if attention_mask is None:
            # Causal mask: [1, 1, seq_len, seq_len]
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device),
                diagonal=1
            )
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Apply transformer layers
        new_kv_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            if use_cache:
                hidden_states, kv = layer(hidden_states, attention_mask, use_cache, past_kv)
                new_kv_cache.append(kv)
            else:
                hidden_states = layer(hidden_states, attention_mask, use_cache, past_kv)

        # Final norm and projection
        hidden_states = self.norm(hidden_states)
        predicted_latent = self.output_proj(hidden_states)

        if use_cache:
            return predicted_latent, tuple(new_kv_cache)
        return predicted_latent

    def get_last_hidden(self, latent_input: torch.Tensor) -> torch.Tensor:
        """
        Get the last hidden state for generation.

        Args:
            latent_input: [batch, seq_len, latent_dim]

        Returns:
            last_hidden: [batch, latent_dim]
        """
        predicted = self.forward(latent_input, use_cache=False)
        return predicted[:, -1, :]  # Get last position
