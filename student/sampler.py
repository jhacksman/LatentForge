"""
Sampling utilities for generating text via latent-space autoregression.
"""
import torch
import torch.nn.functional as F
from typing import Optional
import math


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """
    Filter logits using top-k and/or nucleus (top-p) filtering.

    Args:
        logits: [batch, vocab_size]
        top_k: Keep only top k tokens (0 = disabled)
        top_p: Keep tokens with cumulative probability >= top_p (1.0 = disabled)
        filter_value: Value to replace filtered logits with

    Returns:
        filtered_logits: [batch, vocab_size]
    """
    if top_k > 0:
        # Remove all tokens with rank > k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        # Sort and compute cumulative probabilities
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value

    return logits


def sample_tokens(logits, temperature=1.0, top_k=0, top_p=1.0):
    """
    Sample tokens from logits with temperature, top-k, and top-p.

    Args:
        logits: [batch, vocab_size]
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling

    Returns:
        sampled_ids: [batch]
    """
    if temperature == 0:
        # Greedy
        return logits.argmax(dim=-1)

    logits = logits / temperature
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate_latent_ar(
    student_model,
    ae_model,
    input_ids,
    max_new_tokens=128,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    eos_token_id=2,
):
    """
    Generate text using latent-space autoregression.

    Process:
    1. Encode input tokens to latent sequence
    2. Autoregressively predict next latent vectors
    3. Decode each predicted latent to K tokens
    4. Continue until max_new_tokens or EOS

    Args:
        student_model: Student AR model
        ae_model: Autoencoder
        input_ids: [batch, seq_len] input tokens
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling
        eos_token_id: EOS token ID

    Returns:
        generated_ids: [batch, seq_len + num_generated]
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]
    K = ae_model.K

    # Pad input to multiple of K
    seq_len = input_ids.shape[1]
    remainder = seq_len % K
    if remainder != 0:
        pad_len = K - remainder
        input_ids = F.pad(input_ids, (0, pad_len), value=0)

    # Encode initial tokens to latent sequence
    mean, _ = ae_model.encode(input_ids)  # [batch, num_patches, D]
    latent_sequence = mean  # Use deterministic (mean) encoding

    # Keep track of generated tokens
    generated_ids = input_ids.clone()

    # Calculate how many latent steps to generate
    max_latent_steps = (max_new_tokens + K - 1) // K

    for step in range(max_latent_steps):
        # Predict next latent vector
        next_latent = student_model.get_last_hidden(latent_sequence)  # [batch, D]
        next_latent = next_latent.unsqueeze(1)  # [batch, 1, D]

        # Decode to K tokens
        token_logits = ae_model.decode(next_latent)  # [batch, K, vocab_size]

        # Sample tokens for this patch
        patch_tokens = []
        for k in range(K):
            logits = token_logits[:, k, :]  # [batch, vocab_size]
            sampled = sample_tokens(logits, temperature, top_k, top_p)  # [batch]
            patch_tokens.append(sampled)

        patch_tokens = torch.stack(patch_tokens, dim=1)  # [batch, K]

        # Append to generated sequence
        generated_ids = torch.cat([generated_ids, patch_tokens], dim=1)

        # Check for EOS
        if (patch_tokens == eos_token_id).any():
            break

        # Append latent to sequence for next prediction
        latent_sequence = torch.cat([latent_sequence, next_latent], dim=1)

    # Trim to max_new_tokens
    original_len = seq_len
    if generated_ids.shape[1] > original_len + max_new_tokens:
        generated_ids = generated_ids[:, :original_len + max_new_tokens]

    return generated_ids


@torch.no_grad()
def generate_with_kv_cache(
    student_model,
    ae_model,
    input_ids,
    max_new_tokens=128,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    eos_token_id=2,
):
    """
    Generate text using KV caching for efficiency.

    This is more efficient for long sequences as it caches
    past key-value pairs in the transformer.

    Args:
        Same as generate_latent_ar

    Returns:
        generated_ids: [batch, seq_len + num_generated]
    """
    device = input_ids.device
    K = ae_model.K

    # Pad input to multiple of K
    seq_len = input_ids.shape[1]
    remainder = seq_len % K
    if remainder != 0:
        pad_len = K - remainder
        input_ids = F.pad(input_ids, (0, pad_len), value=0)

    # Encode initial sequence
    mean, _ = ae_model.encode(input_ids)
    latent_sequence = mean

    # Initial forward pass with caching
    _, past_kv = student_model(latent_sequence, use_cache=True)

    generated_ids = input_ids.clone()
    max_latent_steps = (max_new_tokens + K - 1) // K

    for step in range(max_latent_steps):
        # Get last position prediction (only last latent)
        last_latent = latent_sequence[:, -1:, :]  # [batch, 1, D]

        # Forward with cache
        next_latent, past_kv = student_model(
            last_latent,
            use_cache=True,
            past_key_values=past_kv
        )

        # Decode to tokens
        token_logits = ae_model.decode(next_latent)  # [batch, K, vocab_size]

        # Sample patch
        patch_tokens = []
        for k in range(K):
            logits = token_logits[:, k, :]
            sampled = sample_tokens(logits, temperature, top_k, top_p)
            patch_tokens.append(sampled)

        patch_tokens = torch.stack(patch_tokens, dim=1)
        generated_ids = torch.cat([generated_ids, patch_tokens], dim=1)

        # Check EOS
        if (patch_tokens == eos_token_id).any():
            break

        # Update latent sequence
        latent_sequence = torch.cat([latent_sequence, next_latent], dim=1)

    # Trim
    original_len = seq_len
    if generated_ids.shape[1] > original_len + max_new_tokens:
        generated_ids = generated_ids[:, :original_len + max_new_tokens]

    return generated_ids
