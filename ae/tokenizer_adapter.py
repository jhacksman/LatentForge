"""
Utilities for working with tokenizers and preparing data for training.
"""
import torch
from typing import List, Dict, Any
from transformers import PreTrainedTokenizer


def pad_to_multiple(input_ids: torch.Tensor, K: int, pad_token_id: int = 0) -> torch.Tensor:
    """
    Pad sequence length to be a multiple of K.

    Args:
        input_ids: [batch, seq_len]
        K: patch size
        pad_token_id: token to use for padding

    Returns:
        padded_ids: [batch, padded_seq_len] where padded_seq_len % K == 0
    """
    batch_size, seq_len = input_ids.shape
    remainder = seq_len % K

    if remainder == 0:
        return input_ids

    pad_len = K - remainder
    padding = torch.full((batch_size, pad_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat([input_ids, padding], dim=1)


def tokenize_function(examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizer) -> Dict[str, List[List[int]]]:
    """
    Tokenize text examples.

    Args:
        examples: Dictionary with 'text' key
        tokenizer: HuggingFace tokenizer

    Returns:
        Dictionary with 'input_ids' key
    """
    return tokenizer(examples["text"], truncation=False, padding=False)


def group_texts(examples: Dict[str, List[List[int]]], block_size: int, K: int, eos_token_id: int) -> Dict[str, List[List[int]]]:
    """
    Concatenate and pack texts into fixed-size blocks.

    Args:
        examples: Dictionary with 'input_ids' key
        block_size: Size of each training block
        K: patch size (for padding)
        eos_token_id: Token to insert between documents

    Returns:
        Dictionary with packed 'input_ids' and 'labels'
    """
    # Concatenate all texts with EOS tokens between documents
    concatenated_ids = []
    for ids in examples['input_ids']:
        concatenated_ids.extend(ids)
        concatenated_ids.append(eos_token_id)

    # Pad to multiple of K
    total_length = len(concatenated_ids)
    remainder = total_length % K
    if remainder != 0:
        concatenated_ids.extend([0] * (K - remainder))
        total_length = len(concatenated_ids)

    # Split into blocks
    total_length = (total_length // block_size) * block_size
    result = {
        'input_ids': [
            concatenated_ids[i:i + block_size]
            for i in range(0, total_length, block_size)
        ]
    }
    result['labels'] = result['input_ids'].copy()

    return result


def prepare_dataset(dataset, tokenizer: PreTrainedTokenizer, block_size: int, K: int, num_proc: int = 4):
    """
    Prepare dataset for training: tokenize and pack.

    Args:
        dataset: HuggingFace dataset with 'text' field
        tokenizer: HuggingFace tokenizer
        block_size: Size of training blocks
        K: patch size
        num_proc: Number of processes for parallel tokenization

    Returns:
        Processed dataset ready for training
    """
    # Tokenize
    tokenized = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Group into blocks
    grouped = tokenized.map(
        lambda examples: group_texts(
            examples,
            block_size=block_size,
            K=K,
            eos_token_id=tokenizer.eos_token_id or 2
        ),
        batched=True,
        num_proc=num_proc,
        desc="Grouping texts",
    )

    return grouped


def compute_reconstruction_accuracy(model, dataloader, device: str = 'cuda') -> Dict[str, float]:
    """
    Compute reconstruction metrics on a dataset.

    Args:
        model: AutoEncoder model
        dataloader: DataLoader with tokenized data
        device: Device to run on

    Returns:
        Dictionary with metrics:
            - exact_match: fraction of tokens exactly reconstructed
            - patch_exact_match: fraction of patches (K tokens) exactly reconstructed
    """
    model.eval()
    total_tokens = 0
    correct_tokens = 0
    total_patches = 0
    correct_patches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            batch_size, seq_len = input_ids.shape

            # Reconstruct
            reconstructed = model.reconstruct(input_ids)

            # Token-level accuracy
            matches = (reconstructed == input_ids)
            correct_tokens += matches.sum().item()
            total_tokens += matches.numel()

            # Patch-level accuracy
            K = model.K
            num_patches = seq_len // K
            for i in range(num_patches):
                start = i * K
                end = start + K
                patch_matches = matches[:, start:end].all(dim=1)
                correct_patches += patch_matches.sum().item()
                total_patches += batch_size

    return {
        'exact_match': correct_tokens / total_tokens if total_tokens > 0 else 0.0,
        'patch_exact_match': correct_patches / total_patches if total_patches > 0 else 0.0,
    }
