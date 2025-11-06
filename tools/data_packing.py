#!/usr/bin/env python3
"""
Pack text data into sequences for training.
Sequences are packed to be multiples of K (chunk size).
"""
import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Iterator
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from ae.tokenizer_adapter import TokenizerAdapter


def read_jsonl(file_path: str) -> Iterator[str]:
    """
    Read JSONL file and yield text.

    Args:
        file_path: Path to JSONL file

    Yields:
        Text strings
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Try common text fields
                text = data.get("text") or data.get("content") or data.get("input") or str(data)
                if text:
                    yield text
            except json.JSONDecodeError:
                continue


def read_text(file_path: str) -> Iterator[str]:
    """
    Read plain text file.

    Args:
        file_path: Path to text file

    Yields:
        Text strings (by line or paragraph)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def pack_sequences(
    texts: Iterator[str],
    tokenizer: TokenizerAdapter,
    seq_len: int,
    k: int,
    max_sequences: int = None,
) -> List[torch.Tensor]:
    """
    Pack texts into sequences of length seq_len (multiple of k).

    Args:
        texts: Iterator of text strings
        tokenizer: Tokenizer adapter
        seq_len: Target sequence length (must be multiple of k)
        k: Chunk size
        max_sequences: Maximum number of sequences

    Returns:
        List of token sequences
    """
    assert seq_len % k == 0, f"seq_len must be multiple of k ({k})"

    sequences = []
    current_tokens = []

    for text in tqdm(texts, desc="Packing"):
        # Tokenize
        token_ids = tokenizer.encode(text, return_tensors=None, add_special_tokens=False)

        if isinstance(token_ids, list):
            tokens = token_ids
        else:
            tokens = token_ids[0].tolist()

        # Add to current buffer
        current_tokens.extend(tokens)

        # Pack into sequences of seq_len
        while len(current_tokens) >= seq_len:
            sequence = torch.tensor(current_tokens[:seq_len], dtype=torch.long)
            sequences.append(sequence)
            current_tokens = current_tokens[seq_len:]

            if max_sequences and len(sequences) >= max_sequences:
                return sequences

    # Handle remaining tokens (pad to seq_len if needed)
    if current_tokens:
        # Pad to nearest multiple of k
        remainder = len(current_tokens) % k
        if remainder != 0:
            pad_len = k - remainder
            current_tokens.extend([tokenizer.tokenizer.pad_token_id] * pad_len)

        # Create final sequence if long enough
        if len(current_tokens) >= k:
            # Pad to seq_len
            if len(current_tokens) < seq_len:
                pad_len = seq_len - len(current_tokens)
                current_tokens.extend([tokenizer.tokenizer.pad_token_id] * pad_len)

            sequence = torch.tensor(current_tokens[:seq_len], dtype=torch.long)
            sequences.append(sequence)

    return sequences


def main():
    parser = argparse.ArgumentParser(description="Pack text data into sequences")
    parser.add_argument("--input", type=str, required=True, help="Input file (JSONL or text)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--k", type=int, default=8, help="Chunk size")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--max_sequences", type=int, help="Maximum sequences")
    parser.add_argument("--format", choices=["jsonl", "text"], default="jsonl", help="Input format")
    args = parser.parse_args()

    print(f"Packing data from {args.input}...")
    print(f"Sequence length: {args.seq_len} (must be multiple of k={args.k})")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = TokenizerAdapter()

    # Read texts
    if args.format == "jsonl":
        texts = read_jsonl(args.input)
    else:
        texts = read_text(args.input)

    # Pack sequences
    print("Packing sequences...")
    sequences = pack_sequences(
        texts,
        tokenizer,
        seq_len=args.seq_len,
        k=args.k,
        max_sequences=args.max_sequences,
    )

    print(f"Packed {len(sequences)} sequences")

    # Split into train/val
    val_size = int(len(sequences) * args.val_split)
    train_size = len(sequences) - val_size

    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:]

    print(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}")

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train"
    val_path = output_dir / "val"
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)

    # Save train
    torch.save(train_sequences, train_path / "sequences.pt")
    print(f"Saved train sequences to {train_path / 'sequences.pt'}")

    # Save val
    if val_sequences:
        torch.save(val_sequences, val_path / "sequences.pt")
        print(f"Saved val sequences to {val_path / 'sequences.pt'}")

    # Save metadata
    metadata = {
        "seq_len": args.seq_len,
        "k": args.k,
        "train_size": len(train_sequences),
        "val_size": len(val_sequences),
        "vocab_size": tokenizer.vocab_size,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Data packing complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
