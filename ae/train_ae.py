#!/usr/bin/env python3
"""
Train the latent autoencoder.
"""
import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

from ae_model import LatentAutoencoder
from tokenizer_adapter import TokenizerAdapter


class TokenDataset(Dataset):
    """Dataset of token sequences."""

    def __init__(self, data_path: str, k: int):
        """
        Initialize dataset.

        Args:
            data_path: Path to packed data directory
            k: Chunk size
        """
        self.k = k
        self.sequences = []

        # Load packed sequences
        data_file = Path(data_path) / "sequences.pt"
        if data_file.exists():
            self.sequences = torch.load(data_file)
        else:
            raise FileNotFoundError(f"No data found at {data_file}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Split into chunks of size k
        # Return a random chunk
        if len(seq) < self.k:
            # Pad if needed
            padded = torch.cat([seq, torch.zeros(self.k - len(seq), dtype=torch.long)])
            return padded
        else:
            # Random start position
            start = torch.randint(0, len(seq) - self.k + 1, (1,)).item()
            return seq[start : start + self.k]


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    l2_weight: float = 0.001,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_metrics = {
        "ce_loss": 0.0,
        "l2_loss": 0.0,
        "token_accuracy": 0.0,
        "exact_match": 0.0,
    }
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        token_ids = batch.to(device)

        optimizer.zero_grad()

        # Forward
        loss, metrics = model.compute_loss(token_ids, l2_weight=l2_weight)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        for key in total_metrics:
            total_metrics[key] += metrics.get(key, 0.0)
        num_batches += 1

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{metrics['token_accuracy']:.3f}",
                "exact": f"{metrics['exact_match']:.3f}",
            }
        )

    # Average
    avg_metrics = {
        "loss": total_loss / num_batches,
        **{k: v / num_batches for k, v in total_metrics.items()},
    }

    return avg_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    l2_weight: float = 0.001,
) -> dict:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_metrics = {
        "ce_loss": 0.0,
        "l2_loss": 0.0,
        "token_accuracy": 0.0,
        "exact_match": 0.0,
    }
    num_batches = 0

    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        token_ids = batch.to(device)

        # Forward
        loss, metrics = model.compute_loss(token_ids, l2_weight=l2_weight)

        # Accumulate
        total_loss += loss.item()
        for key in total_metrics:
            total_metrics[key] += metrics.get(key, 0.0)
        num_batches += 1

    # Average
    avg_metrics = {
        "loss": total_loss / num_batches,
        **{k: v / num_batches for k, v in total_metrics.items()},
    }

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train latent autoencoder")
    parser.add_argument("--data", type=str, required=True, help="Data directory")
    parser.add_argument("--k", type=int, default=8, help="Chunk size")
    parser.add_argument("--latent_dim", type=int, default=1024, help="Latent dimension")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--l2_weight", type=float, default=0.001, help="L2 weight")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--config", type=str, help="Config file (YAML)")
    args = parser.parse_args()

    # Load config if provided
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # BF16 support
    use_bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"Using dtype: {dtype}")

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = TokenizerAdapter()
    vocab_size = tokenizer.vocab_size

    # Model
    print("Creating model...")
    model = LatentAutoencoder(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        k=args.k,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
    )
    model = model.to(device).to(dtype)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Data
    print("Loading data...")
    train_dataset = TokenDataset(args.data, args.k)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    best_exact_match = 0.0

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, l2_weight=args.l2_weight
        )

        print(f"\nTrain metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")

        # Save checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_path = Path(args.checkpoint_dir) / "ae.pt"

        # Save if best
        if train_metrics["exact_match"] > best_exact_match:
            best_exact_match = train_metrics["exact_match"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "vocab_size": vocab_size,
                        "embed_dim": args.embed_dim,
                        "latent_dim": args.latent_dim,
                        "k": args.k,
                        "num_encoder_layers": args.num_layers,
                        "num_decoder_layers": args.num_layers,
                    },
                    "metrics": train_metrics,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            print(f"✅ Saved checkpoint: {checkpoint_path} (exact_match: {best_exact_match:.4f})")

    print(f"\n🎉 Training complete!")
    print(f"Best exact match: {best_exact_match:.4f}")

    if best_exact_match >= 0.995:
        print("✅ Target ≥99.5% exact reconstruction achieved!")
    else:
        print(f"⚠️  Target not reached (current: {best_exact_match*100:.2f}%)")


if __name__ == "__main__":
    main()
