#!/usr/bin/env python3
"""
Train student transformer with knowledge distillation from teacher.
"""
import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent))

from student.student_model import StudentTransformer
from ae.ae_model import LatentAutoencoder
from ae.tokenizer_adapter import TokenizerAdapter
from kd.kd_client import VeniceKDClient


class LatentDataset(Dataset):
    """Dataset of token sequences for latent training."""

    def __init__(
        self,
        data_path: str,
        k: int,
        seq_len: int = 128,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to packed data directory
            k: Chunk size
            seq_len: Sequence length in latents (num_chunks)
        """
        self.k = k
        self.seq_len = seq_len

        # Load sequences
        data_file = Path(data_path) / "sequences.pt"
        if data_file.exists():
            self.sequences = torch.load(data_file)
        else:
            raise FileNotFoundError(f"No data found at {data_file}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Split into chunks of k tokens
        num_chunks = len(seq) // self.k
        if num_chunks < self.seq_len:
            # Pad
            pad_len = self.seq_len * self.k - len(seq)
            seq = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
            num_chunks = self.seq_len

        # Random start
        if num_chunks > self.seq_len:
            start_chunk = torch.randint(0, num_chunks - self.seq_len + 1, (1,)).item()
            start_idx = start_chunk * self.k
            seq = seq[start_idx : start_idx + self.seq_len * self.k]

        # Reshape to (seq_len, k)
        chunks = seq.reshape(self.seq_len, self.k)

        return chunks


def compute_kl_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute KL divergence loss.

    Args:
        student_logits: Student logits (batch_size, seq_len, vocab_size)
        teacher_probs: Teacher probabilities (batch_size, seq_len, vocab_size)
        temperature: Temperature for distillation

    Returns:
        KL loss
    """
    # Apply temperature to student logits
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL divergence
    kl_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
        log_target=False,
    )

    return kl_loss


def train_epoch(
    student: nn.Module,
    autoencoder: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kd_client: VeniceKDClient,
    tokenizer: TokenizerAdapter,
    kd_weight: float = 1.0,
    mse_weight: float = 1.0,
    ce_weight: float = 1.0,
    use_kd: bool = True,
) -> dict:
    """Train for one epoch."""
    student.train()
    autoencoder.eval()  # Freeze autoencoder

    total_loss = 0.0
    total_metrics = {
        "ce_loss": 0.0,
        "mse_loss": 0.0,
        "kd_loss": 0.0,
    }
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, chunks in enumerate(pbar):
        # chunks: (batch_size, seq_len, k)
        batch_size, seq_len, k = chunks.shape
        chunks = chunks.to(device)

        # Encode all chunks to latents using AE
        with torch.no_grad():
            # Flatten batch and seq_len
            flat_chunks = chunks.reshape(batch_size * seq_len, k)
            latents = autoencoder.encode(flat_chunks)  # (batch_size * seq_len, latent_dim)
            latents = latents.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, latent_dim)

        # Student forward: predict next latent
        optimizer.zero_grad()

        # Input: latents[:-1], Target: latents[1:]
        input_latents = latents[:, :-1, :]
        target_latents = latents[:, 1:, :]

        predicted_latents = student(input_latents)  # (batch_size, seq_len-1, latent_dim)

        # Loss 1: MSE in latent space
        mse_loss = F.mse_loss(predicted_latents, target_latents)

        # Loss 2: CE on decoded tokens
        # Decode predicted latents
        flat_predicted = predicted_latents.reshape(-1, predicted_latents.shape[-1])
        decoded_logits = autoencoder.decode(flat_predicted)  # (batch_size * (seq_len-1), k, vocab_size)

        # Target tokens
        target_chunks = chunks[:, 1:, :]  # (batch_size, seq_len-1, k)
        flat_target_chunks = target_chunks.reshape(-1, k)

        # CE loss
        ce_loss = F.cross_entropy(
            decoded_logits.reshape(-1, decoded_logits.shape[-1]),
            flat_target_chunks.reshape(-1),
        )

        # Loss 3: KD from teacher (optional, sparse for efficiency)
        kd_loss = torch.tensor(0.0, device=device)
        if use_kd and kd_weight > 0 and batch_idx % 10 == 0:  # Only every 10 batches
            # Sample a few examples for KD
            num_kd_samples = min(2, batch_size)
            kd_indices = torch.randperm(batch_size)[:num_kd_samples]

            for kd_idx in kd_indices:
                # Get target chunk
                target_chunk = target_chunks[kd_idx, 0, :].cpu()  # First position only
                prompt_text = tokenizer.decode(target_chunk, skip_special_tokens=True)

                try:
                    # Get teacher distribution
                    teacher_output = kd_client.get_teacher_distribution(
                        prompt=prompt_text,
                        max_tokens=k,
                        temperature=1.0,
                    )

                    # Convert teacher probs to tensor
                    # This is simplified; in practice, align tokens
                    # For now, skip KD if we can't align
                    if len(teacher_output.normalized_probs) > 0:
                        # Placeholder: just add a small penalty
                        # Full implementation would align tokens and compute proper KL
                        pass

                except Exception as e:
                    print(f"\nKD request failed: {e}")

        # Total loss
        total_loss_batch = (
            ce_weight * ce_loss + mse_weight * mse_loss + kd_weight * kd_loss
        )

        # Backward
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        # Accumulate
        total_loss += total_loss_batch.item()
        total_metrics["ce_loss"] += ce_loss.item()
        total_metrics["mse_loss"] += mse_loss.item()
        total_metrics["kd_loss"] += kd_loss.item()
        num_batches += 1

        # Update progress
        pbar.set_postfix(
            {
                "loss": f"{total_loss_batch.item():.4f}",
                "ce": f"{ce_loss.item():.4f}",
                "mse": f"{mse_loss.item():.4f}",
            }
        )

    # Average
    avg_metrics = {
        "loss": total_loss / num_batches,
        **{k: v / num_batches for k, v in total_metrics.items()},
    }

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train student with KD")
    parser.add_argument("--data", type=str, required=True, help="Data directory")
    parser.add_argument("--ae_ckpt", type=str, required=True, help="AE checkpoint")
    parser.add_argument("--k", type=int, default=8, help="Chunk size")
    parser.add_argument("--latent_dim", type=int, default=1024, help="Latent dimension")
    parser.add_argument("--hidden_dim", type=int, default=2048, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length (in latents)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--kd_w", type=float, default=1.0, help="KD loss weight")
    parser.add_argument("--mse_w", type=float, default=1.0, help="MSE loss weight")
    parser.add_argument("--ce_w", type=float, default=1.0, help="CE loss weight")
    parser.add_argument("--use_kd", action="store_true", help="Enable KD")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint dir")
    parser.add_argument("--config", type=str, help="Config file (YAML)")
    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # BF16
    use_bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"Using dtype: {dtype}")

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = TokenizerAdapter()

    # Load AE
    print(f"Loading autoencoder from {args.ae_ckpt}...")
    ae_ckpt = torch.load(args.ae_ckpt, map_location=device)
    ae_config = ae_ckpt["config"]
    autoencoder = LatentAutoencoder(**ae_config)
    autoencoder.load_state_dict(ae_ckpt["model"])
    autoencoder = autoencoder.to(device).to(dtype)
    autoencoder.eval()
    print(f"AE loaded (k={autoencoder.k}, latent_dim={autoencoder.latent_dim})")

    # Create student
    print("Creating student model...")
    student = StudentTransformer(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_seq_len=args.seq_len,
    )
    student = student.to(device).to(dtype)

    num_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {num_params:,}")

    # KD client
    kd_client = None
    if args.use_kd:
        print("Initializing KD client...")
        kd_client = VeniceKDClient()

    # Data
    print("Loading data...")
    train_dataset = LatentDataset(args.data, args.k, args.seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        # Train
        train_metrics = train_epoch(
            student,
            autoencoder,
            train_loader,
            optimizer,
            device,
            kd_client,
            tokenizer,
            kd_weight=args.kd_w,
            mse_weight=args.mse_w,
            ce_weight=args.ce_w,
            use_kd=args.use_kd,
        )

        print(f"\nTrain metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")

        # Save checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_path = Path(args.checkpoint_dir) / "student.pt"

        if train_metrics["loss"] < best_loss:
            best_loss = train_metrics["loss"]
            torch.save(
                {
                    "model": student.state_dict(),
                    "config": {
                        "latent_dim": args.latent_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "max_seq_len": args.seq_len,
                    },
                    "metrics": train_metrics,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            print(f"✅ Saved checkpoint: {checkpoint_path}")

    print(f"\n🎉 Training complete!")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
