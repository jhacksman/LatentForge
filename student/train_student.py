#!/usr/bin/env python3
"""
Train student transformer with knowledge distillation from teacher.
"""
import os
import sys
import argparse
import csv
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent))

# Optional DeepSpeed import
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

from student.student_model import StudentTransformer
from ae.ae_model import LatentAutoencoder
from ae.tokenizer_adapter import TokenizerAdapter
from kd.kd_client import KDClient


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
    kd_client: KDClient,
    tokenizer: TokenizerAdapter,
    kd_weight: float = 1.0,
    mse_weight: float = 1.0,
    ce_weight: float = 1.0,
    use_kd: bool = True,
    gradient_accumulation_steps: int = 1,
    use_activation_checkpointing: bool = False,
    use_deepspeed: bool = False,
) -> dict:
    """Train for one epoch."""
    # Check if student is DeepSpeed engine
    is_deepspeed_engine = hasattr(student, "backward") and hasattr(student, "step")

    if is_deepspeed_engine:
        student.train()
    else:
        student.train()
        autoencoder.eval()  # Freeze autoencoder

        # Enable activation checkpointing if requested
        if use_activation_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
            student.gradient_checkpointing_enable()

    total_loss = 0.0
    total_metrics = {
        "ce_loss": 0.0,
        "mse_loss": 0.0,
        "kd_loss": 0.0,
    }
    num_batches = 0
    accumulation_counter = 0

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
        # Zero gradients at the start of accumulation cycle (not needed for DeepSpeed)
        if not is_deepspeed_engine and accumulation_counter % gradient_accumulation_steps == 0:
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
        # CRITICAL: Use teacher tokenizer for proper alignment
        kd_loss = torch.tensor(0.0, device=device)
        if use_kd and kd_weight > 0 and batch_idx % 10 == 0:  # Only every 10 batches
            # Sample a few examples for KD
            num_kd_samples = min(2, batch_size)
            kd_indices = torch.randperm(batch_size)[:num_kd_samples]

            kd_losses = []
            for kd_idx in kd_indices:
                # Get predicted logits for this sample
                predicted_logits_chunk = decoded_logits[kd_idx, :, :]  # (k, vocab_size)

                # Build prompt from previous context
                context_chunk = chunks[kd_idx, 0, :].cpu()
                prompt_text = tokenizer.decode(context_chunk, skip_special_tokens=True)

                try:
                    # Get teacher distribution (teacher generates with its own tokenizer)
                    teacher_output = kd_client.get_teacher_distribution(
                        prompt=prompt_text,
                        max_tokens=k,
                        temperature=1.0,
                        top_logprobs=20,
                    )

                    # NEW APPROACH: Teacher returns tokens in ITS tokenization
                    # We need to compute KL in the teacher's token space, not student's
                    if len(teacher_output.tokens) > 0 and len(teacher_output.normalized_probs) > 0:
                        position_kls = []

                        # For each position in teacher's output
                        for pos in range(min(k, len(teacher_output.tokens))):
                            if pos >= len(teacher_output.normalized_probs):
                                break

                            teacher_probs_dict = teacher_output.normalized_probs[pos]
                            if not teacher_probs_dict:
                                continue

                            # Teacher token at this position (ground truth from teacher)
                            teacher_token_str = teacher_output.tokens[pos]

                            # Convert teacher token string to ID (using TEACHER tokenizer)
                            # Since tokenizer now uses teacher's tokenizer, this should work
                            teacher_token_ids = tokenizer.encode(teacher_token_str, add_special_tokens=False)
                            if len(teacher_token_ids) == 0:
                                continue
                            teacher_token_id = teacher_token_ids[0]

                            # Get student's probability for this specific teacher token
                            # predicted_logits_chunk[pos] is student's distribution over vocab
                            student_probs = F.softmax(predicted_logits_chunk[pos, :], dim=-1)

                            if teacher_token_id >= student_probs.shape[0]:
                                continue  # Token ID out of range

                            student_prob_for_teacher_token = student_probs[teacher_token_id]

                            # Build teacher distribution as sparse tensor
                            teacher_dist = torch.zeros(student_probs.shape[0], device=device)
                            for tok_str, prob in teacher_probs_dict.items():
                                tok_ids = tokenizer.encode(tok_str, add_special_tokens=False)
                                if len(tok_ids) > 0:
                                    tok_id = tok_ids[0]
                                    if tok_id < teacher_dist.shape[0]:
                                        teacher_dist[tok_id] = prob

                            # Normalize teacher distribution
                            teacher_dist_sum = teacher_dist.sum()
                            if teacher_dist_sum > 0:
                                teacher_dist = teacher_dist / teacher_dist_sum

                                # KL divergence: sum(teacher * log(teacher / student))
                                student_log_probs = F.log_softmax(predicted_logits_chunk[pos, :], dim=-1)
                                teacher_nonzero = teacher_dist > 0
                                if teacher_nonzero.any():
                                    kl_pos = (
                                        teacher_dist[teacher_nonzero] *
                                        (torch.log(teacher_dist[teacher_nonzero]) - student_log_probs[teacher_nonzero])
                                    ).sum()
                                    position_kls.append(kl_pos)

                        # Average KL over positions in this chunk
                        if position_kls:
                            chunk_kl = torch.stack(position_kls).mean()
                            kd_losses.append(chunk_kl)

                except Exception as e:
                    print(f"\nKD request failed: {e}")

            # Average KL over samples
            if kd_losses:
                kd_loss = torch.stack(kd_losses).mean()

        # Total loss
        total_loss_batch = (
            ce_weight * ce_loss + mse_weight * mse_loss + kd_weight * kd_loss
        )

        # Backward and optimizer step
        if is_deepspeed_engine:
            # DeepSpeed handles gradient accumulation internally
            student.backward(total_loss_batch)
            student.step()
        else:
            # Manual gradient accumulation
            scaled_loss = total_loss_batch / gradient_accumulation_steps
            scaled_loss.backward()

            # Step optimizer every gradient_accumulation_steps
            accumulation_counter += 1
            if accumulation_counter % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

        # Accumulate metrics (use original unscaled loss for reporting)
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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--use_activation_checkpointing", action="store_true", help="Enable activation checkpointing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint dir")
    parser.add_argument("--config", type=str, help="Config file (YAML)")
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed training")
    parser.add_argument("--deepspeed_config", type=str, default="configs/deepspeed_gb10.json", help="DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)

    # Set deterministic seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Set random seed: {args.seed}")

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
        kd_client = KDClient()
        print(f"Using teacher backend: {kd_client.backend_type}")

    # Data
    print("Loading data...")
    train_dataset = LatentDataset(args.data, args.k, args.seq_len)

    # DeepSpeed initialization
    use_deepspeed = args.deepspeed and DEEPSPEED_AVAILABLE
    if args.deepspeed and not DEEPSPEED_AVAILABLE:
        print("WARNING: DeepSpeed requested but not available. Install with: pip install deepspeed")
        print("Falling back to standard training...")
        use_deepspeed = False

    if use_deepspeed:
        print(f"Initializing DeepSpeed with config: {args.deepspeed_config}")

        # DeepSpeed handles dataloader internally
        model_engine, optimizer, train_loader, _ = deepspeed.initialize(
            args=args,
            model=student,
            model_parameters=student.parameters(),
            training_data=train_dataset,
            config=args.deepspeed_config,
        )
        print(f"DeepSpeed initialized (ZeRO stage: {model_engine.zero_optimization_stage()})")
        student = model_engine  # Replace student with engine
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    if not use_deepspeed:
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Activation checkpointing: {args.use_activation_checkpointing}")
    best_loss = float("inf")

    # CSV logging
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "student_metrics.csv"

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=["epoch", "loss", "ce_loss", "mse_loss", "kd_loss"],
    )
    csv_writer.writeheader()
    print(f"Logging metrics to {csv_path}")

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
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_activation_checkpointing=args.use_activation_checkpointing,
            use_deepspeed=use_deepspeed,
        )

        print(f"\nTrain metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")

        # GB10 memory tracking
        if torch.cuda.is_available():
            peak_allocated = torch.cuda.max_memory_allocated() / 1e9  # GB
            peak_reserved = torch.cuda.max_memory_reserved() / 1e9  # GB
            current_allocated = torch.cuda.memory_allocated() / 1e9  # GB
            print(f"\nGPU Memory (GB10):")
            print(f"  Peak allocated: {peak_allocated:.2f} GB")
            print(f"  Peak reserved: {peak_reserved:.2f} GB")
            print(f"  Current allocated: {current_allocated:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        # Log to CSV
        csv_writer.writerow({
            "epoch": epoch + 1,
            **train_metrics,
        })
        csv_file.flush()

        # Save checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        if train_metrics["loss"] < best_loss:
            best_loss = train_metrics["loss"]

            if use_deepspeed:
                # DeepSpeed checkpoint saving
                checkpoint_path = Path(args.checkpoint_dir) / f"student_epoch{epoch}"
                student.save_checkpoint(args.checkpoint_dir, f"student_epoch{epoch}")
                print(f"✅ Saved DeepSpeed checkpoint: {checkpoint_path}")
            else:
                # Standard PyTorch checkpoint
                checkpoint_path = Path(args.checkpoint_dir) / "student.pt"
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

    csv_file.close()

    # Print KD cache statistics
    if kd_client:
        kd_client.print_cache_stats()

    print(f"\n🎉 Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
