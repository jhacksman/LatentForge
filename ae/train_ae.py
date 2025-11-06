#!/usr/bin/env python3
"""
Training script for the autoencoder.

Usage:
    python ae/train_ae.py \\
        --data_path <path_to_text_data> \\
        --output_dir checkpoints/ae \\
        --K 8 \\
        --D 1024 \\
        --epochs 2 \\
        --batch_size 8 \\
        --gradient_accumulation_steps 4 \\
        --learning_rate 3e-4
"""
import argparse
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from ae_model import AutoEncoder
from tokenizer_adapter import prepare_dataset, compute_reconstruction_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Train autoencoder")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data (text file or dataset)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/ae", help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-1B", help="Tokenizer to use")

    # Model architecture
    parser.add_argument("--K", type=int, default=8, help="Patch size (tokens to compress)")
    parser.add_argument("--D", type=int, default=1024, help="Latent dimension")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--intermediate_size", type=int, default=1280, help="MLP intermediate dimension")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=2, help="Number of decoder layers")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate")
    parser.add_argument("--kl_weight", type=float, default=1e-3, help="KL divergence weight")
    parser.add_argument("--kl_clamp", type=float, default=0.5, help="Minimum KL divergence")

    # Training
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for full epochs)")
    parser.add_argument("--block_size", type=int, default=2048, help="Training block size")

    # Logging and checkpointing
    parser.add_argument("--eval_steps", type=int, default=500, help="Eval interval")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save interval")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging interval")
    parser.add_argument("--val_split", type=float, default=0.05, help="Validation split fraction")

    # System
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def load_data(args):
    """Load and prepare training data."""
    print(f"Loading data from {args.data_path}")

    # Load dataset
    if os.path.isfile(args.data_path):
        # Single text file
        dataset = load_dataset('text', data_files=args.data_path, split='train')
    else:
        # Directory or HuggingFace dataset
        try:
            dataset = load_dataset(args.data_path, split='train')
        except:
            raise ValueError(f"Could not load dataset from {args.data_path}")

    # Split into train/val
    if args.val_split > 0:
        split_dataset = dataset.train_test_split(test_size=args.val_split, seed=args.seed)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
    else:
        train_dataset = dataset
        val_dataset = None

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare datasets
    print("Preparing training data...")
    train_dataset = prepare_dataset(
        train_dataset,
        tokenizer,
        block_size=args.block_size,
        K=args.K,
        num_proc=args.num_workers
    )

    if val_dataset is not None:
        print("Preparing validation data...")
        val_dataset = prepare_dataset(
            val_dataset,
            tokenizer,
            block_size=args.block_size,
            K=args.K,
            num_proc=args.num_workers
        )

    return train_dataset, val_dataset, tokenizer


def create_model(args, vocab_size):
    """Create autoencoder model."""
    model = AutoEncoder(
        vocab_size=vocab_size,
        K=args.K,
        D=args.D,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
        kl_weight=args.kl_weight,
        kl_clamp=args.kl_clamp,
    )
    return model


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Create learning rate scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0  # Constant after warmup

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    """Main training function."""
    # Set random seed
    torch.manual_seed(args.seed)

    # Load data
    train_dataset, val_dataset, tokenizer = load_data(args)

    # Create model
    print("Creating model...")
    model = create_model(args, vocab_size=len(tokenizer))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        val_dataloader = None

    # Calculate total steps
    steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    if args.max_steps > 0:
        total_steps = args.max_steps
        num_epochs = (total_steps // steps_per_epoch) + 1
    else:
        total_steps = steps_per_epoch * args.epochs
        num_epochs = args.epochs

    # Create scheduler
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.bf16 and device == 'cuda' else None
    autocast_dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_bf16_supported() else torch.float16

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs ({total_steps} steps)")
    print(f"Device: {device}, Mixed precision: {args.bf16}")

    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    model.train()
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        progress_bar = tqdm(train_dataloader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)

            # Forward pass
            with torch.cuda.amp.autocast(dtype=autocast_dtype) if args.bf16 else torch.no_grad():
                outputs = model(input_ids)
                loss = outputs['loss'] / args.gradient_accumulation_steps

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    progress_bar.set_postfix({
                        'loss': outputs['loss'].item(),
                        'recon': outputs['recon_loss'].item(),
                        'kl': outputs['kl_loss'].item(),
                        'lr': scheduler.get_last_lr()[0],
                    })

                # Evaluation
                if val_dataloader is not None and global_step % args.eval_steps == 0:
                    print(f"\nEvaluating at step {global_step}...")
                    model.eval()
                    metrics = compute_reconstruction_accuracy(model, val_dataloader, device)
                    print(f"Exact match: {metrics['exact_match']:.4f}, Patch exact match: {metrics['patch_exact_match']:.4f}")
                    model.train()

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(checkpoint_path, "model.pt"))
                    print(f"Saved checkpoint to {checkpoint_path}")

                # Check max steps
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Final save
    print("\nTraining complete! Saving final model...")
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    # Save config
    config = {
        'vocab_size': len(tokenizer),
        'K': args.K,
        'D': args.D,
        'hidden_size': args.hidden_size,
        'intermediate_size': args.intermediate_size,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'dropout': args.dropout,
        'kl_weight': args.kl_weight,
        'kl_clamp': args.kl_clamp,
    }
    with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)

    # Final evaluation
    if val_dataloader is not None:
        print("\nFinal evaluation...")
        model.eval()
        metrics = compute_reconstruction_accuracy(model, val_dataloader, device)
        print(f"Final exact match: {metrics['exact_match']:.4f}")
        print(f"Final patch exact match: {metrics['patch_exact_match']:.4f}")

        if metrics['exact_match'] >= 0.995:
            print("✓ Target reconstruction accuracy (≥99.5%) achieved!")
        else:
            print(f"✗ Target not met. Need {0.995 - metrics['exact_match']:.4f} more accuracy.")

    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
