#!/usr/bin/env python3
"""
Training script for the student model with knowledge distillation.

Usage:
    python student/train_student.py \\
        --data_path <path_to_text_data> \\
        --ae_path checkpoints/ae \\
        --teacher_model meta-llama/Llama-3.2-1B \\
        --output_dir checkpoints/student \\
        --K 8 \\
        --KD_W 1.0 \\
        --MSE_W 1.0 \\
        --CE_W 1.0
"""
import argparse
import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ae.ae_model import AutoEncoder
from ae.tokenizer_adapter import prepare_dataset
from student_model import StudentModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train student model with knowledge distillation")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--ae_path", type=str, required=True, help="Path to trained autoencoder")
    parser.add_argument("--teacher_model", type=str, required=True, help="Teacher model (HuggingFace ID)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/student", help="Output directory")

    # Model architecture
    parser.add_argument("--K", type=int, default=8, help="Patch size (must match AE)")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=2048, help="MLP intermediate dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    # Loss weights
    parser.add_argument("--KD_W", type=float, default=1.0, help="KL divergence to teacher weight")
    parser.add_argument("--MSE_W", type=float, default=1.0, help="Latent MSE weight")
    parser.add_argument("--CE_W", type=float, default=1.0, help="Token CE weight")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")

    # Training
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps")
    parser.add_argument("--block_size", type=int, default=2048, help="Training block size")

    # Logging and checkpointing
    parser.add_argument("--eval_steps", type=int, default=1000, help="Eval interval")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save interval")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging interval")
    parser.add_argument("--val_split", type=float, default=0.05, help="Validation split")

    # System
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def load_autoencoder(ae_path, device):
    """Load trained autoencoder."""
    print(f"Loading autoencoder from {ae_path}")
    config_path = os.path.join(ae_path, "config.json")
    model_path = os.path.join(ae_path, "model.pt")

    with open(config_path, 'r') as f:
        config = json.load(f)

    ae = AutoEncoder(**config)
    ae.load_state_dict(torch.load(model_path, map_location=device))
    ae = ae.to(device)
    ae.eval()

    # Freeze autoencoder
    for param in ae.parameters():
        param.requires_grad = False

    print(f"Loaded AE: K={config['K']}, D={config['D']}")
    return ae, config


def load_teacher(teacher_model, device, bf16):
    """Load teacher model."""
    print(f"Loading teacher model: {teacher_model}")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map=device,
    )
    teacher.eval()

    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False

    num_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher parameters: {num_params / 1e6:.2f}M")
    return teacher


def compute_distillation_loss(
    student_model,
    ae_model,
    teacher_model,
    input_ids,
    kd_w,
    mse_w,
    ce_w,
    temperature,
):
    """
    Compute combined distillation loss.

    Returns:
        total_loss, mse_loss, ce_loss, kd_loss
    """
    batch_size, seq_len = input_ids.shape
    K = ae_model.K

    # Encode input to latent sequence
    with torch.no_grad():
        mean, _ = ae_model.encode(input_ids)  # [batch, num_patches, D]
        num_patches = mean.shape[1]

    # Prepare input and target latents for student
    # Student predicts next latent, so input = latents[:-1], target = latents[1:]
    if num_patches <= 1:
        return None, None, None, None  # Not enough patches for prediction

    input_latents = mean[:, :-1, :]  # [batch, num_patches-1, D]
    target_latents = mean[:, 1:, :]  # [batch, num_patches-1, D]

    # Student forward pass
    predicted_latents = student_model(input_latents)  # [batch, num_patches-1, D]

    # Loss 1: Latent MSE
    mse_loss = F.mse_loss(predicted_latents, target_latents)

    # Loss 2: Token CE after decoding
    with torch.no_grad():
        target_token_logits = ae_model.decode(target_latents)  # [batch, (num_patches-1)*K, vocab]
    predicted_token_logits = ae_model.decode(predicted_latents)

    target_tokens = input_ids[:, K:]  # Skip first patch, those are the targets
    target_tokens = target_tokens[:, :(num_patches-1)*K]  # Trim to match

    ce_loss = F.cross_entropy(
        predicted_token_logits.reshape(-1, ae_model.vocab_size),
        target_tokens.reshape(-1),
        reduction='mean'
    )

    # Loss 3: KL divergence to teacher logits
    with torch.no_grad():
        # Get teacher logits for the same token positions
        teacher_outputs = teacher_model(input_ids)
        teacher_logits = teacher_outputs.logits[:, K-1:-1, :]  # Align with target tokens
        teacher_logits = teacher_logits[:, :(num_patches-1)*K, :]

    # KL divergence: KL(teacher || student)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(predicted_token_logits / temperature, dim=-1)
    kd_loss = F.kl_div(
        student_log_probs.reshape(-1, ae_model.vocab_size),
        teacher_probs.reshape(-1, ae_model.vocab_size),
        reduction='batchmean'
    ) * (temperature ** 2)

    # Combined loss
    total_loss = mse_w * mse_loss + ce_w * ce_loss + kd_w * kd_loss

    return total_loss, mse_loss, ce_loss, kd_loss


def train(args):
    """Main training function."""
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load autoencoder
    ae_model, ae_config = load_autoencoder(args.ae_path, device)
    assert ae_config['K'] == args.K, f"K mismatch: AE has {ae_config['K']}, args has {args.K}"
    latent_dim = ae_config['D']

    # Load teacher
    teacher_model = load_teacher(args.teacher_model, device, args.bf16)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.ae_path)

    # Load data
    print(f"Loading data from {args.data_path}")
    if os.path.isfile(args.data_path):
        dataset = load_dataset('text', data_files=args.data_path, split='train')
    else:
        dataset = load_dataset(args.data_path, split='train')

    if args.val_split > 0:
        split_dataset = dataset.train_test_split(test_size=args.val_split, seed=args.seed)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
    else:
        train_dataset = dataset
        val_dataset = None

    print("Preparing training data...")
    train_dataset = prepare_dataset(
        train_dataset,
        tokenizer,
        block_size=args.block_size,
        K=args.K,
        num_proc=args.num_workers
    )

    # Create student model
    print("Creating student model...")
    student = StudentModel(
        latent_dim=latent_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
    )
    student = student.to(device)

    num_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {num_params / 1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Calculate steps
    steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    if args.max_steps > 0:
        total_steps = args.max_steps
        num_epochs = (total_steps // steps_per_epoch) + 1
    else:
        total_steps = steps_per_epoch * args.epochs
        num_epochs = args.epochs

    # Scheduler
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.bf16 and device == 'cuda' else None
    autocast_dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_bf16_supported() else torch.float16

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs ({total_steps} steps)")
    print(f"Loss weights: KD={args.KD_W}, MSE={args.MSE_W}, CE={args.CE_W}")

    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    student.train()
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        progress_bar = tqdm(train_dataloader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)

            # Forward pass
            with torch.cuda.amp.autocast(dtype=autocast_dtype) if args.bf16 else torch.no_grad():
                loss, mse_loss, ce_loss, kd_loss = compute_distillation_loss(
                    student,
                    ae_model,
                    teacher_model,
                    input_ids,
                    args.KD_W,
                    args.MSE_W,
                    args.CE_W,
                    args.temperature,
                )

            if loss is None:
                continue

            loss = loss / args.gradient_accumulation_steps

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    progress_bar.set_postfix({
                        'loss': loss.item() * args.gradient_accumulation_steps,
                        'mse': mse_loss.item() if mse_loss is not None else 0,
                        'ce': ce_loss.item() if ce_loss is not None else 0,
                        'kd': kd_loss.item() if kd_loss is not None else 0,
                        'lr': scheduler.get_last_lr()[0],
                    })

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    torch.save(student.state_dict(), os.path.join(checkpoint_path, "model.pt"))
                    print(f"\nSaved checkpoint to {checkpoint_path}")

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Final save
    print("\nTraining complete! Saving final model...")
    torch.save(student.state_dict(), os.path.join(args.output_dir, "model.pt"))

    config = {
        'latent_dim': latent_dim,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'intermediate_size': args.intermediate_size,
        'dropout': args.dropout,
    }
    with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
