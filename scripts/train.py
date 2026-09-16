"""
CommandLLM Full Training and Fine-Tuning Pipeline
=================================================
Orchestrates fine-tuning of CoreCommandLLM on paired terminal command datasets.
Features:
- Token-level classification accuracy computation with selective masking (target != -100)
- AdamW optimizer with decoupled weight decay (biases & LayerNorms excluded)
- Cosine Annealing learning rate schedule with linear warmup
- Mixed-precision training (torch.amp) for CUDA acceleration
- Gradient clipping (1.0 default)
- Epoch validation loss and accuracy tracking with target threshold stopping (e.g. >= 98.0%)
- Checkpoint saving on best validation accuracy record
- Autoregressive generation sanity check after each epoch
- Configurable CLI arguments with dry-run support
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
import contextlib
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from core.config import CommandLMConfig
from core.model import CoreCommandLLM
from core.tokenizer import CommandTokenizer
from data.dataset import CommandDataset


def get_lr(it: int, warmup_steps: int, max_steps: int, learning_rate: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if it < warmup_steps:
        return learning_rate * (it + 1) / max(1, warmup_steps)
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / max(1, max_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def configure_optimizers(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: Tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    """
    Configure AdamW with decoupled weight decay.
    2D parameters (projections, linear layers) are decayed.
    1D parameters (LayerNorm weights/biases, embeddings) are NOT decayed.
    """
    decay_params = []
    no_decay_params = []
    seen = set()

    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim >= 2 and not any(k in pn for k in ["wte", "wpe", "ln"]):
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer


def compute_accuracy(shift_logits: torch.Tensor, shift_targets: torch.Tensor) -> Tuple[float, int, int]:
    """
    Compute token-level classification accuracy on supervised target tokens.

    Args:
        shift_logits: Unnormalized logits tensor of shape (B, T-1, vocab_size).
        shift_targets: Ground truth target token IDs of shape (B, T-1).
                      Tokens with value -100 are ignored (conditioning prompt tokens).

    Returns:
        tuple of (accuracy_percentage, num_correct_tokens, num_valid_tokens)
    """
    preds = shift_logits.argmax(dim=-1)
    mask = shift_targets != -100
    correct = (preds == shift_targets) & mask
    total = mask.sum().item()
    correct_count = correct.sum().item()
    if total == 0:
        return 0.0, 0, 0
    return (correct_count / total) * 100.0, correct_count, total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    ctx: contextlib.AbstractContextManager,
    max_batches: int = -1,
) -> Tuple[float, float]:
    """Compute average validation loss and token-level accuracy across batches."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for i, (x, y) in enumerate(val_loader):
        if max_batches > 0 and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with ctx:
            logits, loss = model(x, targets=y)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = y[..., 1:].contiguous()
            _, correct, total = compute_accuracy(shift_logits, shift_targets)

        total_loss += loss.item()
        total_correct += correct
        total_tokens += total
        num_batches += 1

    model.train()
    avg_loss = total_loss / max(1, num_batches)
    avg_acc = (total_correct / max(1, total_tokens)) * 100.0 if total_tokens > 0 else 0.0
    return avg_loss, avg_acc


def run_sample_generation(
    model: CoreCommandLLM,
    tokenizer: CommandTokenizer,
    device: torch.device,
    epoch: int,
):
    """Run an autoregressive test generation on sample prompt queries."""
    model.eval()
    test_queries = [
        ("linux", "List all running processes sorted by memory usage"),
        ("powershell", "Find all files larger than 100MB in C:\\Logs"),
    ]

    print("\n" + "-" * 72)
    print(f" [Epoch {epoch}] Autoregressive Sample Generation Check:")
    print("-" * 72)

    for os_type, prompt in test_queries:
        formatted = tokenizer.format_prompt(os_type, prompt)
        input_ids = torch.tensor(
            [tokenizer.encode(formatted)], dtype=torch.long, device=device
        )

        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=48,
                temperature=0.2,
                top_k=40,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(out_ids[0].tolist())
        cmd_marker = "<|cmd|>"
        end_marker = "<|end|>"
        if cmd_marker in decoded:
            cmd_part = decoded.split(cmd_marker)[-1].split(end_marker)[0].strip()
        else:
            cmd_part = decoded

        print(f"[{os_type.upper()}] Prompt : {prompt}")
        print(f"[{os_type.upper()}] Output : {cmd_part}")
        print()

    print("-" * 72 + "\n")
    model.train()


def parse_args():
    parser = argparse.ArgumentParser(description="CommandLLM Training Script")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16, help="Batch size per training step")
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", "--min_lr", dest="min_lr", type=float, default=3e-5, help="Minimum decayed learning rate")
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=0.01, help="AdamW decoupled weight decay")
    parser.add_argument("--epochs", "--max-epochs", "--max_epochs", dest="epochs", type=int, default=10, help="Total training epochs (default: 10)")
    parser.add_argument("--target-accuracy", "--target_accuracy", dest="target_accuracy", type=float, default=98.0, help="Target validation accuracy threshold (default: 98.0)")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping without val accuracy improvement (default: 3)")
    parser.add_argument("--warmup-steps", "--warmup_steps", dest="warmup_steps", type=int, default=100, help="Linear warmup step count")
    parser.add_argument("--grad-clip", "--grad_clip", dest="grad_clip", type=float, default=1.0, help="Max gradient norm clipping")
    parser.add_argument(
        "--checkpoint-path",
        "--checkpoint_path",
        dest="checkpoint_path",
        type=str,
        default="checkpoints/custom_base_checkpoint.pt",
        help="Path to initial pre-trained base checkpoint",
    )
    parser.add_argument(
        "--save-path",
        "--save_path",
        dest="save_path",
        type=str,
        default="checkpoints/terminal_model_final.pt",
        help="Destination path for best fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="Execute 1 batch dry-run on CPU/device to verify tensor integrity",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device(args.device)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    print("=" * 72)
    print(" CommandLLM Training & Fine-Tuning Pipeline")
    print("=" * 72)
    print(f"Device              : {device} ({torch.cuda.get_device_name(0) if device_type == 'cuda' else 'CPU'})")
    print(f"Batch Size          : {args.batch_size}")
    print(f"Epochs              : {args.epochs}")
    print(f"Target Accuracy     : {args.target_accuracy:.1f}%")
    print(f"Early Stop Patience : {args.patience}")
    print(f"Learning Rate       : {args.learning_rate} -> {args.min_lr}")
    print(f"Weight Decay        : {args.weight_decay}")
    print(f"Warmup Steps        : {args.warmup_steps}")
    print(f"Gradient Clipping   : {args.grad_clip}")
    print(f"Base Checkpoint     : {args.checkpoint_path}")
    print(f"Save Destination    : {args.save_path}")
    print(f"Dry Run Mode        : {args.dry_run}")
    print("=" * 72)

    # 1. Datasets and DataLoaders
    print("\n[1/5] Initializing memory-mapped datasets...")
    train_dataset = CommandDataset(split="train")
    val_dataset = CommandDataset(split="val")
    print(f"Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device_type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=(device_type == "cuda"),
    )

    # 2. Model Initialization & Base Weight Loading
    print("\n[2/5] Initializing CoreCommandLLM architecture...")
    config = CommandLMConfig(
        vocab_size=50263,
        block_size=256,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        bias=True,
    )
    model = CoreCommandLLM(config)

    if os.path.isfile(args.checkpoint_path):
        print(f"Loading base checkpoint from: {args.checkpoint_path}")
        checkpoint_sd = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint_sd, strict=True)
        model.transformer.wte.weight = model.lm_head.weight
        print("Base weights loaded and tied successfully.")
    else:
        print(f"[Warning] Checkpoint not found at {args.checkpoint_path}! Training with random initialization.")

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters on {device}: {total_params:,}")

    # 3. Optimizer & Precision Setup
    print("\n[3/5] Setting up AdamW optimizer and precision...")
    optimizer = configure_optimizers(model, args.weight_decay, args.learning_rate)

    # Precision context
    use_amp = (device_type == "cuda")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype) if use_amp else contextlib.nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    print(f"Mixed precision: enabled={use_amp}, dtype={amp_dtype if use_amp else 'float32'}")

    tokenizer = CommandTokenizer()

    # Calculate steps
    steps_per_epoch = len(train_loader)
    total_steps = (args.epochs * steps_per_epoch) if not args.dry_run else 1
    warmup_steps = min(args.warmup_steps, max(1, total_steps // 10)) if args.dry_run else args.warmup_steps

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    global_step = 0
    start_time = time.time()

    # 4. Training Loop
    print("\n[4/5] Starting training execution...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_tokens = 0
        num_train_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            total=(1 if args.dry_run else len(train_loader)),
        )

        for step, (x, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Update dynamic learning rate
            lr = get_lr(global_step, warmup_steps, total_steps, args.learning_rate, args.min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            with ctx:
                logits, loss = model(x, targets=y)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_targets = y[..., 1:].contiguous()
                batch_acc, b_correct, b_total = compute_accuracy(shift_logits, shift_targets)

            # Backward pass with scaler or standard backward
            if use_amp and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_correct += b_correct
            epoch_tokens += b_total
            num_train_batches += 1
            global_step += 1

            pbar.set_postfix({
                "train_loss": f"{loss_val:.4f}",
                "train_acc": f"{batch_acc:.2f}%",
                "lr": f"{lr:.2e}",
            })

            if args.dry_run:
                print("\n[Dry Run] Single batch optimization step passed successfully.")
                break

        avg_train_loss = epoch_loss / max(1, num_train_batches)
        avg_train_acc = (epoch_correct / max(1, epoch_tokens)) * 100.0 if epoch_tokens > 0 else 0.0

        # Evaluate validation loss and accuracy
        val_batches = 1 if args.dry_run else -1
        val_loss, val_acc = evaluate(model, val_loader, device, ctx, max_batches=val_batches)

        # Formatted epoch summary line
        print(
            f"\nEpoch [{epoch}/{args.epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        # Checkpoint if validation accuracy achieved a new best record
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"[*] New best validation accuracy: {val_acc:.2f}%! Saved checkpoint to: {args.save_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Run sample generation check
        run_sample_generation(model, tokenizer, device, epoch)

        # Target accuracy early stopping check (after at least 3 epochs to guarantee stability)
        if not args.dry_run and epoch >= 3 and val_acc >= args.target_accuracy:
            print(
                f"\n[TARGET REACHED] Validation accuracy {val_acc:.2f}% reached target threshold "
                f"{args.target_accuracy:.1f}%. Early stopping triggered."
            )
            break

        # Early stopping patience check
        if not args.dry_run and epoch >= 5 and patience_counter >= args.patience:
            print(
                f"\n[EARLY STOPPING] Validation accuracy did not improve for {args.patience} "
                f"consecutive epochs. Stopping training."
            )
            break

        if args.dry_run:
            print("[Dry Run] Completed 1 full dry-run iteration. Exiting verification.")
            break

    elapsed = time.time() - start_time
    print("=" * 72)
    print(f" Training pipeline finished in {elapsed:.2f} seconds.")
    print(f" Best Validation Loss: {best_val_loss:.4f}")
    print(f" Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f" Best Checkpoint Saved: {args.save_path}")
    print("=" * 72)


if __name__ == "__main__":
    train()
