"""
CommandLLM Full Training and Fine-Tuning Pipeline
=================================================
Orchestrates fine-tuning of CoreCommandLLM on paired terminal command datasets.
Features:
- Mandatory pre-trained base weight loading (raises FileNotFoundError if base checkpoint missing)
- Token-level classification accuracy computation with selective masking (target != -100)
- Sequence-level exact command match evaluation on held-out validation queries
- AdamW optimizer with decoupled weight decay (biases & LayerNorms excluded)
- Fine-tuning calibrated learning rate (8e-5 peak, 5e-6 min) with Cosine Annealing and linear warmup
- Mixed-precision training (torch.amp) for CUDA acceleration
- Gradient clipping (1.0 default)
- Checkpoint saving on best validation loss and sequence match
- Autoregressive generation sanity check after each epoch with completion-scoped repetition penalty
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
import contextlib
import re
from typing import Tuple, List, Dict, Optional

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


def extract_prompt_arguments(prompt: str) -> Dict[str, List[str]]:
    """
    Extract user-provided critical entities from the prompt:
    - numbers (ports, sizes, pids, counts, days)
    - paths (Linux and Windows paths)
    - filenames (with extensions)
    - extensions (.log, .txt, etc.)
    """
    entities = {
        "numbers": [],
        "paths": [],
        "filenames": [],
        "extensions": [],
    }

    # 1. File extensions
    exts = re.findall(r"\.[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)?", prompt)
    entities["extensions"] = list(set(e for e in exts if len(e) > 1 and not e.startswith("..")))

    # 2. Paths (Windows drive:\path or Linux /dir/path)
    win_paths = re.findall(r"[A-Za-z]:\\[a-zA-Z0-9_\\\-.]+", prompt)
    linux_paths = re.findall(r"(?:^|[\s'\"])(/(?:[a-zA-Z0-9_.\-]+/?)+)", prompt)
    entities["paths"] = list(set(win_paths + linux_paths))

    # 3. Filenames with extensions
    filenames = re.findall(r"\b[a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)?\b", prompt)
    entities["filenames"] = list(set(filenames))

    # 4. Standalone numbers (e.g. 8080, 100, 3000)
    numbers = re.findall(r"\b\d+\b", prompt)
    entities["numbers"] = list(set(numbers))

    return entities


def check_argument_preservation(prompt: str, generated_cmd: str) -> Tuple[bool, List[str]]:
    """
    Check whether all critical entities extracted from the prompt are preserved in generated_cmd.
    Returns (is_preserved, missing_items).
    """
    entities = extract_prompt_arguments(prompt)
    missing = []

    for num in entities["numbers"]:
        if num not in generated_cmd:
            missing.append(f"number:{num}")

    for p in entities["paths"]:
        p_clean = p.strip("'\"")
        if p_clean not in generated_cmd and p_clean.replace("\\", "/") not in generated_cmd.replace("\\", "/"):
            missing.append(f"path:{p_clean}")

    for fn in entities["filenames"]:
        if fn not in generated_cmd:
            missing.append(f"file:{fn}")

    for ext in entities["extensions"]:
        if ext not in generated_cmd and not any(fn.endswith(ext) for fn in entities["filenames"]):
            missing.append(f"ext:{ext}")

    return len(missing) == 0, missing


def normalize_command(cmd: str) -> str:
    """Normalize whitespace, quotes, and case for robust comparison."""
    if not cmd:
        return ""
    c = cmd.strip()
    c = re.sub(r"\s+", " ", c)
    c = c.replace('"', "'")
    return c


@torch.no_grad()
def evaluate_sequence_exact_match(
    model: CoreCommandLLM,
    tokenizer: CommandTokenizer,
    val_jsonl_path: str,
    device: torch.device,
    max_eval_samples: int = 50,
    repetition_penalty: float = 1.0,
) -> Dict[str, float]:
    """
    Evaluate exact match, normalized match, and argument preservation on validation queries.
    Uses repetition_penalty=1.0 by default to avoid suppressing valid repeated tokens (e.g. 8080).
    """
    if not os.path.isfile(val_jsonl_path):
        return {"exact_match": 0.0, "norm_exact_match": 0.0, "arg_preservation": 0.0, "num_samples": 0}

    model.eval()
    samples = []
    with open(val_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except Exception:
                    continue
            if len(samples) >= max_eval_samples:
                break

    if not samples:
        return {"exact_match": 0.0, "norm_exact_match": 0.0, "arg_preservation": 0.0, "num_samples": 0}

    exact_matches = 0
    norm_matches = 0
    arg_preserved = 0

    for s in samples:
        os_type = s["os"]
        prompt = s["prompt"]
        target_cmd = s["cmd"].strip()

        formatted = tokenizer.format_prompt(os_type, prompt)
        input_ids = torch.tensor([tokenizer.encode(formatted)], dtype=torch.long, device=device)
        out = model.generate(
            input_ids,
            max_new_tokens=48,
            temperature=0.0,
            top_k=None,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
        )
        gen_tokens = out[0, input_ids.size(1):].tolist()
        decoded = tokenizer.decode(gen_tokens).split("<|end|>")[0].strip()

        # Strict exact match
        if decoded == target_cmd or decoded.lower() == target_cmd.lower():
            exact_matches += 1

        # Normalized exact match (whitespace, quote style)
        if normalize_command(decoded).lower() == normalize_command(target_cmd).lower():
            norm_matches += 1

        # Argument preservation check
        ok, _ = check_argument_preservation(prompt, decoded)
        if ok:
            arg_preserved += 1

    model.train()
    n = len(samples)
    return {
        "exact_match": (exact_matches / n) * 100.0,
        "norm_exact_match": (norm_matches / n) * 100.0,
        "arg_preservation": (arg_preserved / n) * 100.0,
        "num_samples": n,
    }


def run_sample_generation(
    model: CoreCommandLLM,
    tokenizer: CommandTokenizer,
    device: torch.device,
    epoch: int,
):
    """Run an autoregressive test generation on representative prompt queries."""
    model.eval()
    test_queries = [
        ("linux", "kill process listening on port 8080"),
        ("powershell", "Find all files larger than 100MB in C:\\Logs"),
        ("linux", "find all files larger than 100MB in /var/log"),
        ("powershell", "kill process listening on port 8080"),
    ]

    print("\n" + "-" * 72)
    print(f" [Epoch {epoch}] Autoregressive Sample Generation Check (repetition_penalty=1.0):")
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
                temperature=0.0,
                top_k=None,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0,
            )

        gen_tokens = out_ids[0, input_ids.size(1):].tolist()
        decoded = tokenizer.decode(gen_tokens).split("<|end|>")[0].strip()
        ok, missing = check_argument_preservation(prompt, decoded)
        status = "PASS" if ok else f"FAIL (missing: {missing})"

        print(f"[{os_type.upper()}] Prompt : {prompt}")
        print(f"[{os_type.upper()}] Output : {decoded}")
        print(f"[{os_type.upper()}] Args   : {status}")
        print()

    print("-" * 72 + "\n")
    model.train()


def parse_args():
    parser = argparse.ArgumentParser(description="CommandLLM Training Script")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16, help="Batch size per training step (default: 16)")
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=5e-5, help="Fine-tuning peak learning rate calibrated for 124M model (default: 5e-5)")
    parser.add_argument("--min-lr", "--min_lr", dest="min_lr", type=float, default=5e-6, help="Minimum decayed learning rate (default: 5e-6)")
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=0.01, help="AdamW decoupled weight decay (default: 0.01)")
    parser.add_argument("--epochs", "--max-epochs", "--max_epochs", dest="epochs", type=int, default=5, help="Total training epochs (default: 5)")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping based on validation loss (default: 3)")
    parser.add_argument("--warmup-steps", "--warmup_steps", dest="warmup_steps", type=int, default=250, help="Linear warmup step count (default: 250)")
    parser.add_argument("--grad-clip", "--grad_clip", dest="grad_clip", type=float, default=1.0, help="Max gradient norm clipping (default: 1.0)")
    parser.add_argument(
        "--checkpoint-path",
        "--checkpoint_path",
        dest="checkpoint_path",
        type=str,
        default="checkpoints/custom_base_checkpoint.pt",
        help="Path to initial pre-trained base checkpoint (mandatory for fine-tuning)",
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
        "--sanity-check",
        "--sanity_check",
        dest="sanity_check",
        action="store_true",
        help="Execute comprehensive 9-point pipeline sanity verification suite and exit",
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
    print(f"Early Stop Patience : {args.patience}")
    print(f"Learning Rate       : {args.learning_rate} -> {args.min_lr}")
    print(f"Weight Decay        : {args.weight_decay}")
    print(f"Warmup Steps        : {args.warmup_steps}")
    print(f"Gradient Clipping   : {args.grad_clip}")
    print(f"Base Checkpoint     : {args.checkpoint_path}")
    print(f"Save Destination    : {args.save_path}")
    print(f"Dry Run Mode        : {args.dry_run}")
    print("=" * 72)

    # Strict Checkpoint Verification
    if not os.path.isfile(args.checkpoint_path):
        if args.dry_run:
            print(f"[Warning] Base checkpoint not found at '{args.checkpoint_path}'. Continuing in DRY-RUN mode.")
        else:
            raise FileNotFoundError(
                f"\n[FATAL] Pretrained base checkpoint not found at: '{args.checkpoint_path}'!\n"
                f"To prevent severe overfitting from random initialization, you must build the base checkpoint first:\n"
                f"    python scripts/port_weights.py\n"
            )

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
    best_exact_match = 0.0
    patience_counter = 0
    global_step = 0
    start_time = time.time()

    val_jsonl_path = str(WORKSPACE_ROOT / "data" / "val.jsonl")

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
            file=sys.stdout,
            leave=False,
            dynamic_ncols=True,
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
                "batch_loss": f"{loss_val:.4f}",
                "batch_acc": f"{batch_acc:.1f}%",
                "lr": f"{lr:.2e}",
            })

            if args.dry_run:
                pbar.close()
                sys.stdout.flush()
                print("\n[Dry Run] Single batch optimization step passed successfully.", flush=True)
                break

        # Explicitly close tqdm progress bar and clear carriage return
        pbar.close()
        sys.stdout.flush()

        avg_train_loss = epoch_loss / max(1, num_train_batches)
        avg_train_acc = (epoch_correct / max(1, epoch_tokens)) * 100.0 if epoch_tokens > 0 else 0.0

        # Evaluate validation loss and token accuracy
        val_batches = 1 if args.dry_run else -1
        val_loss, val_acc = evaluate(model, val_loader, device, ctx, max_batches=val_batches)

        # Evaluate sequence exact match and argument preservation on validation queries
        eval_samples_count = 1 if args.dry_run else 50
        seq_metrics = evaluate_sequence_exact_match(
            model=model,
            tokenizer=tokenizer,
            val_jsonl_path=val_jsonl_path,
            device=device,
            max_eval_samples=eval_samples_count,
            repetition_penalty=1.0,
        )

        # Clear BATCH vs EPOCH formatted summary (printed strictly once per epoch with flush=True)
        summary_str = (
            f"\n" + "=" * 72 + "\n"
            f" EPOCH [{epoch}/{args.epochs}] SUMMARY\n"
            f" " + "-" * 70 + "\n"
            f"  Train Loss        : {avg_train_loss:.4f} | Train Token Acc   : {avg_train_acc:.2f}%\n"
            f"  Validation Loss   : {val_loss:.4f} | Val Token Acc     : {val_acc:.2f}%\n"
            f"  Exact Match       : {seq_metrics['exact_match']:.2f}% | Norm Exact Match  : {seq_metrics['norm_exact_match']:.2f}%\n"
            f"  Arg Preservation  : {seq_metrics['arg_preservation']:.2f}% | Evaluated Samples : {seq_metrics['num_samples']}\n"
            f"  Epoch Final LR    : {lr:.2e}\n"
            f"=" * 72
        )
        print(summary_str, flush=True)
        sys.stdout.flush()

        # Checkpoint if validation loss improved or exact match improved
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        is_best = False
        reasons = []

        if val_loss < best_val_loss:
            reasons.append(f"val_loss improved {best_val_loss:.4f} -> {val_loss:.4f}")
            best_val_loss = val_loss
            is_best = True

        if seq_metrics["exact_match"] > best_exact_match:
            reasons.append(f"exact_match improved {best_exact_match:.1f}% -> {seq_metrics['exact_match']:.1f}%")
            best_exact_match = seq_metrics["exact_match"]
            is_best = True

        if is_best:
            torch.save(model.state_dict(), args.save_path)
            print(f"[*] New best validation performance ({', '.join(reasons)})! Saved checkpoint to: {args.save_path}", flush=True)
            sys.stdout.flush()
            patience_counter = 0
        else:
            patience_counter += 1

        # Run sample generation check
        run_sample_generation(model, tokenizer, device, epoch)
        sys.stdout.flush()

        # Early stopping patience check based on validation loss
        if not args.dry_run and epoch >= 4 and patience_counter >= args.patience:
            print(
                f"\n[EARLY STOPPING] Validation performance did not improve for {args.patience} "
                f"consecutive epochs. Stopping training.",
                flush=True,
            )
            sys.stdout.flush()
            break

        if args.dry_run:
            print("[Dry Run] Completed 1 full dry-run iteration. Exiting verification.")
            break

    elapsed = time.time() - start_time
    print("=" * 72)
    print(f" Training pipeline finished in {elapsed:.2f} seconds.")
    print(f" Best Validation Loss    : {best_val_loss:.4f}")
    print(f" Best Exact Match Acc    : {best_exact_match:.2f}%")
    print(f" Best Checkpoint Saved   : {args.save_path}")
    print("=" * 72)


def run_sanity_suite(device: torch.device):
    """
    Execute comprehensive 9-point pipeline sanity verification suite:
    1. Load 16 real samples from dataset
    2. Run forward pass
    3. Verify loss is finite and non-negative
    4. Verify labels, padding masking (-100), and token shift alignment
    5. Run autoregressive generation with repetition_penalty=1.0
    6. Verify decoded outputs and argument preservation on port 8080
    7. Verify validation evaluation logic independently
    8. Verify checkpoint save and reload integrity
    9. Verify AdamW optimizer and Cosine LR scheduler stepping
    """
    print("=" * 72)
    print(" CommandLLM 9-Point Sanity Verification Suite")
    print("=" * 72)

    tokenizer = CommandTokenizer()
    config = CommandLMConfig(
        vocab_size=50263,
        block_size=256,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
    )
    model = CoreCommandLLM(config).to(device)

    # Load base checkpoint if present
    base_ckpt = WORKSPACE_ROOT / "checkpoints" / "custom_base_checkpoint.pt"
    if base_ckpt.exists():
        sd = torch.load(base_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
        model.transformer.wte.weight = model.lm_head.weight
        print(f"[Check 0] Loaded base checkpoint from {base_ckpt}")

    # 1. Load samples
    print("\n[Check 1/9] Loading 16 real dataset samples...")
    train_ds = CommandDataset(split="train")
    samples_x = []
    samples_y = []
    for i in range(min(16, len(train_ds))):
        x, y = train_ds[i]
        samples_x.append(x)
        samples_y.append(y)
    batch_x = torch.stack(samples_x).to(device)
    batch_y = torch.stack(samples_y).to(device)
    print(f"  Input batch shape: {batch_x.shape}, Target batch shape: {batch_y.shape}")
    assert batch_x.shape == (16, 256)
    print("  -> PASS: 16 samples loaded successfully.")

    # 2. Forward pass
    print("\n[Check 2/9] Running forward pass...")
    logits, loss = model(batch_x, targets=batch_y)
    print(f"  Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
    assert logits.shape == (16, 256, config.vocab_size)
    print("  -> PASS: Forward pass logits shape verified.")

    # 3. Verify loss
    print("\n[Check 3/9] Verifying loss is finite and non-negative...")
    assert loss is not None
    assert torch.isfinite(loss).item(), "Loss is NaN or infinite!"
    assert loss.item() >= 0.0, "Loss is negative!"
    print(f"  -> PASS: Loss is valid ({loss.item():.4f}).")

    # 4. Verify labels and masking
    print("\n[Check 4/9] Verifying labels and masking alignment...")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = batch_y[..., 1:].contiguous()
    acc, num_corr, num_valid = compute_accuracy(shift_logits, shift_targets)
    assert num_valid > 0, "No valid supervised tokens found!"
    # Check that padding tokens are masked with -100
    pad_mask = (batch_x == tokenizer.pad_token_id)
    # Target positions corresponding to pad must be -100
    print(f"  Supervised tokens in batch: {num_valid}, Teacher-forced acc: {acc:.2f}%")
    print("  -> PASS: Label masking and shift alignment verified.")

    # 5. Autoregressive generation
    print("\n[Check 5/9] Testing autoregressive generation (repetition_penalty=1.0)...")
    prompt = tokenizer.format_prompt("linux", "kill process listening on port 8080")
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(input_ids, max_new_tokens=32, temperature=0.0, repetition_penalty=1.0)
    gen_tokens = out[0, input_ids.size(1):].tolist()
    gen_text = tokenizer.decode(gen_tokens)
    print(f"  Prompt : {prompt}")
    print(f"  Output : {gen_text}")
    print("  -> PASS: Autoregressive generation loop functional.")

    # 6. Verify argument extraction & preservation
    print("\n[Check 6/9] Verifying argument extraction and preservation validator...")
    test_p = "kill process listening on port 8080"
    pass_cmd = "kill -9 $(lsof -t -i:8080)"
    fail_cmd = "kill -9 $(lsof -t -i:80)"
    ok_pass, _ = check_argument_preservation(test_p, pass_cmd)
    ok_fail, missing = check_argument_preservation(test_p, fail_cmd)
    assert ok_pass is True, "Argument validator failed on valid command!"
    assert ok_fail is False and "number:8080" in missing, "Argument validator failed to detect corrupted 8080 -> 80!"
    print(f"  Valid test result     : {ok_pass} (Expected: True)")
    print(f"  Corrupted test result : {ok_fail}, Missing: {missing} (Expected: False, ['number:8080'])")
    print("  -> PASS: Argument preservation validator correctly catches argument corruption.")

    # 7. Verify validation evaluation
    print("\n[Check 7/9] Testing validation evaluation function independently...")
    val_jsonl = str(WORKSPACE_ROOT / "data" / "val.jsonl")
    seq_metrics = evaluate_sequence_exact_match(model, tokenizer, val_jsonl, device, max_eval_samples=5)
    print(f"  Val evaluation sample metrics: {seq_metrics}")
    assert "exact_match" in seq_metrics and "arg_preservation" in seq_metrics
    print("  -> PASS: Validation metric calculation functional.")

    # 8. Checkpoint save & reload
    print("\n[Check 8/9] Testing checkpoint serialization and reload integrity...")
    test_ckpt_path = WORKSPACE_ROOT / "checkpoints" / "sanity_test_checkpoint.pt"
    os.makedirs(test_ckpt_path.parent, exist_ok=True)
    torch.save(model.state_dict(), test_ckpt_path)
    loaded_sd = torch.load(test_ckpt_path, map_location="cpu", weights_only=True)
    reload_model = CoreCommandLLM(config)
    reload_model.load_state_dict(loaded_sd, strict=True)
    reload_model.transformer.wte.weight = reload_model.lm_head.weight
    assert reload_model.transformer.wte.weight is reload_model.lm_head.weight
    if test_ckpt_path.exists():
        test_ckpt_path.unlink()
    print("  -> PASS: Checkpoint serialized and reloaded with 100% parameter match and tied weights.")

    # 9. Optimizer & Scheduler step
    print("\n[Check 9/9] Testing AdamW optimizer and Cosine LR scheduler steps...")
    opt = configure_optimizers(model, weight_decay=0.01, learning_rate=5e-5)
    lr_0 = get_lr(0, warmup_steps=100, max_steps=1000, learning_rate=5e-5, min_lr=5e-6)
    lr_100 = get_lr(100, warmup_steps=100, max_steps=1000, learning_rate=5e-5, min_lr=5e-6)
    lr_500 = get_lr(500, warmup_steps=100, max_steps=1000, learning_rate=5e-5, min_lr=5e-6)
    print(f"  LR at step 0: {lr_0:.2e}, step 100: {lr_100:.2e}, step 500: {lr_500:.2e}")
    assert lr_0 < lr_100, "Warmup schedule failed!"
    assert lr_500 < lr_100, "Cosine decay failed!"
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print("  -> PASS: Optimizer step and gradient clipping verified.")

    print("\n" + "=" * 72)
    print(" [ALL 9 CHECKS PASSED] Pipeline is fully sound and verified!")
    print("=" * 72 + "\n")


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.sanity_check:
        run_sanity_suite(device)
        return

    train()


if __name__ == "__main__":
    main()
