#!/usr/bin/env python3
"""
CommandLLM: Tokenization & Selective Loss Masking Pipeline
==========================================================
Tokenizes dual-OS terminal command datasets using Byte-Pair Encoding (BPE):
- Custom special tokens:
    <|start|>, <|os|>, <|prompt|>, <|cmd|>, <|end|>, <|pad|>
- Record format strictly:
    <|start|><|os|>{os}<|prompt|>{prompt}<|cmd|>{cmd}<|end|>
- Selective Loss Masking:
    * input_ids: full token sequence padded/truncated to max_seq_len
    * targets: -100 for all conditioning tokens up to and including <|cmd|> and <|pad|>
    * targets: true token IDs for {cmd} and <|end|>
- Exports processed binary arrays (.npy) for high-performance memory mapping.
- Includes automated verification sanity check decoding a batch to console.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tokenize_dataset")

# ==============================================================================
# 1. Custom Special Tokens Specification
# ==============================================================================

SPECIAL_TOKENS: Dict[str, int] = {
    "<|start|>": 50257,
    "<|os|>": 50258,
    "<|prompt|>": 50259,
    "<|cmd|>": 50260,
    "<|end|>": 50261,
    "<|pad|>": 50262,
}

IGNORE_INDEX = -100  # PyTorch standard CrossEntropyLoss ignore_index


def get_command_tokenizer():
    """Construct a GPT-2 BPE tokenizer extended with CommandLLM custom special tokens."""
    if tiktoken is None:
        raise ImportError("tiktoken is required for tokenization. Please install it with 'pip install tiktoken'.")

    base_enc = tiktoken.get_encoding("gpt2")
    custom_special_tokens = {**base_enc._special_tokens, **SPECIAL_TOKENS}

    enc = tiktoken.Encoding(
        name="command_bpe_gpt2",
        pat_str=base_enc._pat_str,
        mergeable_ranks=base_enc._mergeable_ranks,
        special_tokens=custom_special_tokens,
    )
    return enc


# ==============================================================================
# 2. Sequence Tokenization and Selective Loss Masking
# ==============================================================================

def process_single_record(
    record: Dict[str, str],
    tokenizer,
    max_seq_len: int = 256,
) -> Tuple[List[int], List[int]]:
    """
    Format and tokenize a single JSONL record into input_ids and selectively masked targets.

    Sequence format:
        <|start|><|os|>{os}<|prompt|>{prompt}<|cmd|>{cmd}<|end|>

    Selective Loss Masking:
        - Target is -100 from <|start|> up to and including <|cmd|>
        - Target retains true IDs for {cmd} and <|end|>
        - Target is -100 for all <|pad|> positions
    """
    os_label = record["os"].strip().lower()
    prompt_text = record["prompt"].strip()
    cmd_text = record["cmd"].strip()

    # Conditioning Prefix
    prefix_str = f"<|start|><|os|>{os_label}<|prompt|>{prompt_text}<|cmd|>"
    # Supervised Generation Target
    cmd_str = f"{cmd_text}<|end|>"

    prefix_ids = tokenizer.encode(prefix_str, allowed_special="all")
    cmd_ids = tokenizer.encode(cmd_str, allowed_special="all")

    full_ids = prefix_ids + cmd_ids
    # Prompt and OS tokens masked with -100, command + <|end|> retain real token IDs
    targets = [IGNORE_INDEX] * len(prefix_ids) + list(cmd_ids)

    # Pad or Truncate to max_seq_len
    pad_id = SPECIAL_TOKENS["<|pad|>"]
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        targets = targets[:max_seq_len]
    else:
        pad_len = max_seq_len - len(full_ids)
        full_ids.extend([pad_id] * pad_len)
        targets.extend([IGNORE_INDEX] * pad_len)

    return full_ids, targets


def process_dataset_file(
    file_path: str,
    tokenizer,
    max_seq_len: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process a JSONL file into numpy arrays of input_ids and targets."""
    input_ids_list: List[List[int]] = []
    targets_list: List[List[int]] = []

    file_p = Path(file_path)
    if not file_p.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_p, "r", encoding="utf-8") as f:
        lines = f.readlines()

    logger.info(f"Processing {len(lines):,} records from {file_p.name}...")
    for line in tqdm(lines, desc=f"Tokenizing {file_p.name}"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            inp, tgt = process_single_record(record, tokenizer, max_seq_len=max_seq_len)
            input_ids_list.append(inp)
            targets_list.append(tgt)
        except Exception as e:
            logger.warning(f"Skipping malformed line in {file_p.name}: {e}")

    # Use int64 or int32 (int32 is memory efficient and supports vocab size > 50,000)
    input_ids_arr = np.array(input_ids_list, dtype=np.int32)
    targets_arr = np.array(targets_list, dtype=np.int32)

    return input_ids_arr, targets_arr


# ==============================================================================
# 3. Sanity Check and Verification
# ==============================================================================

def run_sanity_check(
    input_ids_arr: np.ndarray,
    targets_arr: np.ndarray,
    tokenizer,
    num_samples: int = 3,
):
    """
    Verification sanity check:
    Decodes samples and prints:
      1. Full reconstructed input sequence
      2. Exactly which tokens have active loss (target != -100)
    """
    logger.info("=" * 70)
    logger.info(" TOKENIZATION & LOSS MASKING VERIFICATION SANITY CHECK")
    logger.info("=" * 70)

    n_samples = min(num_samples, len(input_ids_arr))
    pad_id = SPECIAL_TOKENS["<|pad|>"]

    for i in range(n_samples):
        inputs = input_ids_arr[i].tolist()
        targets = targets_arr[i].tolist()

        # Remove padding for readable inspection
        non_pad_inputs = [tok for tok in inputs if tok != pad_id]
        active_target_tokens = [tgt for tgt in targets if tgt != IGNORE_INDEX]

        decoded_full = tokenizer.decode(non_pad_inputs)
        decoded_targets = tokenizer.decode(active_target_tokens)

        # Verification asserts
        total_tokens = len(non_pad_inputs)
        active_loss_count = len(active_target_tokens)
        masked_tokens_count = total_tokens - active_loss_count

        print(f"\n--- [Sample {i + 1}] ---")
        print(f"Total non-pad tokens: {total_tokens} | Masked (Loss=None): {masked_tokens_count} | Supervised: {active_loss_count}")
        print(f"[Full Decoded Input]:\n{decoded_full}")
        print(f"[Tokens Supervised by Loss (target != -100)]:\n{decoded_targets}")
        print("-" * 70)

    logger.info("Sanity check completed successfully.")


# ==============================================================================
# 4. Pipeline Execution & File Export
# ==============================================================================

def tokenize_and_export(
    train_file: str = "data/train.jsonl",
    val_file: str = "data/val.jsonl",
    output_dir: str = "data/processed",
    max_seq_len: int = 256,
    num_sanity_samples: int = 3,
):
    """Run tokenization on train/val sets, save binary numpy arrays, and verify."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tokenizer = get_command_tokenizer()
    vocab_size = tokenizer.n_vocab
    logger.info(f"Initialized CommandTokenizer with vocab size: {vocab_size:,}")

    # Process Train Set
    train_inputs, train_targets = process_dataset_file(train_file, tokenizer, max_seq_len=max_seq_len)
    # Process Val Set
    val_inputs, val_targets = process_dataset_file(val_file, tokenizer, max_seq_len=max_seq_len)

    # File paths
    train_in_path = out_path / "train_inputs.npy"
    train_tgt_path = out_path / "train_targets.npy"
    val_in_path = out_path / "val_inputs.npy"
    val_tgt_path = out_path / "val_targets.npy"
    meta_path = out_path / "tokenizer_meta.json"

    # Export binary arrays
    logger.info("Exporting binary numpy arrays...")
    np.save(train_in_path, train_inputs)
    np.save(train_tgt_path, train_targets)
    np.save(val_in_path, val_inputs)
    np.save(val_tgt_path, val_targets)

    # Export metadata
    meta = {
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "ignore_index": IGNORE_INDEX,
        "special_tokens": SPECIAL_TOKENS,
        "train_samples": len(train_inputs),
        "val_samples": len(val_inputs),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("=" * 60)
    logger.info(" TOKENIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f" Train Inputs Shape  : {train_inputs.shape} -> {train_in_path}")
    logger.info(f" Train Targets Shape : {train_targets.shape} -> {train_tgt_path}")
    logger.info(f" Val Inputs Shape    : {val_inputs.shape} -> {val_in_path}")
    logger.info(f" Val Targets Shape   : {val_targets.shape} -> {val_tgt_path}")
    logger.info(f" Tokenizer Metadata  : {meta_path}")
    logger.info("=" * 60)

    # Run Sanity Check on Train Set
    run_sanity_check(train_inputs, train_targets, tokenizer, num_samples=num_sanity_samples)


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CommandLLM Tokenization Pipeline: BPE Tokenization with Selective Loss Masking."
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/train.jsonl",
        help="Path to input train.jsonl (default: 'data/train.jsonl').",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/val.jsonl",
        help="Path to input val.jsonl (default: 'data/val.jsonl').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Destination directory for processed .npy arrays (default: 'data/processed').",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Fixed maximum context length (default: 256).",
    )
    parser.add_argument(
        "--sanity-samples",
        type=int,
        default=3,
        help="Number of samples to decode during sanity check (default: 3).",
    )

    args = parser.parse_args()
    tokenize_and_export(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        num_sanity_samples=args.sanity_samples,
    )


if __name__ == "__main__":
    main()
