"""
Port Hugging Face GPT-2 (124M) weights into custom CoreCommandLLM architecture.
==============================================================================
This script:
1. Downloads pre-trained GPT-2 (124M) weights from Hugging Face Hub.
2. Expands the token embedding table from 50,257 to 50,263 to accommodate
   CommandLLM's 6 custom special conditioning tokens (<|start|>, <|os|>,
   <|prompt|>, <|cmd|>, <|end|>, <|pad|>), initialized via N(0, 0.02).
3. Slices the learned positional embedding table from 1024 down to 256.
4. Transposes Conv1D 1D weights into standard PyTorch nn.Linear matrices.
5. Preserves weight tying between transformer.wte and lm_head.
6. Serializes the mapped state dict to checkpoints/custom_base_checkpoint.pt.
7. Validates parameter compatibility with a clean reload and test forward pass.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from core.config import CommandLMConfig
from core.model import CoreCommandLLM


def port_gpt2_weights(
    checkpoint_dir: str = "checkpoints",
    checkpoint_filename: str = "custom_base_checkpoint.pt",
    model_name: str = "gpt2",
) -> str:
    """Port weights from Hugging Face GPT-2 into custom CoreCommandLLM."""
    print("=" * 72)
    print(" CommandLLM Weight Porting Pipeline (HF GPT-2 -> CoreCommandLLM)")
    print("=" * 72)

    # 1. Instantiate target CoreCommandLLM
    config = CommandLMConfig(
        vocab_size=50263,
        block_size=256,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        bias=True,
    )
    print(f"Instantiating custom CoreCommandLLM with config:\n  {config}\n")
    custom_model = CoreCommandLLM(config)
    custom_sd = custom_model.state_dict()

    # 2. Download pre-trained Hugging Face GPT-2 model
    print(f"Fetching pre-trained '{model_name}' weights from Hugging Face...")
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    hf_sd = hf_model.state_dict()
    print(f"HF weights loaded successfully ({len(hf_sd)} keys in state_dict).\n")

    # 3. Construct mapped state dict for CoreCommandLLM
    mapped_sd = {}

    # (a) Token Embedding Table (wte)
    # HF shape: (50257, 768) -> Custom shape: (50263, 768)
    hf_wte = hf_sd["transformer.wte.weight"]
    custom_wte = torch.empty(config.vocab_size, config.n_embd, dtype=hf_wte.dtype)
    # Copy base 50,257 vocabulary tokens
    custom_wte[:50257, :] = hf_wte
    # Initialize remaining 6 special tokens with N(0, 0.02)
    torch.nn.init.normal_(custom_wte[50257:, :], mean=0.0, std=0.02)
    mapped_sd["transformer.wte.weight"] = custom_wte
    print(f"[wte] Ported 50,257 base tokens + initialized 6 special tokens (shape: {custom_wte.shape})")

    # (b) Learned Positional Embedding Table (wpe)
    # HF shape: (1024, 768) -> Custom shape: (256, 768)
    hf_wpe = hf_sd["transformer.wpe.weight"]
    mapped_sd["transformer.wpe.weight"] = hf_wpe[: config.block_size, :].clone()
    print(f"[wpe] Sliced context window from 1024 to {config.block_size} tokens (shape: {mapped_sd['transformer.wpe.weight'].shape})")

    # (c) Transformer Decoder Blocks (h.0 .. h.11)
    # Layers that require transposition from HF Conv1D (in_features, out_features)
    # to PyTorch nn.Linear (out_features, in_features)
    conv1d_transposed_suffixes = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    for i in range(config.n_layer):
        prefix = f"transformer.h.{i}."

        # ln_1 (LayerNorm 1)
        mapped_sd[f"{prefix}ln_1.weight"] = hf_sd[f"{prefix}ln_1.weight"].clone()
        mapped_sd[f"{prefix}ln_1.bias"] = hf_sd[f"{prefix}ln_1.bias"].clone()

        # Attention QKV projection (c_attn)
        mapped_sd[f"{prefix}attn.c_attn.weight"] = hf_sd[f"{prefix}attn.c_attn.weight"].t().clone()
        mapped_sd[f"{prefix}attn.c_attn.bias"] = hf_sd[f"{prefix}attn.c_attn.bias"].clone()

        # Attention output projection (c_proj)
        mapped_sd[f"{prefix}attn.c_proj.weight"] = hf_sd[f"{prefix}attn.c_proj.weight"].t().clone()
        mapped_sd[f"{prefix}attn.c_proj.bias"] = hf_sd[f"{prefix}attn.c_proj.bias"].clone()

        # Keep custom causal mask buffer
        mapped_sd[f"{prefix}attn.bias"] = custom_sd[f"{prefix}attn.bias"].clone()

        # ln_2 (LayerNorm 2)
        mapped_sd[f"{prefix}ln_2.weight"] = hf_sd[f"{prefix}ln_2.weight"].clone()
        mapped_sd[f"{prefix}ln_2.bias"] = hf_sd[f"{prefix}ln_2.bias"].clone()

        # MLP expansion projection (c_fc)
        mapped_sd[f"{prefix}mlp.c_fc.weight"] = hf_sd[f"{prefix}mlp.c_fc.weight"].t().clone()
        mapped_sd[f"{prefix}mlp.c_fc.bias"] = hf_sd[f"{prefix}mlp.c_fc.bias"].clone()

        # MLP contraction projection (c_proj)
        mapped_sd[f"{prefix}mlp.c_proj.weight"] = hf_sd[f"{prefix}mlp.c_proj.weight"].t().clone()
        mapped_sd[f"{prefix}mlp.c_proj.bias"] = hf_sd[f"{prefix}mlp.c_proj.bias"].clone()

    print(f"[blocks] Ported and transposed weights across all {config.n_layer} Transformer blocks.")

    # (d) Final LayerNorm (ln_f)
    mapped_sd["transformer.ln_f.weight"] = hf_sd["transformer.ln_f.weight"].clone()
    mapped_sd["transformer.ln_f.bias"] = hf_sd["transformer.ln_f.bias"].clone()
    print("[ln_f] Ported final LayerNorm parameters.")

    # (e) Language Model Head (lm_head) - Tied to wte
    mapped_sd["lm_head.weight"] = custom_wte
    print("[lm_head] Tied lm_head.weight directly to transformer.wte.weight.")

    # 4. Load ported weights into custom CoreCommandLLM with strict=True
    print("\nLoading mapped state dict into custom CoreCommandLLM (strict=True)...")
    incompatible = custom_model.load_state_dict(mapped_sd, strict=True)
    print(f"Load result: missing_keys={incompatible.missing_keys}, unexpected_keys={incompatible.unexpected_keys}")

    # Re-enforce weight tying in Python reference
    custom_model.transformer.wte.weight = custom_model.lm_head.weight
    assert custom_model.transformer.wte.weight is custom_model.lm_head.weight, "Weight tying failed!"

    # 5. Create checkpoints directory and save checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    print(f"\nSerializing ported model checkpoint to: {checkpoint_path}")
    torch.save(custom_model.state_dict(), checkpoint_path)
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"Checkpoint saved successfully! File size: {file_size_mb:.2f} MB ({os.path.getsize(checkpoint_path):,} bytes)")

    # 6. Verification: Reload checkpoint from disk into a fresh instance
    print("\n" + "-" * 72)
    print(" Running Checkpoint Reload & Forward Pass Verification...")
    print("-" * 72)

    fresh_model = CoreCommandLLM(config)
    loaded_sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    fresh_model.load_state_dict(loaded_sd, strict=True)
    fresh_model.transformer.wte.weight = fresh_model.lm_head.weight
    fresh_model.eval()
    print("[1/2] Checkpoint reloaded into fresh model with 100% parameter match.")

    # Run forward pass with dummy inputs and targets
    dummy_input = torch.randint(0, config.vocab_size, (2, 32), dtype=torch.long)
    dummy_targets = torch.randint(0, config.vocab_size, (2, 32), dtype=torch.long)
    dummy_targets[:, :16] = -100  # simulate prompt masking

    with torch.no_grad():
        logits, loss = fresh_model(dummy_input, targets=dummy_targets)

    print(f"[2/2] Forward pass successful:")
    print(f"      - Input shape  : {tuple(dummy_input.shape)}")
    print(f"      - Logits shape : {tuple(logits.shape)} (Expected: (2, 32, {config.vocab_size}))")
    print(f"      - Computed loss: {loss.item():.4f}")
    assert logits.shape == (2, 32, config.vocab_size), f"Shape mismatch: {logits.shape}"
    assert loss is not None and not torch.isnan(loss), "Loss computation returned NaN!"

    print("\n" + "=" * 72)
    print(" [SUCCESS] Weight porting and base checkpoint verification complete!")
    print("=" * 72)

    return checkpoint_path


if __name__ == "__main__":
    port_gpt2_weights()
