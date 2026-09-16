"""
Causal Multi-Head Self-Attention Module
=======================================
Implements masked causal self-attention for autoregressive language modeling.
Includes fused FlashAttention acceleration via F.scaled_dot_product_attention
alongside an explicit textbook scaled dot-product attention fallback.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.config import CommandLMConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention layer.

    Key Architectural Details:
    1. Combined QKV Projection:
       Projects Query, Key, and Value vectors simultaneously via a single
       nn.Linear(n_embd, 3 * n_embd). This executes as a single high-throughput
       GEMM kernel, avoiding the overhead of three separate matrix multiplications.

    2. Scaled Dot-Product Attention:
       Computes Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) + mask ) @ V
       The scaling factor 1 / sqrt(d_k) prevents dot-product magnitudes from
       growing exponentially with head dimension, which would push softmax
       into regions with vanishing gradients.

    3. Causal Masking:
       Enforces the autoregressive property: token at step t is prohibited
       from attending to tokens at future positions > t. Registered as a buffer
       so it is saved with the module state dict without being treated as a
       trainable parameter.

    4. Hardware Acceleration:
       Dispatches to PyTorch 2.0+ F.scaled_dot_product_attention when available,
       leveraging fused FlashAttention-2 kernels to achieve sub-quadratic memory
       and significant GPU/CPU speedup.
    """

    def __init__(self, config: CommandLMConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # 1. Fused QKV projection layer
        # Output shape is (B, T, 3 * n_embd) which will be sliced into Q, K, and V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # 2. Output projection layer: projects multi-head concatenated output back to residual dimension
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # 3. Regularization dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 4. Causal mask buffer: lower triangular matrix of ones
        # Shape: (1, 1, block_size, block_size) for broadcasting across (B, n_head, T, T)
        # register_buffer ensures it is automatically transferred to device with the model
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

        # Check for PyTorch 2.0+ fused scaled dot product attention availability
        self.has_sdpa = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for causal self-attention.

        Args:
            x: Input tensor of shape (Batch_size, Sequence_len, n_embd)

        Returns:
            Output tensor of shape (Batch_size, Sequence_len, n_embd)
        """
        B, T, C = x.size()

        # Step 1: Compute Query, Key, Value representations in a single GEMM
        # Shape: (B, T, 3 * n_embd)
        qkv = self.c_attn(x)

        # Split along embedding dimension into three equal chunks of size (B, T, n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Step 2: Reshape for multi-head attention
        # (B, T, n_head, head_dim) -> transpose to (B, n_head, T, head_dim)
        # Transpose places n_head in the batch dimension so matrix multiplications
        # are performed in parallel across all heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Step 3: Compute attention matrix
        if self.has_sdpa:
            # High-performance PyTorch 2.0+ path (FlashAttention / Memory-Efficient)
            # is_causal=True automatically constructs and applies causal masking
            dropout_p = self.dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        else:
            # Explicit standard scaled dot-product attention fallback
            # (Q @ K.T) / sqrt(d_k) -> shape: (B, n_head, T, T)
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale

            # Apply causal mask: mask out upper triangle (future tokens) with -infinity
            # Softmax will map -inf to exactly 0.0 probability
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # Weighted sum over values: (B, n_head, T, T) @ (B, n_head, T, head_dim)
            # -> (B, n_head, T, head_dim)
            y = att @ v

        # Step 4: Re-assemble head outputs
        # (B, n_head, T, head_dim) -> (B, T, n_head, head_dim) -> (B, T, n_embd)
        # contiguous() is mandatory before view() after transpose()
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Step 5: Output projection + residual dropout
        out = self.resid_dropout(self.c_proj(y))
        return out
