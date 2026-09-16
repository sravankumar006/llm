"""
Transformer Feed-Forward Network (MLP) and Decoder Block
========================================================
Implements the multi-layer perceptron (MLP) sub-layer and the standard
Pre-LayerNorm Transformer Block for CoreCommandLLM.
"""

import torch
import torch.nn as nn
from core.config import CommandLMConfig
from core.attention import CausalSelfAttention


class MLP(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN / MLP).

    Architectural Mechanics:
    1. Expansion Projection:
       Expands hidden dimension by 4x (n_embd -> 4 * n_embd). This canonical
       4x expansion allows the network to project token representations into
       a high-dimensional space where complex semantic and grammatical
       features can be disentangled and non-linearly combined.

    2. GELU Non-Linearity:
       Uses the Gaussian Error Linear Unit with tanh approximation:
       GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
       Unlike standard ReLU, GELU provides continuous, smooth probabilistic
       gating, preventing dead neuron zones and improving convergence.

    3. Contraction Projection:
       Projects the 4 * n_embd intermediate representations back to n_embd
       to be added back into the residual stream.
    """

    def __init__(self, config: CommandLMConfig):
        super().__init__()
        # Expansion layer: (B, T, n_embd) -> (B, T, 4 * n_embd)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)

        # Activation: smooth approximate GELU matching GPT-2 specification
        self.gelu = nn.GELU(approximate="tanh")

        # Contraction layer: (B, T, 4 * n_embd) -> (B, T, n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        # Dropout on projection output before addition to residual stream
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP sub-layer.

        Args:
            x: Input tensor of shape (B, T, n_embd)

        Returns:
            Output tensor of shape (B, T, n_embd)
        """
        h = self.c_fc(x)
        h = self.gelu(h)
        h = self.c_proj(h)
        out = self.dropout(h)
        return out


class TransformerBlock(nn.Module):
    """
    Transformer Decoder Block with Pre-LayerNorm formulation.

    Architectural Mechanics (Pre-LN vs Post-LN):
    In the original Vaswani et al. 2017 Transformer (Post-LN), LayerNorm was applied
    after the residual addition: x = LN(x + SubLayer(x)). This caused gradient
    magnitudes to decay or explode with depth, necessitating delicate learning rate
    warmups.

    In modern architectures (GPT-2, LLaMA, GPT-3), Pre-LN is adopted:
        x = x + Attention(LN_1(x))
        x = x + MLP(LN_2(x))

    Why Pre-LN is superior:
    - The residual stream acts as an uninhibited identity highway where gradients
      flow directly back through all layers without attenuation.
    - Each sub-layer only calculates an additive "delta" or refinement to the
      running residual state.
    - Ensures numerical stability even in deep architectures (12+ layers).
    """

    def __init__(self, config: CommandLMConfig):
        super().__init__()
        # First LayerNorm: normalizes inputs prior to multi-head causal self-attention
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Causal Self-Attention mechanism
        self.attn = CausalSelfAttention(config)

        # Second LayerNorm: normalizes intermediate residual state prior to MLP
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Feed-Forward Network
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through one Transformer block.

        Args:
            x: Residual stream tensor of shape (B, T, n_embd)

        Returns:
            Updated residual stream tensor of shape (B, T, n_embd)
        """
        # Sub-layer 1: Pre-LN Causal Attention + Residual Addition
        x = x + self.attn(self.ln_1(x))

        # Sub-layer 2: Pre-LN MLP + Residual Addition
        x = x + self.mlp(self.ln_2(x))

        return x
