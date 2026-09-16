"""
CommandLLM Model Configuration Specification
============================================
Defines the hyperparameter contract and structural specifications for CoreCommandLLM.
"""

from dataclasses import dataclass


@dataclass
class CommandLMConfig:
    """
    Hyperparameter specification for the CoreCommandLLM Transformer.

    Attributes:
        vocab_size: Total vocabulary size.
                    Base GPT-2 tokenizer has 50,257 tokens (0 to 50256).
                    With 6 custom special tokens (<|start|>, <|os|>, <|prompt|>, <|cmd|>,
                    <|end|>, <|pad|>) mapped to 50257..50262, the required embedding
                    table size is 50,263 so index 50262 is in-bounds.
        block_size: Maximum sequence context length (context window T).
        n_layer: Number of decoder Transformer blocks.
        n_head: Number of causal multi-head attention heads.
        n_embd: Hidden dimensionality of token embeddings and residual stream.
        dropout: Dropout rate applied after embeddings, attention matrices, and MLP projections.
        bias: Whether linear layers and LayerNorms include learnable additive bias terms.
              Setting bias=True matches original GPT-2 architecture.
    """
    vocab_size: int = 50263
    block_size: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self):
        # Validate that embedding dimension is evenly divisible across attention heads
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"Embedding dimension n_embd ({self.n_embd}) must be divisible "
                f"by number of attention heads n_head ({self.n_head}). "
                f"Head dimension would otherwise be fractional: {self.n_embd / self.n_head}."
            )


# Alias for backwards compatibility with earlier codebase references
TransformerConfig = CommandLMConfig
