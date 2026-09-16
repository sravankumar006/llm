"""
CoreCommandLLM: Custom Decoder Transformer Language Model from Scratch
=======================================================================
Implements the complete causal Transformer architecture in raw PyTorch (torch.nn.Module).
Features:
- Learned token and positional embeddings
- Weight tying between token embeddings and the LM projection head
- Pre-LayerNorm Transformer blocks with causal self-attention and MLP
- Selective CrossEntropyLoss with ignore_index=-100 support
- Autoregressive text generation with temperature scaling and top-k filtering
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.config import CommandLMConfig
from core.layers import TransformerBlock


class CoreCommandLLM(nn.Module):
    """
    Custom Autoregressive Transformer Language Model built from scratch.

    Key Architectural Principles:
    1. Pure PyTorch Implementation:
       No high-level wrappers or third-party abstractions. Every layer, tensor
       transformation, loss calculation, and sampling step is explicit.

    2. Weight Tying (Press & Wolf, 2017):
       The input token embedding matrix (wte.weight) is shared directly with the
       final language model output projection matrix (lm_head.weight).
       - Theory: Token embeddings project discrete token IDs into dense semantic
         vectors. The LM head performs the dual operation: mapping dense vectors
         back to token probabilities via dot-product.
       - Efficiency: Sharing weights eliminates 50,263 * 768 (~38.6 million)
         redundant parameters, dramatically saving VRAM and boosting generalization.

    3. Selective Loss Masking:
       Supports ignore_index=-100 in F.cross_entropy. In our CommandLLM pipeline,
       only the generated terminal command ({cmd}) and <|end|> contribute to the
       loss. Conditioning prompts and OS labels are ignored.

    4. Scaled Residual Initialization:
       Projections that feed back into the residual stream (c_proj in attention
       and MLP) are initialized with standard deviation scaled down by
       1 / sqrt(2 * n_layer) to prevent the residual variance from exploding as
       depth increases.
    """

    def __init__(self, config: CommandLMConfig):
        super().__init__()
        self.config = config

        # Hierarchical module container for clean state_dict serialization
        self.transformer = nn.ModuleDict(
            dict(
                # Token Embedding table: (vocab_size, n_embd)
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # Learned Positional Embedding table: (block_size, n_embd)
                wpe=nn.Embedding(config.block_size, config.n_embd),
                # Embedding dropout
                drop=nn.Dropout(config.dropout),
                # Stack of N Pre-LayerNorm Transformer decoder blocks
                h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                # Final LayerNorm applied after all blocks before the output projection
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        # Output projection head: maps residual representations (n_embd) to vocabulary logits (vocab_size)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying: bind the embedding matrix to the output linear layer
        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all model weights according to GPT-2 specification
        self.apply(self._init_weights)

        # Apply special scaled initialization to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module):
        """
        GPT-2 standard weight initialization:
        - Linear layers: N(0, 0.02)
        - Embedding tables: N(0, 0.02)
        - Bias vectors: zeros
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return total number of parameters in the model.
        Args:
            non_embedding: If True, subtract position embeddings to report
                           computational parameter count (weight-tied wte is
                           counted once as part of lm_head).
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for training and inference.

        Args:
            idx: Tensor of token indices with shape (Batch_size, Sequence_len)
            targets: Optional tensor of target token indices with shape (Batch_size, Sequence_len).
                     Tokens with value -100 are ignored in the CrossEntropyLoss computation.

        Returns:
            logits: Unnormalized token scores of shape (Batch_size, Sequence_len, vocab_size)
            loss: Scalar cross-entropy loss if targets are provided, otherwise None.
        """
        device = idx.device
        b, t = idx.size()

        # Ensure input sequence length does not exceed maximum context window
        if t > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {t}, block_size is {self.config.block_size}"
            )

        # Construct position IDs: [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # 1. Look up token embeddings: (B, T) -> (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)

        # 2. Look up positional embeddings: (T,) -> (T, n_embd)
        pos_emb = self.transformer.wpe(pos)

        # 3. Sum token + positional embeddings and apply dropout
        x = self.transformer.drop(tok_emb + pos_emb)

        # 4. Sequentially process through all Transformer decoder blocks
        for block in self.transformer.h:
            x = block(x)

        # 5. Apply final LayerNorm
        x = self.transformer.ln_f(x)

        # 6. Compute Logits and Loss
        if targets is not None:
            # During training, compute logits across the full sequence for parallel cross-entropy
            logits = self.lm_head(x)  # Shape: (B, T, vocab_size)

            # Shift logits and targets so token at index t predicts target at t+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()

            # Flatten batch and sequence dimensions for CrossEntropyLoss
            # shift_logits: (B * (T - 1), vocab_size)
            # shift_targets: (B * (T - 1),)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,  # Explicitly ignores conditioning prompt tokens
            )
        else:
            # During inference generation, only the logits at the very last token position are needed
            # Shape: (B, 1, vocab_size)
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive causal generation loop.

        Args:
            idx: Conditioning prompt tensor of token IDs with shape (B, T)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = standard, < 1.0 = more deterministic, > 1.0 = more creative)
            top_k: If set, restricts sampling to the top-K highest probability tokens
            eos_token_id: If generated, early termination is triggered for completed sequences

        Returns:
            Tensor of shape (B, T + generated_tokens) containing the full sequence.
        """
        self.eval()

        for _ in range(max_new_tokens):
            # If the context sequence exceeds block_size, crop to the most recent block_size tokens
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )

            # Forward the model to obtain logits for the last token position
            logits, _ = self(idx_cond)
            # Squeeze sequence dimension: (B, 1, vocab_size) -> (B, vocab_size)
            logits = logits[:, -1, :]

            # Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # Apply optional Top-K filtering: keep only the top_k logits, mask rest with -inf
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # Mask everything lower than the smallest value in the top-k
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # Convert logits to normalized probabilities via Softmax
                probs = F.softmax(logits, dim=-1)
                # Sample the next token from the probability distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # Deterministic Greedy Argmax decoding (temperature <= 0.0)
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            # Append the sampled token to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # Check if all batch elements generated the end-of-sequence token
            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break

        return idx


# Alias for backwards compatibility
CommandLLM = CoreCommandLLM


# ==============================================================================
# Verification and Self-Test Sanity Check
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" CoreCommandLLM Architecture Verification & Sanity Check")
    print("=" * 70)

    # Instantiate a test configuration with smaller dimensions for rapid verification
    test_config = CommandLMConfig(
        vocab_size=50263,
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=True,
    )
    print(f"Instantiating test model with config:\n{test_config}\n")
    model = CoreCommandLLM(test_config)

    # 1. Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    non_emb_params = model.get_num_params(non_embedding=True)
    print(f"Total Parameters          : {total_params:,}")
    print(f"Non-Embedding Parameters  : {non_emb_params:,}")

    # 2. Verify Weight Tying
    is_tied = model.transformer.wte.weight is model.lm_head.weight
    print(f"Weight Tying Verified     : {is_tied} (wte.weight IS lm_head.weight)")
    assert is_tied, "Error: Weight tying between wte and lm_head failed!"

    # 3. Test Forward Pass (Batch=2, SeqLen=32)
    B, T = 2, 32
    dummy_input = torch.randint(0, 1000, (B, T), dtype=torch.long)
    dummy_targets = torch.randint(0, 1000, (B, T), dtype=torch.long)
    # Mask half the targets with -100 to simulate prompt masking
    dummy_targets[:, :16] = -100

    print(f"\nRunning Forward Pass with Input Shape {tuple(dummy_input.shape)} and Targets...")
    logits, loss = model(dummy_input, targets=dummy_targets)

    expected_logits_shape = (B, T, test_config.vocab_size)
    print(f"Output Logits Shape       : {tuple(logits.shape)} (Expected: {expected_logits_shape})")
    assert logits.shape == expected_logits_shape, f"Logits shape mismatch: {logits.shape}"

    print(f"Computed Loss             : {loss.item():.4f}")
    assert loss is not None and loss.item() > 0, "Loss calculation failed or returned non-positive."

    # 4. Test Autoregressive Generation
    prompt = torch.tensor([[50257, 50258, 200, 50259]], dtype=torch.long)  # Toy prefix
    print(f"\nTesting Autoregressive Generation with Prompt Shape {tuple(prompt.shape)}...")
    generated = model.generate(prompt, max_new_tokens=8, temperature=0.8, top_k=50)
    print(f"Generated Output Shape    : {tuple(generated.shape)} (Prompt: 4 -> Total: {generated.shape[1]})")
    assert generated.shape == (1, 12), f"Generation shape mismatch: {generated.shape}"

    print("\n" + "=" * 70)
    print(" [SUCCESS] All CoreCommandLLM checks passed with zero errors!")
    print("=" * 70)
