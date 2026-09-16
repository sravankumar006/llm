"""
CommandLLM Tokenizer Abstraction
================================
Provides a unified Byte-Pair Encoding (BPE) interface built on tiktoken (gpt2 base),
extended with CommandLLM custom special conditioning and delineation tokens.
"""

from typing import Dict, List, Union
import tiktoken


class CommandTokenizer:
    """
    BPE Tokenizer for CommandLLM with custom special token registrations.

    Special Token Specification:
        <|start|>  : Marks beginning of an input conditioning sequence.
        <|os|>     : OS type conditioning tag ('linux' or 'powershell').
        <|prompt|> : Natural language prompt payload indicator.
        <|cmd|>    : Marks the start of the supervised terminal command generation.
        <|end|>    : Marks the completion of the terminal command sequence.
        <|pad|>    : Sequence padding token to reach uniform block_size.
    """

    SPECIAL_TOKENS: Dict[str, int] = {
        "<|start|>": 50257,
        "<|os|>": 50258,
        "<|prompt|>": 50259,
        "<|cmd|>": 50260,
        "<|end|>": 50261,
        "<|pad|>": 50262,
    }

    def __init__(self):
        # 1. Load base GPT-2 BPE tokenizer vocabulary (50,257 mergeable ranks)
        base_enc = tiktoken.get_encoding("gpt2")

        # 2. Register custom special tokens directly into the BPE encoding table
        custom_special_tokens = {**base_enc._special_tokens, **self.SPECIAL_TOKENS}

        self._enc = tiktoken.Encoding(
            name="command_bpe_gpt2",
            pat_str=base_enc._pat_str,
            mergeable_ranks=base_enc._mergeable_ranks,
            special_tokens=custom_special_tokens,
        )

        # Cache frequently accessed token IDs for fast inference lookups
        self.start_token_id = self.SPECIAL_TOKENS["<|start|>"]
        self.os_token_id = self.SPECIAL_TOKENS["<|os|>"]
        self.prompt_token_id = self.SPECIAL_TOKENS["<|prompt|>"]
        self.cmd_token_id = self.SPECIAL_TOKENS["<|cmd|>"]
        self.eos_token_id = self.SPECIAL_TOKENS["<|end|>"]
        self.pad_token_id = self.SPECIAL_TOKENS["<|pad|>"]

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens (50,263)."""
        return self._enc.n_vocab

    def encode(self, text: str, allowed_special: Union[str, set] = "all") -> List[int]:
        """
        Encode text into a list of token IDs.

        Args:
            text: Raw string to tokenize.
            allowed_special: Set of special token strings permitted in the text.
                             Defaults to "all" to allow custom special markers.
        """
        return self._enc.encode(text, allowed_special=allowed_special)

    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token IDs back into text.

        Args:
            tokens: List or sequence of integer token IDs.
        """
        return self._enc.decode(tokens)

    def format_prompt(self, os_type: str, prompt_text: str) -> str:
        """
        Format a natural language request and target OS into the canonical prompt format.

        Canonical format:
            <|start|><|os|>{os_type}<|prompt|>{prompt_text}<|cmd|>

        Args:
            os_type: Target operating system ('linux' or 'powershell').
            prompt_text: User query or instruction.

        Returns:
            Formatted prompt string ending with <|cmd|>, ready for causal completion.
        """
        os_clean = os_type.strip().lower()
        p_clean = prompt_text.strip()
        return f"<|start|><|os|>{os_clean}<|prompt|>{p_clean}<|cmd|>"
