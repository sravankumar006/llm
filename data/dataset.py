"""
CommandLLM Dataset Loader
=========================
Memory-efficient PyTorch Dataset implementation for tokenized CommandLLM corpora.
Utilizes memory-mapping (np.load with mmap_mode='r') for zero-copy, low-memory reads.
"""

import os
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class CommandDataset(Dataset):
    """
    Memory-mapped PyTorch Dataset for pre-tokenized inputs and targets.

    Attributes:
        inputs (np.ndarray): Memory-mapped array of shape (N, block_size), dtype=int32.
        targets (np.ndarray): Memory-mapped array of shape (N, block_size), dtype=int32.
        block_size (int): Sequence context window length.
    """

    def __init__(
        self,
        inputs_path: Optional[str] = None,
        targets_path: Optional[str] = None,
        split: str = "train",
        data_dir: str = "data/processed",
        block_size: Optional[int] = None,
    ):
        """
        Initialize the dataset either from explicit file paths or via split and directory.

        Args:
            inputs_path: Direct path to inputs .npy file (e.g. data/processed/train_inputs.npy).
            targets_path: Direct path to targets .npy file (e.g. data/processed/train_targets.npy).
            split: Dataset split ('train' or 'val') if paths are not directly specified.
            data_dir: Root directory for processed .npy artifacts.
            block_size: Context length constraint (optional validation).
        """
        if inputs_path is None:
            inputs_path = os.path.join(data_dir, f"{split}_inputs.npy")
        if targets_path is None:
            targets_path = os.path.join(data_dir, f"{split}_targets.npy")

        if not os.path.isfile(inputs_path):
            raise FileNotFoundError(f"Inputs file not found: {inputs_path}")
        if not os.path.isfile(targets_path):
            raise FileNotFoundError(f"Targets file not found: {targets_path}")

        self.inputs_path = inputs_path
        self.targets_path = targets_path

        # Memory-map numpy arrays for zero-copy, memory-efficient CPU reads
        self.inputs = np.load(inputs_path, mmap_mode="r")
        self.targets = np.load(targets_path, mmap_mode="r")

        assert len(self.inputs) == len(self.targets), (
            f"Inputs count ({len(self.inputs)}) does not match "
            f"targets count ({len(self.targets)})"
        )

        self.block_size = block_size or (self.inputs.shape[1] if self.inputs.ndim > 1 else None)

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve sample at index as 64-bit integer PyTorch tensors.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (input_tensor, target_tensor) with torch.long dtype.
        """
        # Safely convert memory-mapped array slice to int64 torch.Tensor without non-writable buffer warnings
        x = torch.tensor(self.inputs[idx], dtype=torch.long)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y

