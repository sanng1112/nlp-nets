"""
Utility functions for working with tokenizers.
"""

from typing import List, Optional, Dict, Any
import torch


def pad_sequence(
    sequences: List[torch.Tensor],
    padding_value: int = 0,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Pad a list of variable-length token ID sequences to the same length.

    Args:
        sequences: List of 1D tensors of token IDs.
        padding_value: The padding token ID (default: 0).
        max_length: If provided, pad/truncate to this length.

    Returns:
        Tensor of shape (batch_size, seq_len).
    """
    if max_length is not None:
        sequences = [seq[:max_length] for seq in sequences]
    return torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=padding_value
    )


def create_attention_mask(
    input_ids: torch.Tensor,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """
    Create an attention mask (1 = real token, 0 = padding).

    Args:
        input_ids: Tensor of shape (batch_size, seq_len).
        pad_token_id: Token ID used for padding.

    Returns:
        Attention mask of shape (batch_size, seq_len).
    """
    return (input_ids != pad_token_id).long()


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create a causal (upper-triangular) attention mask for autoregressive models.

    Args:
        seq_len: Sequence length.

    Returns:
        Causal mask of shape (1, 1, seq_len, seq_len).
    """
    return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)
