import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from layers.base_layer import BaseNLPLayer


class SinusoidalPositionalEncoding(BaseNLPLayer):
    """
    Sinusoidal positional encoding (from "Attention Is All You Need").
    Not learned; fixed sinusoidal functions.

    Args:
        hidden_size: Model dimension.
        max_seq_length: Maximum sequence length.
    """

    def __init__(self, hidden_size: int, max_seq_length: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_seq_length, hidden_size)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_length, hidden_size)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1), :]


class LearnablePositionalEncoding(BaseNLPLayer):
    """
    Learnable absolute positional encoding.

    Args:
        max_seq_length: Maximum sequence length.
        hidden_size: Model dimension.
    """

    def __init__(self, max_seq_length: int, hidden_size: int) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_seq_length, hidden_size))
        nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1), :]


class RotaryPositionalEncoding(BaseNLPLayer):
    """
    Rotary Positional Embedding (RoPE, from "RoFormer").
    Applies rotation to query and key vectors.

    Args:
        head_dim: Dimension per attention head.
        max_seq_length: Maximum sequence length.
        base: Base for the frequency computation.
    """

    def __init__(self, head_dim: int, max_seq_length: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_length = max_seq_length

    def _compute_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        seq_len: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply rotary embeddings to query and key.

        Args:
            query: (batch_size, num_heads, seq_len, head_dim)
            key: (batch_size, num_heads, seq_len, head_dim)

        Returns:
            Rotated query and key.
        """
        if seq_len is None:
            seq_len = query.size(-2)

        cos, sin = self._compute_cos_sin(seq_len, query.device)
        cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

        query = query * cos + self._rotate_half(query) * sin
        key = key * cos + self._rotate_half(key) * sin
        return query, key


class ALiBiPositionalEncoding(BaseNLPLayer):
    """
    Attention with Linear Biases (ALiBi, from "Train Short, Test Long").
    Adds a linear bias to attention scores based on distance between positions.

    Args:
        num_heads: Number of attention heads.
    """

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", slopes, persistent=False)

    @staticmethod
    def _get_slopes(num_heads: int) -> Tensor:
        def get_slopes_power_of_2(n: int) -> list:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(num_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(num_heads))
        else:
            closest_power = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power)
            extra = num_heads - closest_power
            extra_slopes = get_slopes_power_of_2(2 * closest_power)
            slopes += extra_slopes[0::2][:extra]
            return torch.tensor(slopes)

    def forward(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Returns ALiBi bias matrix of shape (1, num_heads, seq_len, seq_len).

        Args:
            seq_len: Sequence length.
            device: Target device.

        Returns:
            Bias tensor to add to attention scores.
        """
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        relative_positions = positions.T - positions  # (seq_len, seq_len)
        alibi = self.slopes.unsqueeze(1).unsqueeze(2) * relative_positions.unsqueeze(0)
        return alibi.unsqueeze(0)  # (1, num_heads, seq_len, seq_len)
