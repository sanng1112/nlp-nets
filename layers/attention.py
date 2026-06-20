import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from layers.base_layer import BaseNLPLayer


class MultiHeadAttention(BaseNLPLayer):
    """
    Multi-Head Attention (from "Attention Is All You Need").

    Args:
        hidden_size: Model dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout probability.
        bias: Whether to use bias in linear projections.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _reshape_for_attention(self, x: Tensor) -> Tensor:
        """Reshape (batch_size, seq_len, hidden_size) -> (batch_size, num_heads, seq_len, head_dim)."""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        key_value_states: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, 1, seq_len, seq_len) or (batch_size, 1, 1, seq_len) with -inf for masked positions
            key_value_states: Optional separate source for keys/values (cross-attention)
            past_key_value: Cached (key, value) tensors for fast decoding
            use_cache: Whether to return updated key/value cache

        Returns:
            output: (batch_size, seq_len, hidden_size)
            present_key_value: Optional updated cache
        """
        query = self.q_proj(hidden_states)

        if key_value_states is not None:
            key = self.k_proj(key_value_states)
            value = self.v_proj(key_value_states)
        else:
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query = self._reshape_for_attention(query)   # (B, H, Q_len, D)
        key = self._reshape_for_attention(key)       # (B, H, K_len, D)
        value = self._reshape_for_attention(value)   # (B, H, K_len, D)

        # KV cache for incremental decoding
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=-2)
            value = torch.cat([past_key_value[1], value], dim=-2)

        present_key_value = (key, value) if use_cache else None

        # Scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(attn_output.size(0), -1, self.hidden_size)
        output = self.out_proj(attn_output)

        return output, present_key_value

    def init_weights(self) -> None:
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1.0)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)


class SelfAttention(MultiHeadAttention):
    """
    Self-attention convenience wrapper (same as MultiHeadAttention with no cross-attention).
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            key_value_states=None,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
