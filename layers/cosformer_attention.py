"""
CosFormer Attention from "CosFormer: Rethinking Softmax in Attention" (Qin et al., 2022).

Combines a ReLU-based feature map with cosine re-weighting to incorporate
positional locality bias into linear attention.

Key ideas:
- φ(x) = ReLU(x)  (non-negative, element-wise)
- Cosine re-weighting: w(i) = cos(π/2 · i/N) applied per position
- Attention ≈ (φ(Q) ⊙ W_Q) @ ((φ(K) ⊙ W_K)^T V)  where W are position weights
- Maintains O(N) complexity while approaching full-attention quality

Reference: https://arxiv.org/abs/2202.08791
"""

from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from layers.base_layer import BaseNLPLayer


class CosFormerAttention(BaseNLPLayer):
    """
    CosFormer linear attention with ReLU feature map and cosine re-weighting.

    Args:
        hidden_size: Model dimension.
        num_heads: Number of attention heads.
        max_seq_length: Maximum sequence length for pre-computing position weights.
        dropout: Attention dropout probability.
        bias: Whether to use bias in Q/K/V linear projections.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_seq_length: int = 2048,
        dropout: float = 0.0,
        bias: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # Pre-compute cosine weights for all positions up to max_seq_length
        pos = torch.arange(max_seq_length, dtype=torch.float32)
        cos_weights = torch.cos(torch.pi / 2.0 * pos / max_seq_length)
        self.register_buffer("cos_weights", cos_weights)  # (max_seq_length,)

    def _reshape_for_attention(self, x: Tensor) -> Tensor:
        """Reshape (B, N, D) -> (B, H, N, D/H)."""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _apply_cosine_weight(self, x: Tensor, seq_len: int) -> Tensor:
        """
        Apply cosine position weight along the sequence dimension.

        x: (B, H, N, d)
        Returns: (B, H, N, d) with cos(π/2 * i/N) applied per position.
        """
        weights = self.cos_weights[:seq_len].view(1, 1, seq_len, 1)  # (1, 1, N, 1)
        return x * weights.to(x.dtype)

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
            attention_mask: (batch_size, 1, seq_len, seq_len) with -inf for masked positions.
            key_value_states: Optional separate source for keys/values (cross-attention).
            past_key_value: Cached (key, value) for incremental decoding.
            use_cache: Whether to return updated key/value cache.

        Returns:
            output: (batch_size, seq_len, hidden_size)
            present_key_value: Optional (key, value) cache.
        """
        batch_size, seq_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        if key_value_states is not None:
            key = self.k_proj(key_value_states)
            value = self.v_proj(key_value_states)
        else:
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)

        # Reshape for multi-head
        query = self._reshape_for_attention(query)  # (B, H, N, D/H)
        key = self._reshape_for_attention(key)
        value = self._reshape_for_attention(value)

        # KV cache
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=-2)
            value = torch.cat([past_key_value[1], value], dim=-2)
        present_key_value = (key, value) if use_cache else None

        # ReLU feature map
        query_f = F.relu(query)   # (B, H, N, D/H)
        key_f = F.relu(key)

        # Apply cosine re-weighting
        query_f = self._apply_cosine_weight(query_f, seq_len)
        key_f = self._apply_cosine_weight(key_f, seq_len)

        # Causal masking: zero out future positions' key features
        if attention_mask is not None:
            if attention_mask.dim() == 4 and attention_mask.size(-2) != 1:
                causal_mask = (attention_mask == 0.0).float()
                key_f = key_f * causal_mask[:, :, -key_f.size(-2):, :]
            elif attention_mask.dim() == 4:
                causal_mask = (attention_mask == 0.0).float()
                query_f = query_f * causal_mask.transpose(-2, -1)

        # Linear attention with context-retrieve
        # S = K_f^T V  (chunking cos weight into K_f)
        context = torch.matmul(key_f.transpose(-2, -1), value)  # (B, H, D/H, D/H)

        # O = Q_f @ S
        attn_output = torch.matmul(query_f, context)  # (B, H, N, D/H)

        # Normalization
        key_sum = key_f.sum(dim=-2).unsqueeze(-2)
        norm = torch.matmul(query_f, key_sum.transpose(-2, -1))
        attn_output = attn_output / (norm + self.eps)

        attn_output = self.dropout(attn_output)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.hidden_size)
        output = self.out_proj(attn_output)

        return output, present_key_value

    def init_weights(self) -> None:
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1.0)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
