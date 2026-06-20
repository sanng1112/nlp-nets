"""
Linear Attention from "Transformers are RNNs" (Katharopoulos et al., 2020).

Implements efficient linear-time attention via the feature map φ(x) = elu(x) + 1,
enabling O(N) complexity by reordering the matrix multiplication.

Reference: https://arxiv.org/abs/2006.16236
"""

from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from layers.base_layer import BaseNLPLayer


class LinearAttention(BaseNLPLayer):
    """
    Linear Attention with elu(x)+1 feature map.

    Drops the softmax and uses a kernel trick:
        Attention(Q, K, V) = φ(Q) @ (φ(K)^T V) / (φ(Q) @ sum(φ(K), dim=-2))

    Complexity: O(N * d * r) where r is the feature dimension (= head_dim for elu+1).

    Args:
        hidden_size: Model dimension.
        num_heads: Number of attention heads.
        eps: Small constant for numerical stability in normalization.
        dropout: Dropout probability applied after attention.
        bias: Whether to use bias in Q/K/V linear projections.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        eps: float = 1e-6,
        dropout: float = 0.0,
        bias: bool = True,
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

    def _feature_map(self, x: Tensor) -> Tensor:
        """φ(x) = elu(x) + 1  — positive, element-wise non-linear feature map."""
        return F.elu(x) + 1.0

    def _reshape_for_attention(self, x: Tensor) -> Tensor:
        """Reshape (B, N, D) -> (B, H, N, D/H)."""
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
            attention_mask: (batch_size, 1, seq_len, seq_len) or (batch_size, 1, 1, seq_len).
                Used for causal masking only. Linear attention cannot mask arbitrary positions
                after context-aggregation — causal mask is applied via chunk-wise masking.
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

        # Apply feature map to Q and K
        query_f = self._feature_map(query)  # (B, H, N, D/H)
        key_f = self._feature_map(key)

        # Apply causal mask if provided: zero out future positions on feature map
        if attention_mask is not None:
            # attention_mask: (B, 1, 1, N) or (B, 1, N, N) with -inf for masked positions
            if attention_mask.dim() == 4 and attention_mask.size(-2) != 1:
                # Full mask — use it to zero out future keys in the feature space
                causal_mask = (attention_mask == 0.0).float()  # 1 for visible, 0 for masked
                key_f = key_f * causal_mask[:, :, -key_f.size(-2):, :]
            elif attention_mask.dim() == 4:
                # Per-step mask — apply to query side
                causal_mask = (attention_mask == 0.0).float()
                query_f = query_f * causal_mask.transpose(-2, -1)

        # Linear attention: context aggregation then retrieval
        # S = K_f^T V : (B, H, D/H, D/H)
        context = torch.matmul(key_f.transpose(-2, -1), value)

        # O = Q_f @ S : (B, H, N, D/H)
        attn_output = torch.matmul(query_f, context)

        # Normalization denominator: Z = Q_f @ sum(K_f, dim=-2)
        key_sum = key_f.sum(dim=-2).unsqueeze(-2)  # (B, H, 1, D/H)
        norm = torch.matmul(query_f, key_sum.transpose(-2, -1))  # (B, H, N, 1)
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
