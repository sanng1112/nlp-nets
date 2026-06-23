"""
Performer Attention from "Rethinking Attention with Performers" (Choromanski et al., 2020).

Implements FAVOR+ (Fast Attention Via Orthogonal Random features) mechanism
for unbiased estimation of softmax attention in linear time.

Key ideas:
- Random feature map φ(x) = 1/√m * exp(w_i^T x - ||x||²/2) for i=1..m
- Orthogonal random features reduce estimation variance
- Enables O(N) complexity while approximating softmax attention

Reference: https://arxiv.org/abs/2009.14794
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from layers.base_layer import BaseNLPLayer


class PerformerAttention(BaseNLPLayer):
    """
    Performer FAVOR+ attention with orthogonal random features.

    Args:
        hidden_size: Model dimension.
        num_heads: Number of attention heads.
        num_random_features: Number of random features (default: head_dim * 2).
            Larger values give better approximation but higher compute.
        kernel_type: 'softmax' (default) or 'relu'.
        softmax_temp: Temperature for softmax kernel (default: 1/√(head_dim)).
        dropout: Attention dropout probability.
        bias: Whether to use bias in Q/K/V linear projections.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_random_features: Optional[int] = None,
        kernel_type: str = "softmax",
        softmax_temp: Optional[float] = None,
        dropout: float = 0.0,
        bias: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        assert kernel_type in ("softmax", "relu"), f"kernel_type must be 'softmax' or 'relu', got {kernel_type}"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_random_features = num_random_features or self.head_dim * 2
        self.kernel_type = kernel_type
        self.softmax_temp = softmax_temp if softmax_temp is not None else self.head_dim ** -0.5
        self.eps = eps

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # Orthogonal random matrix (per-head, shared across heads if tied)
        self.register_buffer(
            "random_matrix",
            self._generate_orthogonal_matrix(self.head_dim, self.num_random_features),
        )

    def _generate_orthogonal_matrix(self, head_dim: int, m: int) -> Tensor:
        """
        Generate orthogonal random matrix for FAVOR+ variance reduction.

        Uses QR decomposition on a Gaussian matrix.
        If m > head_dim, tiles orthogonal blocks.
        """
        # Number of orthogonal blocks needed
        num_blocks = (m + head_dim - 1) // head_dim
        blocks = []
        for _ in range(num_blocks):
            w = torch.randn(head_dim, head_dim)
            q, _ = torch.linalg.qr(w)
            blocks.append(q)
        # Concatenate and truncate to exact m columns
        w_ortho = torch.cat(blocks, dim=-1)[:, :m]
        return w_ortho  # (head_dim, m)

    def _feature_map(self, x: Tensor) -> Tensor:
        """
        Compute FAVOR+ random features.

        For softmax kernel:
            φ(x) = 1/√m * exp(w^T x - ||x||²/2)
        For relu kernel:
            φ(x) = 1/√m * max(0, w^T x)³  (cubic ReLU)
        """
        batch_size, num_heads, seq_len, head_dim = x.size()
        device = x.device

        # Ensure random matrix is on the correct device
        w = self.random_matrix.to(device=device, dtype=x.dtype)  # (head_dim, m)
        m = self.num_random_features

        if self.kernel_type == "softmax":
            # exp(w^T x) / sqrt(m) where x is normalized by -||x||²/2
            # First normalize x: w^T (x / σ) — softmax kernel uses temperature
            x_normalized = x * self.softmax_temp
            # Project: (B, H, N, d) @ (d, m) -> (B, H, N, m)
            wx = torch.matmul(x_normalized, w)
            # Subtract ||x||²/2 for proper softmax kernel
            x_norm_sq = (x_normalized ** 2).sum(dim=-1, keepdim=True)  # (B, H, N, 1)
            phi = torch.exp(wx - x_norm_sq / 2.0) / math.sqrt(m)
        else:
            # Cubic ReLU kernel
            wx = torch.matmul(x, w)
            # (max(0, wx))³ / sqrt(m)
            phi = (F.relu(wx) ** 3) / math.sqrt(m)

        return phi  # (B, H, N, m)

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
            attention_mask: (batch_size, 1, seq_len, seq_len) with -inf for masked positions.
                Used for causal masking via feature-space zeroing.
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

        # Apply feature map
        query_f = self._feature_map(query)  # (B, H, N, m)
        key_f = self._feature_map(key)

        # Causal masking: zero out future positions' key features
        if attention_mask is not None:
            if attention_mask.dim() == 4 and attention_mask.size(-2) != 1:
                # Full mask (B, 1, N, N) — reduce to key-level visibility
                causal_mask = (attention_mask == 0.0).float()
                key_mask = causal_mask.max(dim=-2, keepdim=True).values  # (B, 1, 1, N)
                key_mask = key_mask.transpose(-2, -1)  # (B, 1, N, 1)
                key_f = key_f * key_mask  # (B, H, N, m) * (B, 1, N, 1)
            elif attention_mask.dim() == 4:
                causal_mask = (attention_mask == 0.0).float()
                query_f = query_f * causal_mask.transpose(-2, -1)

        # Context: S = K_f^T V : (B, H, m, D/H)
        context = torch.matmul(key_f.transpose(-2, -1), value)

        # Retrieve: O = Q_f @ S : (B, H, N, D/H)
        attn_output = torch.matmul(query_f, context)

        # Normalize: Z = Q_f @ sum(K_f, dim=-2)
        key_sum = key_f.sum(dim=-2).unsqueeze(-2)  # (B, H, 1, m)
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
