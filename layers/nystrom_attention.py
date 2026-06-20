"""
Nyströmformer Attention from "Nyströmformer: A Nyström-Based Algorithm for
Approximating Self-Attention" (Xiong et al., 2021).

Uses the Nyström approximation to approximate the softmax attention matrix
via landmark (anchor) points, achieving O(N) complexity.

Key idea:
    softmax(QK^T / √d) ≈ softmax(QM^T / √d) @ softmax(KM^T / √d)^+ @ softmax(KM^T / √d)
where M denotes landmark points (learned or sampled from Q and K).

Reference: https://arxiv.org/abs/2102.03902
"""

from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from layers.base_layer import BaseNLPLayer


class NystromAttention(BaseNLPLayer):
    """
    Nyströmformer-style approximate self-attention.

    Approximates the full N×N attention matrix using num_landmarks anchor points,
    reducing complexity from O(N²) to O(N × num_landmarks).

    Args:
        hidden_size: Model dimension.
        num_heads: Number of attention heads.
        num_landmarks: Number of Nyström landmark points (default: 32).
            Higher values give better approximation but increase cost.
        landmark_pool_ratio: If > 0, pool ratio of sequence for landmarks (alternative to fixed count).
        dropout: Attention dropout probability.
        bias: Whether to use bias in Q/K/V linear projections.
        eps: Small constant for numerical stability in pseudo-inverse.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_landmarks: int = 32,
        landmark_pool_ratio: Optional[float] = None,
        dropout: float = 0.0,
        bias: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_landmarks = num_landmarks
        self.landmark_pool_ratio = landmark_pool_ratio
        self.eps = eps

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _reshape_for_attention(self, x: Tensor) -> Tensor:
        """Reshape (B, N, D) -> (B, H, N, D/H)."""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _sample_landmarks(self, x: Tensor) -> Tensor:
        """
        Sample landmark points from input via average pooling.

        If seq_len <= num_landmarks, return the input as-is (no pooling needed).
        Otherwise, use adaptive average pooling to reduce N -> num_landmarks.
        """
        batch_size, num_heads, seq_len, head_dim = x.size()

        if seq_len <= self.num_landmarks:
            return x

        # Reshape for pooling: (B*H, 1, N, D/H) -> avg pool -> (B*H, 1, L, D/H)
        x_reshaped = x.reshape(batch_size * num_heads, 1, seq_len, head_dim)
        pooled = F.adaptive_avg_pool2d(x_reshaped, (self.num_landmarks, head_dim))
        return pooled.reshape(batch_size, num_heads, self.num_landmarks, head_dim)

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

        # Nyström approximation
        # 1. Sample landmarks from Q and K
        q_landmarks = self._sample_landmarks(query)  # (B, H, L, D/H)
        k_landmarks = self._sample_landmarks(key)    # (B, H, L, D/H)

        # 2. Compute three attention matrices
        # K1 = softmax(Q @ M_K^T / √d)  — query-landmark attention
        attn_qm = torch.matmul(query, k_landmarks.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn_qm = attn_qm + F.interpolate(
                attention_mask[:, :, -1:, :], size=(seq_len, k_landmarks.size(-2)), mode="nearest"
            )
        attn_qm = F.softmax(attn_qm, dim=-1, dtype=torch.float32).to(query.dtype)

        # K2 = softmax(M_Q @ K^T / √d)  — landmark-key attention
        attn_mk = torch.matmul(q_landmarks, key.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn_mk = attn_mk + F.interpolate(
                attention_mask[:, :, -1:, :], size=(q_landmarks.size(-2), seq_len), mode="nearest"
            )
        attn_mk = F.softmax(attn_mk, dim=-1, dtype=torch.float32).to(query.dtype)

        # K3 = softmax(M_Q @ M_K^T / √d)  — landmark-landmark attention (for pseudo-inverse)
        attn_mm = torch.matmul(q_landmarks, k_landmarks.transpose(-2, -1)) * self.scale
        attn_mm = F.softmax(attn_mm, dim=-1, dtype=torch.float32).to(query.dtype)

        # 3. Nyström approximation of attention
        # A ≈ K1 @ (K3)+ @ K2
        # where (K3)+ is the Moore-Penrose pseudo-inverse

        # Compute pseudo-inverse via iterative approximation (more stable than direct inverse)
        attn_mm_inv = self._pseudo_inverse(attn_mm)

        # A @ V ≈ K1 @ attn_mm_inv @ (K2 @ V)
        attn_output = torch.matmul(attn_mk, value)                     # (B, H, L, D/H)
        attn_output = torch.matmul(attn_mm_inv, attn_output)            # (B, H, L, D/H)
        attn_output = torch.matmul(attn_qm, attn_output)                # (B, H, N, D/H)

        attn_output = self.dropout(attn_output)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.hidden_size)
        output = self.out_proj(attn_output)

        return output, present_key_value

    @staticmethod
    def _pseudo_inverse(x: Tensor, num_iter: int = 6) -> Tensor:
        """
        Compute Moore-Penrose pseudo-inverse via iterative method.

        Uses Newton-Schulz iteration for stability on GPU:
            V_{k+1} = V_k (2I - A V_k)
        where V_0 = A^T / ||A||²_F
        """
        dim = x.size(-1)
        # Initialize V = A^T / (||A||²_F)
        norm_sq = (x ** 2).sum(dim=(-2, -1), keepdim=True)
        v = x.transpose(-2, -1) / (norm_sq + 1e-10)

        # Newton-Schulz iterations
        for _ in range(num_iter):
            v = v @ (2 * torch.eye(dim, device=x.device, dtype=x.dtype) - x @ v)

        return v

    def init_weights(self) -> None:
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1.0)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
