"""
INLA — Inverted Nonlinear Lifting Attention (this paper).

Proposes a learned, nonlinear lifting of the query and key representations
before applying linear attention. The lifting function Φ_INLA uses a
bottleneck (compression → nonlinearity → expansion) to increase the
representational capacity of the feature map while maintaining O(N) complexity.

Core formula:
    Φ_INLA(X) = σ(X @ W_low + b_low) @ W_exp + b_exp
    φ(X)      = ELU(Φ_INLA(X)) + 1        (positivity for linear attention)

    Q_hat = Φ_INLA(Q)   ∈ ℝ^{N × r}
    K_hat = Φ_INLA(K)   ∈ ℝ^{N × r}
    V     = V             (unchanged)

    Attention(Q, K, V) = D^{-1} · φ(Q_hat) @ (φ(K_hat)^T V)

where:
- d_k: bottleneck dimension (compressed)
- r:   expanded feature dimension (r > d_k)
- σ:   non-linear activation (GELU, SiLU, or ReLU)

References: See INLA.tex (this paper).
"""

from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from layers.base_layer import BaseNLPLayer


class INLALifting(nn.Module):
    """
    Φ_INLA(X) — Inverted Nonlinear Lifting Module.

    Architecture: Compression → Nonlinearity → Expansion

        X ∈ ℝ^{d}  →  X @ W_low + b_low  →  σ  →  @ W_exp + b_exp  →  ℝ^{r}

    The bottleneck d_k < d forces information compression; the expansion to
    r > d_k enables rich, high-dimensional feature maps for the linear
    attention kernel.

    Args:
        d: Input dimension.
        d_k: Bottleneck (compressed) dimension. Default: d // 4.
        r: Output (expanded) dimension. Default: 2 * d.
        activation: Non-linearity — 'gelu' (default), 'silu', or 'relu'.
        bias: Whether to use bias in both linear layers.
    """

    def __init__(
        self,
        d: int,
        d_k: Optional[int] = None,
        r: Optional[int] = None,
        activation: str = "gelu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.d = d
        self.d_k = d_k if d_k is not None else max(1, d // 4)
        self.r = r if r is not None else 2 * d

        self.fc_down = nn.Linear(d, self.d_k, bias=bias)
        self.fc_up = nn.Linear(self.d_k, self.r, bias=bias)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu":
            self.activation = F.silu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}. Choose from 'gelu', 'silu', 'relu'.")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., d) Input tensor.

        Returns:
            (..., r) Lifted representation.
        """
        x = self.fc_down(x)          # → d_k
        x = self.activation(x)       # nonlinear
        x = self.fc_up(x)            # → r
        return x

    def init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc_down.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc_up.weight, gain=1.0)
        if self.fc_down.bias is not None:
            nn.init.zeros_(self.fc_down.bias)
        if self.fc_up.bias is not None:
            nn.init.zeros_(self.fc_up.bias)

    def extra_repr(self) -> str:
        return f"d={self.d}, d_k={self.d_k}, r={self.r}"


class INLAAttention(BaseNLPLayer):
    """
    INLA Attention — learns a feature map via non-linear bottleneck lifting.

    Q, K ∈ ℝ^{N × d} → Φ(Q), Φ(K) ∈ ℝ^{N × r}
    V ∈ ℝ^{N × d_v} (unchanged)

    Phase 1 (Context): S = Φ(K)^T V  ∈ ℝ^{r × d_v}
    Phase 2 (Retrieve): O = D^{-1} · Φ(Q) @ S  ∈ ℝ^{N × d_v}

    Complexity: O(N · r · d_v)  — linear in N.

    Args:
        hidden_size: Model dimension (d).
        num_heads: Number of attention heads.
        r: Expanded feature dimension after lifting. Default: 2 * hidden_size / num_heads.
        d_k: Bottleneck dimension. Default: hidden_size // 4 // num_heads.
        activation: Non-linearity for lifting — 'gelu' (default), 'silu', 'relu'.
        dropout: Attention dropout probability.
        bias: Whether to use bias in Q/K/V and lifting projections.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        r: Optional[int] = None,
        d_k: Optional[int] = None,
        activation: str = "gelu",
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

        # INLA lifting: applied per-head to Q and K
        self.q_lifting = INLALifting(
            d=self.head_dim,
            d_k=d_k if d_k is not None else max(1, self.head_dim // 4),
            r=r if r is not None else 2 * self.head_dim,
            activation=activation,
        )
        self.k_lifting = INLALifting(
            d=self.head_dim,
            d_k=d_k if d_k is not None else max(1, self.head_dim // 4),
            r=r if r is not None else 2 * self.head_dim,
            activation=activation,
        )

        # Feature dimension after lifting
        self.r = self.q_lifting.r

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

        # INLA lifting of Q and K (B, H, N, D/H -> B, H, N, r)
        query_f = self.q_lifting(query)
        key_f = self.k_lifting(key)

        # Positivity: linear attention REQUIRES positive feature maps
        # for stable normalization. φ(x) = ELU(x) + 1 follows the same
        # guarantee as Katharopoulos et al. (2020).
        query_f = F.elu(query_f) + 1.0
        key_f = F.elu(key_f) + 1.0

        # Causal masking: zero out future positions' key features
        if attention_mask is not None:
            if attention_mask.dim() == 4 and attention_mask.size(-2) != 1:
                # Full mask (B,1,N,N) -> reduce to (B,1,N,1) to broadcast with (B,H,N,r)
                causal_mask = (attention_mask == 0.0).float()
                key_mask = causal_mask.max(dim=-2, keepdim=True).values  # (B,1,1,N)
                key_mask = key_mask.transpose(-2, -1)                 # (B,1,N,1)
                key_f = key_f * key_mask
            elif attention_mask.dim() == 4:
                causal_mask = (attention_mask == 0.0).float()
                query_f = query_f * causal_mask.transpose(-2, -1)

        # Linear attention: context then retrieve
        # S = K_f^T V : (B, H, r, D/H)
        context = torch.matmul(key_f.transpose(-2, -1), value)

        # O = Q_f @ S : (B, H, N, D/H)
        attn_output = torch.matmul(query_f, context)

        # Normalization: Z = Q_f @ sum(K_f, dim=-2)
        key_sum = key_f.sum(dim=-2).unsqueeze(-2)  # (B, H, 1, r)
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
        self.q_lifting.init_weights()
        self.k_lifting.init_weights()
